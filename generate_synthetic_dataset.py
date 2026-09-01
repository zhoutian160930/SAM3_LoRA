#!/usr/bin/env python3
"""Copy-Paste 合成数据生成器：从少量 labelme 标注样本生成密集堆叠训练数据。

用法:
    python generate_synthetic_dataset.py \
        --src 数据集生成_原始数据/长龙航空餐具 \
        --out 数据集生成_生成数据 \
        --num 200 --k-min 4 --k-max 14 --rotate 20 --seed 42

    # 查看 preview/ 满意后合并进训练集:
    python generate_synthetic_dataset.py --merge-to dataset/长龙航空餐具/train \
        --out 数据集生成_生成数据

流程:
    sample/(labelme) 抠 RGBA 素材 -> 贴到 bg/ 背景上(密集高斯聚集, z-order 堆叠)
    -> 每实例精确 mask 自动生成 COCO GT(RLE/bbox/area)
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import cv2
import pycocotools.mask as mask_utils
from PIL import Image, ImageDraw, ImageEnhance

CATEGORY = {"id": 1, "name": "obj_0", "supercategory": "none"}


# ----------------------------------------------------------------------------
# 素材构建
# ----------------------------------------------------------------------------

def build_cutouts(sample_dir: Path):
    """从 labelme 标注抠出 RGBA 素材列表。"""
    cutouts = []
    for jf in sorted(sample_dir.glob("*.json")):
        d = json.loads(jf.read_text())
        W, H = d["imageWidth"], d["imageHeight"]
        img = Image.open(jf.with_suffix(".jpg")).convert("RGB")

        for si, shape in enumerate(d["shapes"]):
            if shape["shape_type"] != "polygon" or len(shape["points"]) < 3:
                continue
            mask_img = Image.new("L", (W, H), 0)
            ImageDraw.Draw(mask_img).polygon(
                [tuple(p) for p in shape["points"]], fill=255)
            mask = np.asarray(mask_img) > 127

            xs = [p[0] for p in shape["points"]]
            ys = [p[1] for p in shape["points"]]
            x0, y0 = int(max(0, min(xs))), int(max(0, min(ys)))
            x1, y1 = int(min(W, max(xs))), int(min(H, max(ys)))
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue

            crop = np.asarray(img)[y0:y1, x0:x1]
            m = mask[y0:y1, x0:x1]
            rgba = np.dstack([crop, m.astype(np.uint8) * 255])
            cutouts.append({
                "img": Image.fromarray(rgba, mode="RGBA"),
                "w": x1 - x0, "h": y1 - y0,
                "src": f"{jf.stem}#{si}",
            })
    return cutouts


# ----------------------------------------------------------------------------
# 背景处理
# ----------------------------------------------------------------------------

def make_background(bg_paths, rng):
    bg = Image.open(rng.choice(bg_paths)).convert("RGB")
    if rng.random() < 0.5:
        bg = bg.transpose(Image.FLIP_LEFT_RIGHT)
    bg = ImageEnhance.Brightness(bg).enhance(rng.uniform(0.85, 1.15))
    bg = ImageEnhance.Contrast(bg).enhance(rng.uniform(0.9, 1.1))
    bg = ImageEnhance.Color(bg).enhance(rng.uniform(0.85, 1.1))
    return bg


# ----------------------------------------------------------------------------
# 单实例变换 + 粘贴
# ----------------------------------------------------------------------------

def transform_cutout(cutout, rng, scale_min, scale_max, max_rotate):
    s = rng.uniform(scale_min, scale_max)
    w, h = max(8, int(cutout["w"] * s)), max(8, int(cutout["h"] * s))
    im = cutout["img"].resize((w, h), Image.BILINEAR)
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    if max_rotate > 0:
        ang = rng.uniform(-max_rotate, max_rotate)
        im = im.rotate(ang, resample=Image.BICUBIC, expand=True,
                       fillcolor=(0, 0, 0, 0))
    return im


def paste_all(canvas, placements, W, H):
    """按 z-order 依次贴图, 返回 (合成图, 每实例全画布 mask list)。"""
    masks = []
    for pl in placements:
        x, y = pl["x"], pl["y"]
        im = pl["im"]
        canvas.paste(im, (x, y), im)  # 以 alpha 为掩膜硬贴
        m = np.zeros((H, W), dtype=bool)
        a = np.asarray(im)[:, :, 3] > 128
        mh, mw = a.shape
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + mw), min(H, y + mh)
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = a[y0 - y:y1 - y, x0 - x:x1 - x]
        masks.append(m)
    return canvas, masks


# ----------------------------------------------------------------------------
# 单张合成
# ----------------------------------------------------------------------------

def compose_one(bg_paths, cutouts, rng, k_min, k_max, args, W, H):
    bg = make_background(bg_paths, rng)
    k = rng.randint(k_min, k_max)

    placements = []
    for _ in range(k):
        im = transform_cutout(rng.choice(cutouts), rng,
                              args.scale_min, args.scale_max, args.rotate)
        mw, mh = im.size
        # 高斯聚集在画面中心 -> 密集堆叠; sigma 取画面短边的 1/4
        cx = min(max(rng.gauss(W / 2, W / 4), mw / 2), W - mw / 2)
        cy = min(max(rng.gauss(H / 2, H / 4), mh / 2), H - mh / 2)
        placements.append({"im": im, "x": int(cx - mw / 2), "y": int(cy - mh / 2)})

    # z-order 随机打乱决定遮挡层次
    rng.shuffle(placements)
    canvas, masks = paste_all(bg, placements, W, H)

    # 标注只保留可见部分: 实例 mask 减去所有上层实例的 mask(与人工
    # 标注遮挡物体的习惯一致, GT 与图像像素严格一致)。
    # 物料被上层"拦腰截断"(可见区域分成>=2个有效连通域)时不输出标注。
    anns, dropped, dropped_split = [], 0, 0
    for i, m in enumerate(masks):
        above = np.zeros_like(m)
        for mj in masks[i + 1:]:
            above |= mj
        visible = m & ~above
        if m.sum() == 0 or visible.sum() < args.keep_visible * m.sum():
            dropped += 1
            continue

        # 截断检查: 先剔除 <50px 的边缘噪点连通域, 若剩余 >=2 个
        # 有效连通域则视为物料被截断, 不输出标注
        cc = cv2.connectedComponentsWithStats(visible.astype(np.uint8))
        n_labels, cc_labels, cc_stats = cc[0], cc[1], cc[2]
        areas = cc_stats[1:, cv2.CC_STAT_AREA]
        valid = areas >= 50
        n_valid = int(valid.sum())
        if n_valid >= 2:
            dropped_split += 1
            continue
        if n_valid == 0:
            dropped += 1
            continue
        # 清除噪点后的干净可见 mask
        visible = np.isin(cc_labels, np.nonzero(valid)[0] + 1)

        anns.append({"mask": visible,
                     "visible_ratio": visible.sum() / max(1, m.sum())})

    return canvas, anns, dropped, dropped_split


# ----------------------------------------------------------------------------
# COCO 写出
# ----------------------------------------------------------------------------

def rle_encode(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def save_preview(img, anns, path):
    """GT 轮廓叠加预览图(红色描边)供人工目检。"""
    vis = img.copy()
    dr = ImageDraw.Draw(vis)
    for a in anns:
        m = a["mask"]
        edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0)
                     & np.roll(m, 1, 1) & np.roll(m, -1, 1))
        ys, xs = np.nonzero(edge)
        dr.point(list(zip(xs.tolist(), ys.tolist())), fill=(255, 0, 0))
    vis.save(path, quality=92)


# ----------------------------------------------------------------------------
# labelme / vis 导出 (多边形抽稀)
# ----------------------------------------------------------------------------

def export_labelme_and_vis(args):
    """从 COCO RLE 导出 labelme json(抽稀后的 polygon)与 vis 轮廓图。

    simplify_eps 为道格拉斯-普克抽稀的容差(px), 越大越稀疏;
    RLE 真值不受影响, 仅改变 labelme 打点密度。
    """
    from concurrent.futures import ThreadPoolExecutor

    out = Path(args.out) / "train"
    vis_dir = Path(args.out) / "vis"
    vis_dir.mkdir(exist_ok=True)
    coco = json.loads((out / "_annotations.coco.json").read_text())
    anns_by_img = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    stats = {"shapes": 0, "pts_dense": 0, "pts_sparse": 0}

    def process(im):
        img_id, fname = im["id"], im["file_name"]
        vis = cv2.imread(str(out / fname))
        shapes = []
        for a in anns_by_img.get(img_id, []):
            m = mask_utils.decode({"size": a["segmentation"]["size"],
                                   "counts": a["segmentation"]["counts"].encode()})
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if len(c) < 3:
                    continue
                stats["pts_dense"] += len(c)
                approx = cv2.approxPolyDP(c, args.simplify_eps, True)
                if len(approx) < 3:
                    continue
                stats["pts_sparse"] += len(approx)
                stats["shapes"] += 1
                shapes.append({
                    "label": CATEGORY["name"],
                    "points": [[float(x), float(y)]
                               for x, y in approx.reshape(-1, 2)],
                    "group_id": None, "description": "",
                    "shape_type": "polygon", "flags": {}, "mask": None,
                })
                cv2.drawContours(vis, [approx], -1, (0, 255, 255),
                                 args.vis_thickness)
        cv2.imwrite(str(vis_dir / fname), vis)
        (out / (Path(fname).stem + ".json")).write_text(json.dumps({
            "version": "5.3.0", "flags": {}, "shapes": shapes,
            "imagePath": fname, "imageData": None,
            "imageHeight": im["height"], "imageWidth": im["width"],
        }, ensure_ascii=False))

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(process, coco["images"]))

    print(f"labelme 导出: {len(coco['images'])} json / {stats['shapes']} polygon -> {out}")
    print(f"打点抽稀: {stats['pts_dense']} -> {stats['pts_sparse']} 点 "
          f"(压缩 {1 - stats['pts_sparse'] / max(1, stats['pts_dense']):.0%}, "
          f"eps={args.simplify_eps}px)")
    print(f"vis 导出: {len(coco['images'])} 张 (线宽{args.vis_thickness}) -> {vis_dir}")


def generate(args):
    src = Path(args.src)
    out = Path(args.out) / "train"
    preview_dir = Path(args.out) / "preview"
    out.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    cutouts = build_cutouts(src / "sample")
    bg_paths = sorted((src / "bg").glob("*.jpg")) + sorted((src / "bg").glob("*.png"))
    assert cutouts and bg_paths, "sample/ 或 bg/ 为空"
    print(f"素材: {len(cutouts)} 个实例, 背景 {len(bg_paths)} 张")

    W, H = Image.open(bg_paths[0]).size
    rng = random.Random(args.seed)

    images, annotations = [], []
    ann_id, n_dropped, n_split, k_hist = 1, 0, 0, []
    for i in range(args.num):
        canvas, anns, dropped, dropped_split = compose_one(
            bg_paths, cutouts, rng, args.k_min, args.k_max, args, W, H)
        n_dropped += dropped
        n_split += dropped_split
        k_hist.append(len(anns))

        fname = f"syn__{i:05d}.jpg"
        canvas.save(out / fname, quality=95)
        img_id = i + 1
        images.append({"id": img_id, "file_name": fname, "width": W, "height": H})
        for a in anns:
            m = a["mask"]
            ys, xs = np.nonzero(m)
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": CATEGORY["id"],
                "bbox": [float(xs.min()), float(ys.min()),
                         float(xs.max() - xs.min() + 1),
                         float(ys.max() - ys.min() + 1)],
                "area": float(m.sum()), "segmentation": rle_encode(m),
                "iscrowd": 0, "occlusion_visible": round(a["visible_ratio"], 3),
            })
            ann_id += 1
        if i < args.num_preview:
            save_preview(canvas, anns, preview_dir / fname)

    coco = {"images": images, "annotations": annotations, "categories": [CATEGORY]}
    (out / "_annotations.coco.json").write_text(json.dumps(coco, ensure_ascii=False))

    k_arr = np.array(k_hist)
    print(f"\n生成完成: {args.num} 张 -> {out}")
    print(f"  实例数/图: mean={k_arr.mean():.2f} min={k_arr.min()} max={k_arr.max()}")
    print(f"  总标注: {len(annotations)}, 重遮挡剔除: {n_dropped}, 截断剔除: {n_split}")
    print(f"  preview: {preview_dir} (前 {min(args.num, args.num_preview)} 张红轮廓叠加图)")


# ----------------------------------------------------------------------------
# 合并进现有训练集 (可选)
# ----------------------------------------------------------------------------

def merge(args):
    gen_json = Path(args.out) / "train" / "_annotations.coco.json"
    gen = json.loads(gen_json.read_text())
    target = Path(args.merge_to)
    tjson = target / "_annotations.coco.json"
    coco = json.loads(tjson.read_text())

    bak = tjson.with_suffix(".coco.json.bak")
    if not bak.exists():
        shutil.copy2(tjson, bak)

    existing = {i["file_name"] for i in coco["images"]}
    next_img = max(i["id"] for i in coco["images"]) + 1
    next_ann = max(a["id"] for a in coco["annotations"]) + 1
    added_img = added_ann = 0
    for gi in gen["images"]:
        if gi["file_name"] in existing:  # 幂等
            continue
        iid = next_img; next_img += 1
        coco["images"].append({"id": iid, "file_name": gi["file_name"],
                               "width": gi["width"], "height": gi["height"]})
        shutil.copy2(Path(args.out) / "train" / gi["file_name"],
                     target / gi["file_name"])
        added_img += 1
        for ga in gen["annotations"]:
            if ga["image_id"] == gi["id"]:
                ga = dict(ga)
                ga["id"] = next_ann; next_ann += 1
                ga["image_id"] = iid
                coco["annotations"].append(ga)
                added_ann += 1

    tjson.write_text(json.dumps(coco, ensure_ascii=False))
    print(f"合并: +{added_img} 图 +{added_ann} 标注 -> {tjson}")
    print(f"备份: {bak}")


def main():
    ap = argparse.ArgumentParser(description="Copy-Paste 合成数据生成器")
    ap.add_argument("--src", default="数据集生成_原始数据/长龙航空餐具")
    ap.add_argument("--out", default="数据集生成_生成数据")
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--k-min", type=int, default=4)
    ap.add_argument("--k-max", type=int, default=14)
    ap.add_argument("--scale-min", type=float, default=1.0,
                    help="素材缩放下限, 默认1.0=保持原尺寸不缩放")
    ap.add_argument("--scale-max", type=float, default=1.0,
                    help="素材缩放上限, 默认1.0=保持原尺寸不缩放")
    ap.add_argument("--rotate", type=float, default=20)
    ap.add_argument("--keep-visible", type=float, default=0.6,
                    help="可见面积低于标准物料此比例的实例不输出标注")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-preview", type=int, default=10)
    ap.add_argument("--simplify-eps", type=float, default=4.0,
                    help="labelme polygon 道格拉斯抽稀容差(px), 越大越稀疏")
    ap.add_argument("--vis-thickness", type=int, default=4,
                    help="vis 轮廓图线宽")
    ap.add_argument("--post-only", action="store_true",
                    help="不重新生成图像, 仅基于现有 COCO 重出 labelme/vis")
    ap.add_argument("--merge-to", default=None,
                    help="合成数据目录, 直接追加进该 train 目录后退出")
    args = ap.parse_args()

    if args.merge_to:
        merge(args)
    elif args.post_only:
        export_labelme_and_vis(args)
    else:
        generate(args)
        export_labelme_and_vis(args)


if __name__ == "__main__":
    main()
