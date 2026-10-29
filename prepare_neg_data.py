#!/usr/bin/env python3
"""把「正常标注 + 错误标注」两份 COCO json 组装成 trainer 需要的目录结构。

trainer 期望：

    <out_dir>/
    └── <split>/
        ├── _annotations.coco.json   # 正确标签（= normal_ann.json）
        ├── wrong_ann.json           # 错误标签（= wrong_ann.json）
        └── images/...               # 图片（file_name 按 json 里的相对路径）

用法示例（sample_200，单 split）：

    python prepare_neg_data.py \
        --normal_json /root/work/六婆_负样本增广合并版_20261029_084208/sample_200/normal_ann.json \
        --wrong_json  /root/work/六婆_负样本增广合并版_20261029_084208/sample_200/wrong_ann.json \
        --images_dir  /root/work/六婆_负样本增广合并版_20261029_084208/sample_200/images \
        --out_dir     dataset/六婆_neg --split train --link

按 7:3 划分 train/valid 请用 split_neg_dataset.py。
"""

import argparse
import json
import os
import shutil
from pathlib import Path


def load(p):
    with open(p, "r") as f:
        return json.load(f)


def check_pair(normal, wrong):
    """两份 json 的图片集合必须一致。返回 file_name 列表（顺序同 normal）。"""
    n_names = [im["file_name"] for im in normal["images"]]
    w_names = [im["file_name"] for im in wrong["images"]]
    if set(n_names) != set(w_names):
        only_n = sorted(set(n_names) - set(w_names))[:5]
        only_w = sorted(set(w_names) - set(n_names))[:5]
        raise SystemExit(f"两份 json 的图片集合不一致：只在正常里 {only_n}，只在错误里 {only_w}")
    dup = len(n_names) - len(set(n_names))
    if dup:
        raise SystemExit(f"正常标注里有 {dup} 个重复 file_name，请先按图片聚合")
    return n_names


def subset_coco(coco, keep_names):
    """按 file_name 子集裁剪 COCO dict（images / annotations 同步裁剪）。"""
    keep = set(keep_names)
    imgs = [im for im in coco["images"] if im["file_name"] in keep]
    ids = {im["id"] for im in imgs}
    out = {
        "images": imgs,
        "annotations": [a for a in coco["annotations"] if a["image_id"] in ids],
        "categories": coco["categories"],
    }
    return out


def build_split(normal, wrong, images_dir, out_split, link=False):
    """写出一个 split 目录：两个 json + 图片。返回统计信息。"""
    out_split = Path(out_split)
    out_split.mkdir(parents=True, exist_ok=True)
    n_names = check_pair(normal, wrong)

    with open(out_split / "_annotations.coco.json", "w") as f:
        json.dump(normal, f, ensure_ascii=False)
    with open(out_split / "wrong_ann.json", "w") as f:
        json.dump(wrong, f, ensure_ascii=False)

    img_root = Path(images_dir)
    copied, relinked = 0, 0
    for name in n_names:
        src = img_root / name
        if not src.exists():
            src = img_root / Path(name).name        # file_name 可能带 images/ 前缀
        if not src.exists():
            raise SystemExit(f"图片缺失: {src}")
        dst = out_split / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            if link and dst.is_symlink() and Path(os.readlink(dst)).resolve() == src.resolve():
                relinked += 1
                continue
            dst.unlink()
        if link:
            os.symlink(src.resolve(), dst)
        else:
            shutil.copy2(src, dst)
        copied += 1

    return {
        "path": str(out_split),
        "images": len(n_names),
        "normal_anns": len(normal["annotations"]),
        "wrong_anns": len(wrong["annotations"]),
        "copied": copied,
        "relinked": relinked,
    }


def print_result(stats, link=False):
    print(f"✅ {stats['path']}")
    print(f"   正确标注: {stats['images']} 图 / {stats['normal_anns']} 实例 -> _annotations.coco.json")
    print(f"   错误标注: {stats['images']} 图 / {stats['wrong_anns']} 实例 -> wrong_ann.json")
    mode = "软链接" if link else "复制"
    print(f"   图片: {stats['copied']} 张({mode})" +
          (f" + {stats['relinked']} 张已存在" if stats['relinked'] else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal_json", required=True, help="正确标注 COCO json")
    ap.add_argument("--wrong_json", required=True, help="错误标注 COCO json")
    ap.add_argument("--images_dir", required=True, help="图片根目录（file_name 相对它）")
    ap.add_argument("--out_dir", required=True, help="输出数据集根目录")
    ap.add_argument("--split", default="train", choices=["train", "valid", "test"])
    ap.add_argument("--link", action="store_true", help="用软链接代替复制图片（省空间）")
    args = ap.parse_args()

    normal, wrong = load(args.normal_json), load(args.wrong_json)
    stats = build_split(normal, wrong, args.images_dir,
                        Path(args.out_dir) / args.split, link=args.link)
    print_result(stats, link=args.link)
    print("\n训练时在配置里加：")
    print("negative_supervision:")
    print("  enable: true")
    print("  ann_file: \"wrong_ann.json\"")


if __name__ == "__main__":
    main()
