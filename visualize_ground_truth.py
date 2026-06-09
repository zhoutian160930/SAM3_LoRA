#!/usr/bin/env python3
"""
可视化 COCO 真值标注：将 bbox 和 mask 画在原始图像上。

用法:
    python visualize_ground_truth.py --data-dir /root/SAM3_LoRA/dataset/liupo/ --split train
    python visualize_ground_truth.py --data-dir /root/SAM3_LoRA/dataset/liupo/ --split valid
    python visualize_ground_truth.py --data-dir /root/SAM3_LoRA/dataset/liupo/ --split test
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pycocotools.mask as mask_utils
from tqdm import tqdm

COLORS = [
    (0, 0, 255), (255, 0, 0), (0, 255, 0),
    (255, 255, 0), (0, 255, 255), (255, 0, 255),
    (128, 0, 128), (128, 128, 0), (0, 128, 128),
    (255, 128, 0), (0, 128, 255), (128, 0, 255),
]


def get_color(idx):
    return COLORS[idx % len(COLORS)]


def load_coco_annotations(ann_file):
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)

    images = {img['id']: img for img in coco_data['images']}

    img_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

    return images, img_to_anns, categories


def decode_mask(segmentation, orig_h, orig_w):
    if segmentation is None:
        return None
    try:
        if isinstance(segmentation, dict):
            mask_np = mask_utils.decode(segmentation)
        elif isinstance(segmentation, list):
            rles = mask_utils.frPyObjects(segmentation, orig_h, orig_w)
            rle = mask_utils.merge(rles)
            mask_np = mask_utils.decode(rle)
        else:
            return None
        return mask_np.astype(np.uint8) * 255
    except Exception as e:
        print(f"  警告: mask 解码失败: {e}")
        return None


def draw_annotations(img_path, annotations, categories, output_path, alpha=0.4):
    """
    在图像上绘制所有真值标注（bbox + mask + 类别名）。

    Args:
        img_path: 原始图像路径
        annotations: COCO annotations 列表
        categories: {category_id: category_name}
        output_path: 输出路径
        alpha: mask 叠加透明度 (0=完全透明, 1=完全不透明)
    """
    pil_img = Image.open(img_path).convert("RGB")
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    overlay = img.copy()

    for i, ann in enumerate(annotations):
        color = get_color(i)
        category_id = ann.get("category_id", 0)
        class_name = categories.get(category_id, "object")

        bbox = ann.get("bbox", None)
        if bbox is not None:
            x, y, bw, bh = bbox
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + bw), int(y + bh)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = f"{class_name} #{i}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
            cv2.putText(img, label, (x1 + 1, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        segmentation = ann.get("segmentation", None)
        if segmentation is not None:
            mask = decode_mask(segmentation, h, w)
            if mask is not None:
                colored_mask = np.zeros_like(img, dtype=np.uint8)
                colored_mask[:, :] = color
                mask_bool = mask > 0
                overlay[mask_bool] = cv2.addWeighted(
                    overlay[mask_bool], 1 - alpha,
                    colored_mask[mask_bool], alpha, 0
                )
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    cv2.drawContours(img, [cnt], -1, color, 1, cv2.LINE_AA)

    result = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    cv2.imwrite(output_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description="可视化 COCO 真值标注")
    parser.add_argument("--data-dir", required=True,
                        help="数据根目录，包含 train/valid/test 子目录")
    parser.add_argument("--split", default="train",
                        choices=["train", "valid", "test"],
                        help="要可视化的数据分片")
    parser.add_argument("--out-dir", default=None,
                        help="输出目录（默认: data-dir/split_vis/)")
    parser.add_argument("--alpha", type=float, default=0.35,
                        help="mask 叠加透明度 (0-1)")
    parser.add_argument("--max-images", type=int, default=0,
                        help="最大可视化图像数 (0=全部)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    split_dir = data_dir / args.split
    ann_file = split_dir / "_annotations.coco.json"

    if not ann_file.exists():
        print(f"错误: 找不到标注文件 {ann_file}")
        return

    out_dir = args.out_dir or str(data_dir / f"{args.split}_vis")
    os.makedirs(out_dir, exist_ok=True)

    print(f"加载标注: {ann_file}")
    images, img_to_anns, categories = load_coco_annotations(ann_file)
    print(f"  图像数: {len(images)}")
    print(f"  类别数: {len(categories)}")
    print(f"  类别: {categories}")

    image_ids = sorted(images.keys())
    if args.max_images > 0:
        image_ids = image_ids[:args.max_images]

    for img_id in tqdm(image_ids, desc="绘制真值"):
        img_info = images[img_id]
        img_path = split_dir / img_info['file_name']

        if not img_path.exists():
            print(f"  跳过: 图像不存在 {img_path}")
            continue

        annotations = img_to_anns.get(img_id, [])
        name = Path(img_info['file_name']).stem
        out_path = os.path.join(out_dir, f"{name}_gt.png")

        draw_annotations(str(img_path), annotations, categories, out_path, alpha=args.alpha)

    print(f"\n完成! 可视化结果保存在: {out_dir}")


if __name__ == "__main__":
    main()
