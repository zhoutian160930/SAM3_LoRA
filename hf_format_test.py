#!/usr/bin/env python3
"""测试 HuggingFace 格式 SAM3 模型的 YOLO-box 提示推理"""

import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from transformers import Sam3Model, Sam3Processor

COLORS = [
    (0, 0, 255), (255, 0, 0), (0, 255, 0),
    (255, 255, 0), (0, 255, 255), (255, 0, 255),
    (128, 0, 128), (128, 128, 0), (0, 128, 128),
]


def get_color(idx):
    return COLORS[idx % len(COLORS)]


def load_yolo_boxes_yolo(txt_path, img_w, img_h):
    """读取 YOLO 标签，返回 pixel xyxy 格式的 boxes。
    YOLO 格式: class_id cx cy bw bh (归一化 0~1)
    返回: [[x1, y1, x2, y2], ...]  (pixel xyxy)
    """
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = list(map(float, line.split()))
            if len(parts) < 5:
                continue
            cx, cy, bw, bh = parts[1:5]
            x1 = (cx - bw / 2) * img_w
            y1 = (cy - bh / 2) * img_h
            x2 = (cx + bw / 2) * img_w
            y2 = (cy + bh / 2) * img_h
            boxes.append([x1, y1, x2, y2])
    return boxes


def draw_results(image, results, input_boxes=None):
    """在图像上绘制推理结果和输入 box。

    Args:
        image: numpy BGR 图像
        results: dict with keys "boxes" (xyxy tensor), "scores" (tensor), "masks" (bool tensor)
        input_boxes: 可选的 YOLO 输入 boxes (pixel xyxy list)
    """
    vis = image.copy()
    h, w = vis.shape[:2]

    if input_boxes:
        for x1, y1, x2, y2 in input_boxes:
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)

    for i in range(len(results["scores"])):
        score = results["scores"][i].item()
        box = results["boxes"][i].cpu().numpy()  # [x1, y1, x2, y2]
        mask = results["masks"][i].cpu().numpy()  # [H, W] bool

        color = get_color(i)
        x1, y1, x2, y2 = box.astype(int)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{score:.2f}"
        cv2.putText(vis, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 2)

    return vis


def main():
    parser = argparse.ArgumentParser(description="HF 格式 SAM3 YOLO-box 提示测试")
    parser.add_argument("--model-path", default="/root/sam3weight/base_v2/")
    parser.add_argument("--img-dir", default="/root/SAM3_LoRA/dataset/liupo/test/")
    parser.add_argument("--label-dir", default="/root/SAM3_LoRA/dataset/liupo/test_labels/")
    parser.add_argument("--out-dir", default="/root/SAM3_LoRA/dataset/liupo/hf_test_output/")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"加载模型: {args.model_path}")
    model = Sam3Model.from_pretrained(args.model_path).to(args.device).eval()
    processor = Sam3Processor.from_pretrained(args.model_path)
    print("模型加载完成")

    img_exts = (".jpg", ".jpeg", ".png", ".bmp")
    img_paths = sorted([f for f in os.listdir(args.img_dir) if f.lower().endswith(img_exts)])

    if args.max_images > 0:
        img_paths = img_paths[:args.max_images]

    for fname in tqdm(img_paths, desc="推理"):
        img_path = os.path.join(args.img_dir, fname)
        name = Path(fname).stem
        txt_path = os.path.join(args.label_dir, name + ".txt")

        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        boxes_xyxy = load_yolo_boxes_yolo(txt_path, w, h)
        if not boxes_xyxy:
            print(f"跳过 {fname}：无标签")
            continue

        input_boxes = [boxes_xyxy]
        input_boxes_labels = [[1] * len(boxes_xyxy)]

        inputs = processor(
            images=image,
            input_boxes=input_boxes,
            input_boxes_labels=input_boxes_labels,
            return_tensors="pt",
        ).to(args.device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=args.threshold,
            mask_threshold=0.5,
            target_sizes=[(h, w)],
        )[0]

        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        vis = draw_results(img_cv, results, input_boxes=boxes_xyxy)

        out_path = os.path.join(args.out_dir, fname)
        cv2.imwrite(out_path, vis)

    print(f"完成! 结果保存在: {args.out_dir}")


if __name__ == "__main__":
    main()
