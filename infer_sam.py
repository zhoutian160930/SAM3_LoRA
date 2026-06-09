#!/usr/bin/env python3
import argparse
import os
import sys
import cv2
import yaml
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# SAM3 imports
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.sam3_image_dataset import Datapoint, Image as SAMImage, FindQueryLoaded, InferenceMetadata
from sam3.train.data.collator import collate_fn_api
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI

# LoRA
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights

# 颜色池（你原来的风格）
COLORS = []
def get_random_color(name):
    global COLORS
    if not COLORS:
        COLORS = [
            (0, 0, 255), (255, 0, 0), (0, 255, 0),
            (255, 255, 0), (0, 255, 255), (255, 0, 255)
        ]
    idx = hash(name) % len(COLORS)
    return COLORS[idx]

# ====================== 你提供的可视化：只画轮廓 ======================
def osd(image, results):
    if image is None or not results:
        return image
    vis_img = image.copy()
    h_img, w_img = vis_img.shape[:2]

    for obj in results:
        color = get_random_color(obj["class"])
        x1, y1, x2, y2 = obj["box"]
        x1 = max(0, min(int(x1), w_img))
        y1 = max(0, min(int(y1), h_img))
        x2 = max(0, min(int(x2), w_img))
        y2 = max(0, min(int(y2), h_img))
        box_w = x2 - x1
        box_h = y2 - y1

        if box_w > 0 and box_h > 0 and obj["mask"] is not None:
            mask = obj["mask"]
            if mask.shape[:2] != (box_h, box_w):
                mask = cv2.resize(mask, (box_w, box_h), interpolation=cv2.INTER_NEAREST)
            mask_uint8 = (mask > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt[:, :, 0] += x1
                cnt[:, :, 1] += y1
                cv2.drawContours(vis_img, [cnt], -1, color, 3)
    return vis_img

# ====================== SAM3 + LoRA 批量推理 ======================
class SAM3BatchInfer:
    def __init__(self, config_path, weights=None, resolution=1008, threshold=0.5, checkpoint=None, use_base=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resolution = resolution
        self.threshold = threshold
        self.use_base = use_base

        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        # 模型
        self.model = build_sam3_image_model(
            device=self.device.type,
            compile=False,
            load_from_HF=False,
            checkpoint_path=checkpoint,
            bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            eval_mode=True
        )

        if not use_base:
            lora_cfg = self.cfg["lora"]
            lora_config = LoRAConfig(
                rank=lora_cfg["rank"],
                alpha=lora_cfg["alpha"],
                dropout=0.0,
                target_modules=lora_cfg["target_modules"],
                apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
                apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
                apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
                apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
                apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
                apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
            )
            self.model = apply_lora_to_model(self.model, lora_config)
            load_lora_weights(self.model, weights)

        self.model.to(self.device).eval()

        self.transform = ComposeAPI([
            RandomResizeAPI(sizes=resolution, max_size=resolution, square=True,consistent_transform=False),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

    @torch.no_grad()
    def infer_one_image(self, img_path, boxes_cxcywh, class_ids=None):
        pil_img = Image.open(img_path).convert("RGB")
        w, h = pil_img.size
        datapoint = _make_datapoint(pil_img, boxes_cxcywh)
        datapoint = self.transform(datapoint)
        batch = collate_fn_api([datapoint], "input")["input"]
        batch = copy_data_to_device(batch, self.device)
        out = self.model(batch)
        last = out[-1]

        # 解析结果：每个查询输出一个预测
        scores = last["pred_logits"].sigmoid().max(-1)[0][0]
        boxes = last["pred_boxes"][0]
        masks = last.get("pred_masks", None)
        masks_np = masks[0].cpu().numpy() if masks is not None else None

        results = []
        for i, (cx, cy, bw, bh) in enumerate(boxes.cpu().numpy()):
            score = scores[i].item()
            if score <= self.threshold:
                continue
            x1 = (cx - bw/2) * w
            y1 = (cy - bh/2) * h
            x2 = (cx + bw/2) * w
            y2 = (cy + bh/2) * h
            m = masks_np[i] if masks_np is not None else None
            results.append({
                "box": [x1, y1, x2, y2],
                "mask": m,
                "score": score,
                "class": "object"
            })
        return pil_img, results

def _make_datapoint(pil_img, boxes_cxcywh):
    """构建 Datapoint，每个 YOLO box 作为一个独立的 box 几何提示.

    Args:
        pil_img: PIL 图像
        boxes_cxcywh: list of [cx, cy, bw, bh]  (归一化 CxCyWH, 0~1)
    """
    w, h = pil_img.size
    from sam3.train.data.sam3_image_dataset import Object

    objs = []
    for i, (cx, cy, bw, bh) in enumerate(boxes_cxcywh):
        area = bw * bh * w * h
        objs.append(Object(
            bbox=torch.tensor([cx, cy, bw, bh], dtype=torch.float32),
            area=area, object_id=i
        ))

    sam_img = SAMImage(data=pil_img, objects=objs, size=[h, w])

    # 每个 YOLO box 创建一个独立查询，设置 input_bbox 让模型收到几何提示
    queries = []
    for i in range(len(boxes_cxcywh)):
        box_tensor = torch.tensor(boxes_cxcywh[i], dtype=torch.float32)
        q = FindQueryLoaded(
            query_text="object",
            image_id=0,
            object_ids_output=[i],
            is_exhaustive=True,
            query_processing_order=0,
            input_bbox=box_tensor,
            input_bbox_label=torch.tensor([1], dtype=torch.long),
            inference_metadata=InferenceMetadata(
                coco_image_id=0,
                original_image_id=0,
                original_category_id=1,
                object_id=0,
                frame_index=0,
                original_size=[w, h]
            )
        )
        queries.append(q)

    return Datapoint(find_queries=queries, images=[sam_img])

# ====================== YOLO 标签读取 ======================
def load_yolo_boxes(txt_path):
    """读取 YOLO 标签，返回 (boxes_cxcywh, class_ids).
    YOLO 格式: class_id cx cy bw bh (均为归一化 0~1)
    """
    boxes = []
    class_ids = []
    if not os.path.exists(txt_path):
        return boxes, class_ids
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = list(map(float, line.split()))
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            cx, cy, bw, bh = parts[1:5]
            boxes.append([cx, cy, bw, bh])
            class_ids.append(class_id)
    return boxes, class_ids

# ====================== 批量主函数 ======================
def batch_infer():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--out-dir", default="sam_output")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--checkpoint", default="/home/model/sam3_pth/sam3pt/sam3.pt")
    parser.add_argument("--base-model", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    infer = SAM3BatchInfer(
        config_path=args.config,
        weights=args.weights,
        threshold=args.threshold,
        checkpoint=args.checkpoint,
        use_base=args.base_model
    )

    img_exts = (".jpg", ".jpeg", ".png", ".bmp")
    img_paths = [f for f in os.listdir(args.img_dir) if f.lower().endswith(img_exts)]

    for fname in tqdm(img_paths):
        img_path = os.path.join(args.img_dir, fname)
        name = Path(fname).stem
        txt_path = os.path.join(args.label_dir, name + ".txt")

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        boxes_cxcywh, class_ids = load_yolo_boxes(txt_path)
        if not boxes_cxcywh:
            print(f"跳过 {fname}：无标签")
            continue

        pil_img, results = infer.infer_one_image(img_path, boxes_cxcywh, class_ids)
        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # 绘制 YOLO 输入 box（青色虚线，与模型输出区分）
        for cx, cy, bw, bh in boxes_cxcywh:
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (255, 255, 0), 2)

        vis = osd(img_cv, results)

        out_path = os.path.join(args.out_dir, fname)
        cv2.imwrite(out_path, vis)

if __name__ == "__main__":
    batch_infer()