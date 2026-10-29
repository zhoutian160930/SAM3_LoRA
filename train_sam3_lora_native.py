
#!/usr/bin/env python3
"""
SAM3 LoRA Training Script

Validation Strategy (Following SAM3):
  - During training: Only compute validation LOSS (fast, no metrics)
  - After training: Run validate_sam3_lora.py for full metrics (mAP, cgF1) with NMS

This approach significantly speeds up training by avoiding expensive metric computation
during each epoch, while still monitoring overfitting via validation loss.

Multi-GPU Training:
  Single GPU:
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml

  Single GPU with local checkpoint:
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml --checkpoint /path/to/sam3.pt

  Multi-GPU (DDP):
    torchrun --nproc_per_node=2 train_sam3_lora_native.py --config configs/full_lora_config.yaml --multi-gpu

  Multi-GPU with specific GPUs:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_sam3_lora_native.py --config configs/full_lora_config.yaml --multi-gpu
"""

import os

# Must be set before torch import: reduces CUDA fragmentation OOM on
# box-prompted training (query count varies per batch -> spiky allocations).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import math
import yaml
import json
import shutil
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path
import numpy as np
from PIL import Image as PILImage
import contextlib

# Distributed training imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# SAM3 Imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.model_misc import SAM3Output
from sam3.train.loss.loss_fns import (
    IABCEMdetr,
    Boxes,
    Masks,
    CORE_LOSS_KEY,
    sigmoid_focal_loss,
    dice_loss,
)
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import Datapoint, Image, Object, FindQueryLoaded, InferenceMetadata
from sam3.model.box_ops import box_xywh_to_xyxy
from lora_layers import LoRAConfig, apply_lora_to_model, save_lora_weights, count_parameters

from torchvision.transforms import v2
import pycocotools.mask as mask_utils  # Required for RLE mask decoding in COCO dataset
from sam3.train.masks_ops import rle_encode  # For encoding masks to RLE format

# Note: Evaluation modules (mAP, cgF1, NMS) are in validate_sam3_lora.py
# Training only computes validation loss, following SAM3's approach


# ============================================================================
# Distributed Training Utilities
# ============================================================================

def setup_distributed():
    """Initialize distributed training environment."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    return local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size():
    """Get the number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """Get the rank of current process."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def print_rank0(*args, **kwargs):
    """Print only on rank 0."""
    if is_main_process():
        print(*args, **kwargs)


def jitter_box(box_cxcywh, scale=0.1, shift=0.1):
    """Jitter a normalized [cx,cy,w,h] box to simulate detector noise.

    Applies random center shift (fraction of box size) and scale change,
    then clamps corners to [0, 1] and returns CxCyWH.
    """
    cx, cy, w, h = box_cxcywh.tolist()
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    bw, bh = x2 - x1, y2 - y1
    x1 = x1 + (torch.rand(1).item() * 2 - 1) * shift * bw
    x2 = x2 + (torch.rand(1).item() * 2 - 1) * shift * bw
    y1 = y1 + (torch.rand(1).item() * 2 - 1) * shift * bh
    y2 = y2 + (torch.rand(1).item() * 2 - 1) * shift * bh
    s = 1 + (torch.rand(1).item() * 2 - 1) * scale
    cx2, cy2 = (x1 + x2) / 2, (y1 + y2) / 2
    nw, nh = (x2 - x1) * s, (y2 - y1) * s
    x1, y1, x2, y2 = cx2 - nw / 2, cy2 - nh / 2, cx2 + nw / 2, cy2 + nh / 2
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(1.0, x2), min(1.0, y2)
    x2 = max(x2, x1 + 1e-4)
    y2 = max(y2, y1 + 1e-4)
    return torch.tensor([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1],
                        dtype=box_cxcywh.dtype)


def compute_source_weights(dataset, alpha=1.0):
    """Compute per-sample weights from "source__file" name prefixes.

    Weight_i = 1 / count(source_i)^alpha, normalized so sum(weights) == N.
    alpha=1.0 -> each source contributes equally per epoch (full balancing);
    alpha=0.5 -> softer balancing; alpha=0 -> original distribution.

    Returns None when filenames don't carry a "source__" prefix.
    """
    names = [dataset.images[img_id]['file_name'] for img_id in dataset.image_ids]
    has_prefix = sum(1 for n in names if "__" in n)
    if has_prefix < 0.5 * len(names):
        print_rank0("balanced_source_sampling: most filenames lack 'source__' prefix, "
                    "falling back to plain shuffling")
        return None

    from collections import Counter
    sources = [n.split("__", 1)[0] for n in names]
    counts = Counter(sources)
    weights = torch.tensor(
        [1.0 / (counts[s] ** alpha) for s in sources], dtype=torch.double
    )
    weights *= len(weights) / weights.sum()

    n = len(weights)
    # after normalization sum(weights) == N, so expected draws per source:
    expected = {s: sum(weights[i].item() for i, src in enumerate(sources) if src == s)
                for s in counts}
    print_rank0("balanced_source_sampling: per-source stats "
                f"(alpha={alpha}, {len(counts)} sources)")
    for s, c in counts.most_common():
        print_rank0(f"  {s}: {c} entries -> ~{expected[s]:.0f} draws/epoch "
                    f"(x{expected[s] / c:.2f})")
    return weights


class DistributedWeightedSampler(Sampler):
    """DDP-compatible weighted sampler: all ranks draw the same weighted
    permutation from a shared seed, then take interleaved shards."""

    def __init__(self, weights, num_replicas, rank, seed=42):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = len(self.weights)
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        # pad total draws to be divisible by num_replicas
        self.total_size = (self.num_samples + self.num_replicas - 1) // self.num_replicas * self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights, self.total_size, replacement=True, generator=g
        ).tolist()
        return iter(indices[self.rank:self.total_size:self.num_replicas])

    def __len__(self):
        return self.total_size // self.num_replicas


# ============================================================================
# 负样本（错误标注）加权 Mask 损失
# ============================================================================

class WeightedMasks(Masks):
    """在 SAM3 原生 Masks 损失上加“错误标注区域”像素权重。

    监督目标仍然是**正确**的 mask，只是让 loss 更关注人容易标错的像素：
        code=1（错误 mask 多标出来的区域，FP 高发）→ weight_extra
        code=2（正确 mask 被漏标的区域，FN 高发）  → weight_missed
        code=0（其余像素）                         → 1.0

    权重码图由 COCOSegmentDataset 生成，经 collate 以
    targets["neg_weight_codes"]（[B, H, W] uint8，每张图一张）传入，
    这里按 targets["num_boxes"] 展开成逐实例，与 targets["masks"] 打包顺序对齐。

    没有权重码图时（例如 val 集没配 wrong_ann.json）自动退化为原生 Masks。
    """

    def __init__(
        self,
        weight_dict=None,
        compute_aux=False,
        focal_alpha=0.25,
        focal_gamma=2.0,
        neg_weight_extra=3.0,
        neg_weight_missed=3.0,
        apply_to_dice=False,
        **kwargs,
    ):
        super().__init__(
            weight_dict=weight_dict,
            compute_aux=compute_aux,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            **kwargs,
        )
        self.neg_weight_extra = float(neg_weight_extra)
        self.neg_weight_missed = float(neg_weight_missed)
        self.apply_to_dice = bool(apply_to_dice)
        if "neg_weight_codes" not in self.target_keys:
            self.target_keys.append("neg_weight_codes")

    def _weight_map(self, codes, size):
        """uint8 码图 → float 权重图 [N, H, W]。"""
        if tuple(codes.shape[-2:]) != tuple(size):
            codes = torch.nn.functional.interpolate(
                codes.float().unsqueeze(1), size=tuple(size), mode="nearest"
            ).squeeze(1).to(codes.dtype)
        w = torch.ones(codes.shape, dtype=torch.float32, device=codes.device)
        if self.neg_weight_extra != 1.0:
            w = torch.where(codes == 1, torch.full_like(w, self.neg_weight_extra), w)
        if self.neg_weight_missed != 1.0:
            w = torch.where(codes == 2, torch.full_like(w, self.neg_weight_missed), w)
        return w

    def get_loss(self, outputs, targets, indices, num_boxes):
        codes = targets.get("neg_weight_codes", None)
        if codes is None:
            # 没有负样本权重 → 原生行为
            return super().get_loss(
                outputs=outputs, targets=targets, indices=indices, num_boxes=num_boxes
            )

        assert "pred_masks" in outputs
        src_masks = outputs["pred_masks"]
        if targets["masks"] is None:
            zero = torch.tensor(0.0, device=src_masks.device)
            return {"loss_mask": zero, "loss_dice": zero.clone()}

        target_masks = (
            targets["masks"] if indices[2] is None else targets["masks"][indices[2]]
        )
        target_masks = target_masks.to(src_masks)
        keep = (
            targets["is_valid_mask"]
            if indices[2] is None
            else targets["is_valid_mask"][indices[2]]
        )
        src_masks = src_masks[(indices[0], indices[1])]

        # 码图已经是【逐实例】顺序（collate 时按每图实例数展开），
        # 与 targets["masks"] 的打包顺序一致；数量对不上时退化为不加权，避免越界。
        n_targets = targets["masks"].shape[0]
        if codes.shape[0] != n_targets:
            if not getattr(self, "_warned_mismatch", False):
                print(f"[WeightedMasks] 码图数量({codes.shape[0]}) != 目标数({n_targets})，"
                      f"本次不加权（退化为原生 Masks）")
                self._warned_mismatch = True
            codes = torch.zeros(
                (n_targets,) + tuple(codes.shape[1:]), dtype=codes.dtype, device=codes.device
            )
        if indices[2] is not None:
            codes = codes[indices[2]]

        src_masks = src_masks[keep]
        target_masks = target_masks[keep]
        codes = codes[keep]

        if target_masks.shape[0] == 0:
            zero = torch.zeros((), device=src_masks.device)
            return {"loss_mask": zero, "loss_dice": zero.clone()}

        mask_hw = target_masks.shape[-2:]

        # 与原生实现一致：上采样到 GT 尺寸（bf16 不支持双线性插值）
        if len(src_masks.shape) == 3:
            src_masks = src_masks[:, None]
        if src_masks.dtype == torch.bfloat16:
            src_masks = src_masks.to(dtype=torch.float32)
        src_masks = torch.nn.functional.interpolate(
            src_masks, size=mask_hw, mode="bilinear", align_corners=False
        )
        src_masks = src_masks[:, 0].flatten(1)
        target_masks = target_masks.flatten(1)

        w = self._weight_map(codes, mask_hw).flatten(1).to(src_masks.dtype)

        # 权重化 focal：w 全 1 时与原生 sigmoid_focal_loss(reduce=True) 完全等价
        focal = sigmoid_focal_loss(
            src_masks,
            target_masks,
            1.0,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduce=False,
        )
        loss_mask = (w * focal).sum() / (num_boxes * src_masks.shape[1])

        if self.apply_to_dice:
            prob = src_masks.sigmoid()
            numerator = (w * prob * target_masks).sum(1)
            denominator = (w * prob).sum(1) + (w * target_masks).sum(1)
            loss_dice = (1 - (numerator * 2 + 1) / (denominator + 1)).sum() / num_boxes
        else:
            loss_dice = dice_loss(src_masks, target_masks, num_boxes)

        return {"loss_mask": loss_mask, "loss_dice": loss_dice}


class COCOSegmentDataset(Dataset):
    """Dataset class for COCO format segmentation data"""

    def __init__(self, data_dir, split="train", neg_ann_file=None, neg_cache=True):
        """
        Args:
            data_dir: Root directory containing train/valid/test folders
            split: One of 'train', 'valid', 'test'
            neg_ann_file: 与 _annotations.coco.json 同目录的错误标注文件名
                （如 "wrong_ann.json"）。仅用于生成“错误区域”权重图，
                监督目标仍然是 _annotations.coco.json 里的正确 mask。None 表示关闭。
            neg_cache: 是否缓存 1008×1008 的权重码图（.neg_weight_cache/*.npy）
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.is_train = (split == "train")
        self.split_dir = self.data_dir / split

        # Load COCO annotations
        ann_file = self.split_dir / "_annotations.coco.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")

        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)

        # Build index: image_id -> image info
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.image_ids = sorted(list(self.images.keys()))

        # Build index: image_id -> list of annotations
        self.img_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        # Load categories
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        print(f"Loaded COCO dataset: {split} split")
        print(f"  Images: {len(self.image_ids)}")
        print(f"  Annotations: {len(self.coco_data['annotations'])}")
        print(f"  Categories: {self.categories}")

        self.resolution = 1008

        # ------------------------------------------------------------------
        # 负样本（错误标注）监督：读取同目录下的错误标注文件
        #   只用来构造“错误区域”权重图，训练目标仍是正确 mask
        # ------------------------------------------------------------------
        self.neg_anns = {}
        self.neg_enabled = bool(neg_ann_file)
        self.neg_cache_dir = None
        if self.neg_enabled:
            neg_path = self.split_dir / neg_ann_file
            if neg_path.exists():
                with open(neg_path, "r") as f:
                    neg_data = json.load(f)
                neg_id2name = {img["id"]: img["file_name"] for img in neg_data["images"]}
                for ann in neg_data["annotations"]:
                    name = neg_id2name.get(ann["image_id"])
                    if name is None:
                        continue
                    self.neg_anns.setdefault(name, []).append(ann)
                if neg_cache:
                    self.neg_cache_dir = self.split_dir / ".neg_weight_cache"
                    self.neg_cache_dir.mkdir(parents=True, exist_ok=True)
                print(f"  错误标注(负样本): {len(self.neg_anns)} 张图 <- {neg_path.name}")
            else:
                print(f"  ⚠️ 未找到错误标注 {neg_path}，本 split 只用正常标注")
                self.neg_enabled = False

        # Train-only photometric jitter (geometric flip handled in __getitem__
        # so box/mask stay in sync with the image).
        if self.is_train:
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        return len(self.image_ids)

    # ------------------------------------------------------------------
    # 负样本（错误标注）相关工具
    # ------------------------------------------------------------------
    @staticmethod
    def decode_segmentation(segmentation, h, w):
        """polygon / 未压缩 RLE / 压缩 RLE 统一解码为 bool mask (h, w)。"""
        if isinstance(segmentation, dict):
            counts = segmentation.get("counts")
            size = list(segmentation.get("size", [h, w]))
            if isinstance(counts, list):
                if counts and sum(counts) == int(size[0]) * int(size[1]):
                    # 未压缩 RLE：counts 是交替游程长度
                    rle = mask_utils.frPyObjects({"size": size, "counts": counts}, *size)
                else:
                    rle = {"size": size, "counts": bytes(counts)}
                return mask_utils.decode(rle) > 0
            if isinstance(counts, str):
                return mask_utils.decode({"size": size, "counts": counts.encode("ascii")}) > 0
            return mask_utils.decode(segmentation) > 0
        if isinstance(segmentation, list):
            rles = mask_utils.frPyObjects(segmentation, h, w)
            return mask_utils.decode(mask_utils.merge(rles)) > 0
        raise TypeError(f"Unknown segmentation type: {type(segmentation)}")

    @classmethod
    def _union_masks(cls, anns, h, w):
        """把一组标注的 mask 取并集（instance 无关，只看前景覆盖）。"""
        union = np.zeros((h, w), dtype=bool)
        for ann in anns:
            seg = ann.get("segmentation")
            if not seg:
                continue
            try:
                union |= cls.decode_segmentation(seg, h, w)
            except Exception:
                continue
        return union

    def _neg_weight_code(self, img_info):
        """生成错误区域权重码图 [res, res] uint8：0=正常, 1=错误多标, 2=漏标。"""
        if not self.neg_enabled:
            return None
        name = img_info["file_name"]
        wrong_anns = self.neg_anns.get(name)
        if not wrong_anns:
            return None

        cache_path = None
        if self.neg_cache_dir is not None:
            cache_path = self.neg_cache_dir / f"{Path(name).stem}.npy"
            if cache_path.exists():
                return torch.from_numpy(np.load(cache_path).copy())

        h = int(img_info.get("height") or 0)
        w = int(img_info.get("width") or 0)
        if h <= 0 or w <= 0:
            return None

        normal = self._union_masks(self.img_to_anns.get(img_info["id"], []), h, w)
        wrong = self._union_masks(wrong_anns, h, w)
        codes = np.zeros((h, w), dtype=np.uint8)
        codes[wrong & ~normal] = 1   # 错误标注多出来的像素（FP 高发）
        codes[normal & ~wrong] = 2   # 正确标注被漏掉的像素（FN 高发）

        # 与 mask 一致：用 nearest 缩放到模型分辨率
        codes_small = np.asarray(
            PILImage.fromarray(codes).resize(
                (self.resolution, self.resolution), PILImage.NEAREST
            )
        )
        if cache_path is not None:
            np.save(cache_path, codes_small)
        # .copy() 保证 tensor 可写（np.load 返回的数组是只读的）
        return torch.from_numpy(codes_small.copy())

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        # Load image
        img_path = self.split_dir / img_info['file_name']
        pil_image = PILImage.open(img_path).convert("RGB")
        orig_w, orig_h = pil_image.size

        # Resize image
        pil_image = pil_image.resize((self.resolution, self.resolution), PILImage.BILINEAR)

        # Transform to tensor (train: includes color jitter)
        image_tensor = self.transform(pil_image)

        # Train-only horizontal flip, synced with boxes/masks:
        # image/segment flip along W; normalized CxCyWH box: cx -> 1 - cx.
        do_flip = self.is_train and (torch.rand(1).item() < 0.5)
        if do_flip:
            image_tensor = torch.flip(image_tensor, dims=[-1])

        # Get annotations for this image
        annotations = self.img_to_anns.get(img_id, [])

        objects = []
        object_class_names = []

        # Scale factors
        scale_w = self.resolution / orig_w
        scale_h = self.resolution / orig_h

        for i, ann in enumerate(annotations):
            # Get bbox - format is [x, y, width, height] in COCO format
            bbox_coco = ann.get("bbox", None)
            if bbox_coco is None:
                continue

            # Get class name from category_id
            category_id = ann.get("category_id", 0)
            class_name = self.categories.get(category_id, "object")
            object_class_names.append(class_name)

            # Convert from COCO [x, y, w, h] to normalized [cx, cy, w, h] (CxCyWH)
            # SAM3 internally expects boxes in CxCyWH format normalized to [0, 1]
            x, y, w, h = bbox_coco
            cx = x + w / 2.0
            cy = y + h / 2.0

            # Scale to resolution and normalize to [0, 1]
            box_tensor = torch.tensor([
                cx * scale_w / self.resolution,
                cy * scale_h / self.resolution,
                w * scale_w / self.resolution,
                h * scale_h / self.resolution,
            ], dtype=torch.float32)

            # Apply synced horizontal flip to box (cx -> 1 - cx)
            if do_flip:
                box_tensor = box_tensor.clone()
                box_tensor[0] = 1.0 - box_tensor[0]

            # Handle segmentation mask (polygon or RLE format)
            segment = None
            segmentation = ann.get("segmentation", None)

            if segmentation:
                try:
                    # polygon / 未压缩 RLE / 压缩 RLE 统一走同一个解码器
                    mask_np = self.decode_segmentation(segmentation, orig_h, orig_w)

                    # Resize mask to model resolution
                    mask_t = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
                    mask_t = torch.nn.functional.interpolate(
                        mask_t,
                        size=(self.resolution, self.resolution),
                        mode="nearest"
                    )
                    segment = mask_t.squeeze() > 0.5  # [1008, 1008] boolean tensor
                    if do_flip:
                        segment = torch.flip(segment, dims=[-1])

                except Exception as e:
                    print(f"Warning: Error processing mask for image {img_id}, ann {i}: {e}")
                    segment = None

            obj = Object(
                bbox=box_tensor,
                area=(box_tensor[2] * box_tensor[3]).item(),
                object_id=i,
                segment=segment
            )
            objects.append(obj)

        image_obj = Image(
            data=image_tensor,
            objects=objects,
            size=(self.resolution, self.resolution)
        )

        # Construct Queries - BOX-PROMPTED mode (one query per GT object).
        # Matches deployment (YOLO box -> SAM3 mask). Training boxes are
        # jittered to simulate detector noise; eval uses clean GT boxes.
        # obj.bbox is already normalized CxCyWH (same format as infer_sam input_bbox).
        queries = []
        if len(objects) > 0:
            for obj, class_name in zip(objects, object_class_names):
                box_cxcywh = obj.bbox.clone()
                if self.is_train:
                    box_cxcywh = jitter_box(box_cxcywh, scale=0.1, shift=0.1)
                query = FindQueryLoaded(
                    query_text=class_name.lower(),
                    image_id=0,
                    object_ids_output=[obj.object_id],
                    is_exhaustive=True,
                    query_processing_order=0,
                    input_bbox=box_cxcywh,
                    input_bbox_label=torch.tensor([1], dtype=torch.long),
                    inference_metadata=InferenceMetadata(
                        coco_image_id=img_id,
                        original_image_id=img_id,
                        original_category_id=0,
                        original_size=(orig_h, orig_w),
                        object_id=obj.object_id,
                        frame_index=-1
                    )
                )
                queries.append(query)
        else:
            # No GT objects: emit a single text-only query (no box) as fallback.
            query = FindQueryLoaded(
                query_text="transparent object",
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=img_id,
                    original_image_id=img_id,
                    original_category_id=0,
                    original_size=(orig_h, orig_w),
                    object_id=-1,
                    frame_index=-1
                )
            )
            queries.append(query)

        return_data = Datapoint(
            find_queries=queries,
            images=[image_obj],
            raw_images=[pil_image]
        )

        # 负样本（错误标注）权重码图：每张图一张，collate 时单独收集
        # （Datapoint 是 sam3 的 dataclass，这里用动态属性避免改 vendored 源码）
        if self.neg_enabled:
            neg_code = self._neg_weight_code(img_info)
            if neg_code is not None:
                if do_flip:
                    neg_code = torch.flip(neg_code, dims=[-1])
                return_data.neg_weight_code = neg_code

        return return_data


def merge_overlapping_masks(binary_masks, scores, boxes, iou_threshold=0.3):
    """
    Merge overlapping masks that likely represent the same object.

    Args:
        binary_masks: Binary masks [N, H, W]
        scores: Confidence scores [N]
        boxes: Bounding boxes [N, 4]
        iou_threshold: IoU threshold for merging (default: 0.3)

    Returns:
        Tuple of (merged_masks, merged_scores, merged_boxes)
    """
    if len(binary_masks) == 0:
        return binary_masks, scores, boxes

    # Sort by score (highest first)
    sorted_indices = torch.argsort(scores, descending=True)
    binary_masks = binary_masks[sorted_indices]
    scores = scores[sorted_indices]
    boxes = boxes[sorted_indices]

    merged_masks = []
    merged_scores = []
    merged_boxes = []
    used = torch.zeros(len(binary_masks), dtype=torch.bool)

    for i in range(len(binary_masks)):
        if used[i]:
            continue

        current_mask = binary_masks[i].clone()
        current_score = scores[i].item()
        current_box = boxes[i]
        used[i] = True

        # Find overlapping masks and merge them
        for j in range(i + 1, len(binary_masks)):
            if used[j]:
                continue

            # Compute IoU
            intersection = (current_mask & binary_masks[j]).sum().item()
            union = (current_mask | binary_masks[j]).sum().item()
            iou = intersection / union if union > 0 else 0

            # If overlaps significantly, merge it
            if iou > iou_threshold:
                current_mask = current_mask | binary_masks[j]
                current_score = max(current_score, scores[j].item())
                used[j] = True

        merged_masks.append(current_mask)
        merged_scores.append(current_score)
        merged_boxes.append(current_box)

    if len(merged_masks) > 0:
        merged_masks = torch.stack(merged_masks)
        merged_scores = torch.tensor(merged_scores, device=scores.device)
        merged_boxes = torch.stack(merged_boxes)
    else:
        merged_masks = binary_masks[:0]
        merged_scores = scores[:0]
        merged_boxes = boxes[:0]

    return merged_masks, merged_scores, merged_boxes


def convert_predictions_to_coco_format(predictions_list, image_ids, resolution=288, score_threshold=0.0, merge_overlaps=True, iou_threshold=0.3, debug=False):
    """
    Convert model predictions to COCO format for evaluation.

    OPTIMIZATION: Keep masks at native model output resolution (288×288)
    GT is downsampled to match, so no upsampling needed!

    Args:
        predictions_list: List of prediction dictionaries from the model
        image_ids: List of image IDs corresponding to predictions
        resolution: Mask resolution for evaluation (default: 288, model's native output)
        score_threshold: Minimum score threshold for predictions
        merge_overlaps: Whether to merge overlapping predictions (default: True)
        iou_threshold: IoU threshold for merging overlaps (default: 0.3)
        debug: Print debug information

    Returns:
        List of prediction dictionaries in COCO format
    """
    coco_predictions = []
    pred_id = 0

    for img_id, preds in zip(image_ids, predictions_list):
        if preds is None or len(preds.get('pred_logits', [])) == 0:
            continue

        # Extract predictions
        logits = preds['pred_logits']  # [num_queries, 1]
        boxes = preds['pred_boxes']    # [num_queries, 4]
        masks = preds['pred_masks']    # [num_queries, H, W]

        scores = torch.sigmoid(logits).squeeze(-1)  # [num_queries]

        # Filter by score threshold
        valid_mask = scores > score_threshold
        num_before = len(scores)
        scores = scores[valid_mask]
        boxes = boxes[valid_mask]
        masks = masks[valid_mask]

        if debug and img_id == image_ids[0]:  # Debug first image only
            print(f"  Image {img_id}: {num_before} queries -> {len(scores)} after filtering (threshold={score_threshold})")

        # Convert masks to binary (apply sigmoid first, then threshold)
        binary_masks = (torch.sigmoid(masks) > 0.5).cpu()

        # Merge overlapping predictions to avoid over-segmentation penalty
        if merge_overlaps and len(binary_masks) > 0:
            num_before_merge = len(binary_masks)
            binary_masks, scores, boxes = merge_overlapping_masks(
                binary_masks, scores.cpu(), boxes.cpu(), iou_threshold=iou_threshold
            )
            if debug and img_id == image_ids[0]:
                print(f"  Merged {num_before_merge} predictions -> {len(binary_masks)} (IoU threshold={iou_threshold})")

        # Encode masks to RLE (at native resolution - much faster!)
        if len(binary_masks) > 0:
            # Check if masks have content
            mask_areas = binary_masks.flatten(1).sum(1)

            if debug and img_id == image_ids[0]:
                print(f"  Mask shape: {binary_masks.shape}")
                print(f"  Mask areas: min={mask_areas.min():.0f}, max={mask_areas.max():.0f}, mean={mask_areas.float().mean():.0f}")

            rles = rle_encode(binary_masks)

            for idx, (rle, score, box) in enumerate(zip(rles, scores.cpu().tolist(), boxes.cpu().tolist())):
                # Convert box from normalized [cx, cy, w, h] to [x, y, w, h] in pixel coordinates
                cx, cy, w, h = box
                x = (cx - w/2) * resolution
                y = (cy - h/2) * resolution
                w = w * resolution
                h = h * resolution

                coco_predictions.append({
                    'image_id': int(img_id),
                    'category_id': 1,  # Single category for instance segmentation
                    'segmentation': rle,
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'score': float(score),
                    'id': pred_id
                })
                pred_id += 1

    return coco_predictions


def create_coco_gt_from_dataset(dataset, image_ids=None, mask_resolution=288):
    """
    Create COCO ground truth dictionary from SimpleSAM3Dataset.

    OPTIMIZATION: Downsample GT masks to match prediction resolution (288×288)
    instead of upsampling predictions to 1008×1008. Much faster!

    Args:
        dataset: SimpleSAM3Dataset instance
        image_ids: Optional list of specific image IDs to include
        mask_resolution: Resolution to downsample masks to (default: 288 to match model output)

    Returns:
        Dictionary in COCO format
    """
    coco_gt = {
        'info': {
            'description': 'SAM3 LoRA Validation Dataset',
            'version': '1.0',
            'year': 2024
        },
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'object'}]
    }

    ann_id = 0
    indices = range(len(dataset)) if image_ids is None else image_ids

    # Scale factor for boxes (masks will be at mask_resolution, boxes scaled accordingly)
    scale_factor = mask_resolution / dataset.resolution

    for idx in indices:
        # Add image entry at mask resolution
        coco_gt['images'].append({
            'id': int(idx),
            'width': mask_resolution,
            'height': mask_resolution,
            'is_instance_exhaustive': True  # Required for cgF1 evaluation
        })

        # Get datapoint
        datapoint = dataset[idx]

        # Add annotations
        for obj in datapoint.images[0].objects:
            # Convert normalized CxCyWH box to COCO [x, y, w, h] at mask_resolution
            cx, cy, bw, bh = (obj.bbox * mask_resolution).tolist()
            x, y, w, h = cx - bw / 2, cy - bh / 2, bw, bh

            ann = {
                'id': ann_id,
                'image_id': int(idx),
                'category_id': 1,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
                'ignore': 0
            }

            # Add segmentation if available - downsample to mask_resolution
            if obj.segment is not None:
                # Downsample mask from 1008×1008 to mask_resolution×mask_resolution
                mask_tensor = obj.segment.unsqueeze(0).unsqueeze(0).float()
                downsampled_mask = torch.nn.functional.interpolate(
                    mask_tensor,
                    size=(mask_resolution, mask_resolution),
                    mode='bilinear',
                    align_corners=False
                ) > 0.5

                mask_np = downsampled_mask.squeeze().cpu().numpy().astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask_np))
                rle['counts'] = rle['counts'].decode('utf-8')
                ann['segmentation'] = rle

            coco_gt['annotations'].append(ann)
            ann_id += 1

    return coco_gt


def convert_predictions_to_coco_format_original_res(predictions_list, image_ids, dataset, model_resolution=288, score_threshold=0.0, merge_overlaps=True, iou_threshold=0.3, debug=False):
    """
    Convert model predictions to COCO format at ORIGINAL image resolution.

    This matches the inference approach (infer_sam.py) where:
    1. Masks are upsampled from 288x288 to original image size
    2. Boxes are scaled to original image size
    3. Evaluation happens at original resolution

    Args:
        predictions_list: List of predictions per image
        image_ids: List of image IDs (indices into dataset)
        dataset: Dataset to get original image sizes
        model_resolution: Model output resolution (default: 288)
        score_threshold: Confidence threshold
        merge_overlaps: Whether to merge overlapping predictions
        iou_threshold: IoU threshold for merging
        debug: Print debug info
    """
    coco_predictions = []
    pred_id = 0

    if debug:
        print(f"\n[DEBUG] Converting {len(predictions_list)} predictions to COCO format (ORIGINAL RESOLUTION)...")
        if merge_overlaps:
            print(f"[DEBUG] Overlapping segment merging ENABLED (IoU threshold={iou_threshold})")

    for img_id, preds in zip(image_ids, predictions_list):
        if preds is None or len(preds.get('pred_logits', [])) == 0:
            continue

        # Get original image size from dataset
        datapoint = dataset[img_id]
        orig_h, orig_w = datapoint.find_queries[0].inference_metadata.original_size

        logits = preds['pred_logits']
        boxes = preds['pred_boxes']
        masks = preds['pred_masks']  # [N, 288, 288]

        scores = torch.sigmoid(logits).squeeze(-1)

        # Filter by score threshold
        valid_mask = scores > score_threshold
        num_before = len(scores)
        scores = scores[valid_mask]
        boxes = boxes[valid_mask]
        masks = masks[valid_mask]

        if debug and img_id == image_ids[0]:
            print(f"[DEBUG] Image {img_id}: {num_before} queries -> {len(scores)} after filtering (threshold={score_threshold})")
            if len(scores) > 0:
                print(f"[DEBUG]   Original size: {orig_w}x{orig_h}")
                print(f"[DEBUG]   Filtered scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")

        if len(masks) == 0:
            continue

        # Upsample masks from 288x288 to original resolution (like infer_sam.py)
        # Process on GPU then immediately move to CPU to save memory
        masks_sigmoid = torch.sigmoid(masks)  # [N, 288, 288]
        masks_upsampled = torch.nn.functional.interpolate(
            masks_sigmoid.unsqueeze(1).float(),  # [N, 1, 288, 288]
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # [N, orig_h, orig_w]

        binary_masks = (masks_upsampled > 0.5).cpu()

        # Free GPU memory immediately after upsampling
        del masks_sigmoid, masks_upsampled
        torch.cuda.empty_cache()

        # Merge overlapping predictions
        if merge_overlaps and len(binary_masks) > 0:
            num_before_merge = len(binary_masks)
            binary_masks, scores, boxes = merge_overlapping_masks(
                binary_masks, scores.cpu(), boxes.cpu(), iou_threshold=iou_threshold
            )
            if debug and img_id == image_ids[0]:
                print(f"[DEBUG]   Merged {num_before_merge} predictions -> {len(binary_masks)} (IoU threshold={iou_threshold})")

        if len(binary_masks) > 0:
            mask_areas = binary_masks.flatten(1).sum(1)

            if debug and img_id == image_ids[0]:
                print(f"[DEBUG]   Upsampled mask shape: {binary_masks.shape}")
                print(f"[DEBUG]   Mask areas: min={mask_areas.min():.0f}, max={mask_areas.max():.0f}, mean={mask_areas.float().mean():.0f}")

            rles = rle_encode(binary_masks)

            for idx, (rle, score, box) in enumerate(zip(rles, scores.cpu().tolist(), boxes.cpu().tolist())):
                # Convert box from normalized [0,1] to original image coordinates
                cx, cy, w_norm, h_norm = box
                x = (cx - w_norm/2) * orig_w
                y = (cy - h_norm/2) * orig_h
                w = w_norm * orig_w
                h = h_norm * orig_h

                # Clamp coordinates to image bounds
                x = max(0, min(x, orig_w))
                y = max(0, min(y, orig_h))
                w = max(0, min(w, orig_w - x))
                h = max(0, min(h, orig_h - y))

                # Skip if box is too small after clamping
                if w < 1 or h < 1:
                    continue

                pred_dict = {
                    'image_id': int(img_id),
                    'category_id': 1,
                    'segmentation': rle,
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'score': float(score),
                    'id': pred_id
                }

                if debug and img_id == image_ids[0] and idx == 0:
                    print(f"[DEBUG]   First prediction bbox (at {orig_w}x{orig_h}): {pred_dict['bbox']}")

                coco_predictions.append(pred_dict)
                pred_id += 1

    return coco_predictions


def create_coco_gt_from_dataset_original_res(dataset, image_ids=None, debug=False):
    """
    Create COCO ground truth dictionary from dataset at ORIGINAL resolution.

    This matches the inference approach (infer_sam.py) where GT is kept
    at original image size for evaluation.

    Args:
        dataset: Dataset with images and annotations
        image_ids: List of image IDs to include (None = all)
        debug: Print debug info
    """
    if debug:
        print(f"\n[DEBUG] Creating COCO ground truth (ORIGINAL RESOLUTION)...")

    coco_gt = {
        'info': {
            'description': 'SAM3 LoRA Validation Dataset',
            'version': '1.0',
            'year': 2024
        },
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'object'}]
    }

    ann_id = 0
    indices = range(len(dataset)) if image_ids is None else image_ids

    for idx in indices:
        datapoint = dataset[idx]

        # Get original image size
        orig_h, orig_w = datapoint.find_queries[0].inference_metadata.original_size

        coco_gt['images'].append({
            'id': int(idx),
            'width': orig_w,
            'height': orig_h,
            'is_instance_exhaustive': True
        })

        for obj in datapoint.images[0].objects:
            # Convert normalized CxCyWH box to COCO [x, y, w, h] at original size
            cx, cy, bw, bh = obj.bbox.tolist()
            w = bw * orig_w
            h = bh * orig_h
            x = cx * orig_w - w / 2
            y = cy * orig_h - h / 2

            ann = {
                'id': ann_id,
                'image_id': int(idx),
                'category_id': 1,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
                'ignore': 0
            }

            if obj.segment is not None:
                # Upsample mask from 1008x1008 to original size
                mask_tensor = obj.segment.unsqueeze(0).unsqueeze(0).float()
                upsampled_mask = torch.nn.functional.interpolate(
                    mask_tensor,
                    size=(orig_h, orig_w),
                    mode='bilinear',
                    align_corners=False
                ) > 0.5

                mask_np = upsampled_mask.squeeze().cpu().numpy().astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask_np))
                rle['counts'] = rle['counts'].decode('utf-8')
                ann['segmentation'] = rle

            coco_gt['annotations'].append(ann)
            ann_id += 1

    if debug:
        print(f"[DEBUG] Created {len(coco_gt['images'])} images, {len(coco_gt['annotations'])} annotations")
        if len(coco_gt['annotations']) > 0:
            sample_gt = coco_gt['annotations'][0]
            sample_img = coco_gt['images'][0]
            print(f"[DEBUG] Sample GT: image_id={sample_gt['image_id']}, bbox={sample_gt['bbox']}, image_size={sample_img['width']}x{sample_img['height']}")

    return coco_gt


class SAM3TrainerNative:
    def __init__(self, config_path, multi_gpu=False, checkpoint_path=None):
        self.config_path = Path(config_path)
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Multi-GPU setup
        self.multi_gpu = multi_gpu
        self.local_rank = 0
        self.world_size = 1

        if self.multi_gpu:
            self.local_rank = setup_distributed()
            self.world_size = get_world_size()
            self.device = torch.device(f"cuda:{self.local_rank}")
            print_rank0(f"Multi-GPU training enabled with {self.world_size} GPUs")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Determine checkpoint path
        # Priority: 1. command line argument, 2. config file, 3. default local path, 4. HuggingFace
        if checkpoint_path is None:
            model_cfg = self.config.get("model", {})
            checkpoint_path = model_cfg.get("name") or model_cfg.get("checkpoint_path")
        
        if checkpoint_path is None:
            # Try default local path first
            default_local_path = "/home/data-1/yjs/niu/code/sam3/pth/sam3.pt"
            if os.path.exists(default_local_path):
                checkpoint_path = default_local_path
                print_rank0(f"Using local checkpoint: {checkpoint_path}")
        
        # Build Model
        print_rank0("Building SAM3 model...")
        if checkpoint_path is not None:
            print_rank0(f"Loading model from: {checkpoint_path}")
            self.model = build_sam3_image_model(
                device=self.device.type,
                compile=False,
                load_from_HF=False,  # Don't download from HF when using local checkpoint
                checkpoint_path=checkpoint_path,
                bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
                eval_mode=False
            )
        else:
            print_rank0("No local checkpoint found, will attempt to download from HuggingFace...")
            self.model = build_sam3_image_model(
                device=self.device.type,
                compile=False,
                load_from_HF=True,  # Download from HF as fallback
                bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
                eval_mode=False
            )

        # Apply LoRA
        print_rank0("Applying LoRA...")
        lora_cfg = self.config["lora"]
        lora_config = LoRAConfig(
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
            apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
            apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
            apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
            apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
            apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
        )
        self.model = apply_lora_to_model(self.model, lora_config)

        stats = count_parameters(self.model)
        print_rank0(f"Trainable params: {stats['trainable_parameters']:,} ({stats['trainable_percentage']:.2f}%)")

        self.model.to(self.device)

        # Wrap model with DDP if multi-GPU
        if self.multi_gpu:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False  # Frozen params (requires_grad=False) don't need this flag
            )
            print_rank0(f"Model wrapped with DistributedDataParallel")

        # Store reference to unwrapped model for accessing custom methods
        self._unwrapped_model = self.model.module if self.multi_gpu else self.model

        # Optimizer
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(self.config["training"]["learning_rate"]),
            weight_decay=self.config["training"]["weight_decay"]
        )
        
        # Matcher & Loss
        self.matcher = BinaryHungarianMatcherV2(
            cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, focal=True
        )

        # 负样本（错误标注）监督配置：只对“错误区域”加权，GT 仍是正确 mask
        neg_cfg = self.config.get("negative_supervision", {}) or {}
        self.neg_enabled = bool(neg_cfg.get("enable", False))
        self.neg_ann_file = neg_cfg.get("ann_file", "wrong_ann.json")
        self.neg_weight_extra = float(neg_cfg.get("weight_extra", 3.0))
        self.neg_weight_missed = float(neg_cfg.get("weight_missed", 3.0))
        self.neg_apply_to_dice = bool(neg_cfg.get("apply_to_dice", False))
        self.neg_cache = bool(neg_cfg.get("cache", True))
        if self.neg_enabled:
            print_rank0(
                f"负样本监督: 开启 (ann_file={self.neg_ann_file}, "
                f"weight_extra={self.neg_weight_extra}, "
                f"weight_missed={self.neg_weight_missed}, "
                f"apply_to_dice={self.neg_apply_to_dice})"
            )

        # Create loss functions with correct weights (from original SAM3 training config)
        # Note: These weights are for mask-based training
        loss_fns = [
            Boxes(weight_dict={
                "loss_bbox": 5.0,
                "loss_giou": 2.0
            }),
            IABCEMdetr(
                pos_weight=10.0,
                weight_dict={
                    "loss_ce": 20.0,
                    "presence_loss": 20.0
                },
                pos_focal=False,
                alpha=0.25,
                gamma=2,
                use_presence=True,
                pad_n_queries=200,
            ),
            WeightedMasks(
                weight_dict={
                    "loss_mask": 200.0,  # Much higher weight for mask loss!
                    "loss_dice": 10.0
                },
                focal_alpha=0.25,
                focal_gamma=2.0,
                compute_aux=False,
                neg_weight_extra=self.neg_weight_extra,
                neg_weight_missed=self.neg_weight_missed,
                apply_to_dice=self.neg_apply_to_dice,
            ) if self.neg_enabled else Masks(
                weight_dict={
                    "loss_mask": 200.0,  # Much higher weight for mask loss!
                    "loss_dice": 10.0
                },
                focal_alpha=0.25,
                focal_gamma=2.0,
                compute_aux=False
            )
        ]

        # Create one-to-many matcher for auxiliary outputs
        o2m_matcher = BinaryOneToManyMatcher(
            alpha=0.3,
            threshold=0.4,
            topk=4
        )

        # Use Sam3LossWrapper for proper loss computation
        self.loss_wrapper = Sam3LossWrapper(
            loss_fns_find=loss_fns,
            matcher=self.matcher,
            o2m_matcher=o2m_matcher,
            o2m_weight=2.0,
            use_o2m_matcher_on_o2m_aux=False,
            normalization="local",  # Use local normalization (no distributed training)
            normalize_by_valid_object_num=False,
        )
        
    def train(self):
        # Get data directory from config (should point to directory containing train/valid folders)
        data_dir = self.config["training"]["data_dir"]

        # Load datasets using COCO format
        print_rank0(f"\nLoading training data from {data_dir}...")
        neg_ann_file = self.neg_ann_file if self.neg_enabled else None
        train_ds = COCOSegmentDataset(
            data_dir=data_dir,
            split="train",
            neg_ann_file=neg_ann_file,
            neg_cache=self.neg_cache,
        )

        # Check if validation data exists
        has_validation = False
        val_ds = None

        try:
            print_rank0(f"\nLoading validation data from {data_dir}...")
            val_ds = COCOSegmentDataset(
                data_dir=data_dir,
                split="valid",
                neg_ann_file=neg_ann_file,
                neg_cache=self.neg_cache,
            )
            if len(val_ds) > 0:
                has_validation = True
                print_rank0(f"Found validation data: {len(val_ds)} images")
            else:
                print_rank0(f"Validation dataset is empty.")
                val_ds = None
        except Exception as e:
            print_rank0(f"Could not load validation data: {e}")
            val_ds = None

        if not has_validation:
            val_ds = None

        def collate_fn(batch):
            out = collate_fn_api(batch, dict_key="input", with_seg_masks=True)
            # 负样本权重码图：数据集里是「每图一张」，这里按每图的实例数展开成
            # 「逐实例一张」，顺序与 collate_fn_api 里 segments.extend 的顺序完全一致
            # （都是 for dp in batch: for q in dp.find_queries: for object_id ...）
            per_instance = []
            n_obj_total = 0
            ref_shape = None
            for dp in batch:
                n_obj = sum(len(q.object_ids_output) for q in dp.find_queries)
                n_obj_total += n_obj
                code = getattr(dp, "neg_weight_code", None)
                if code is not None and ref_shape is None:
                    ref_shape = code.shape
                per_instance.extend([code] * n_obj)
            if n_obj_total == 0 or ref_shape is None:
                out["neg_weight_codes"] = None
            else:
                out["neg_weight_codes"] = torch.stack([
                    c if c is not None else torch.zeros(ref_shape, dtype=torch.uint8)
                    for c in per_instance
                ])
            return out

        # Create samplers for distributed training
        train_sampler = None
        val_sampler = None

        # Optional source-balanced sampling: weights from "source__file"
        # prefixes (e.g. mixed_balanced), mitigates domain imbalance.
        if self.config["training"].get("balanced_source_sampling", False):
            alpha = float(self.config["training"].get("source_weight_alpha", 1.0))
            weights = compute_source_weights(train_ds, alpha=alpha)
            if weights is not None:
                if self.multi_gpu:
                    train_sampler = DistributedWeightedSampler(
                        weights,
                        num_replicas=self.world_size,
                        rank=get_rank(),
                        seed=self.config["training"].get("seed", 42),
                    )
                else:
                    train_sampler = WeightedRandomSampler(
                        weights, num_samples=len(train_ds), replacement=True
                    )

        if self.multi_gpu and train_sampler is None:
            train_sampler = DistributedSampler(
                train_ds,
                num_replicas=self.world_size,
                rank=get_rank(),
                shuffle=True
            )
            if has_validation:
                val_sampler = DistributedSampler(
                    val_ds,
                    num_replicas=self.world_size,
                    rank=get_rank(),
                    shuffle=False
                )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config["training"]["batch_size"],
            shuffle=(train_sampler is None),  # Only shuffle if not using sampler
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=self.config["training"].get("num_workers", 0),
            pin_memory=True
        )

        if has_validation:
            val_loader = DataLoader(
                val_ds,
                batch_size=self.config["training"]["batch_size"],
                shuffle=False,
                sampler=val_sampler,
                collate_fn=collate_fn,
                num_workers=self.config["training"].get("num_workers", 0),
                pin_memory=True
            )
        else:
            val_loader = None

        self.model.train()

        # Weights from a standard SAM config roughly
        weight_dict = {
            "loss_ce": 2.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_mask": 5.0,
            "loss_dice": 5.0
        }

        epochs = self.config["training"]["num_epochs"]
        best_val_loss = float('inf')
        print_rank0(f"Starting training for {epochs} epochs...")

        if has_validation:
            print_rank0(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
        else:
            print_rank0(f"Training samples: {len(train_ds)}")
            print_rank0("⚠️  No validation data found - training without validation")

        if self.multi_gpu:
            print_rank0(f"Effective batch size: {self.config['training']['batch_size']} x {self.world_size} = {self.config['training']['batch_size'] * self.world_size}")

        # Helper to move BatchedDatapoint to device
        def move_to_device(obj, device):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            elif isinstance(obj, list):
                return [move_to_device(x, device) for x in obj]
            elif isinstance(obj, tuple):
                return tuple(move_to_device(x, device) for x in obj)
            elif isinstance(obj, dict):
                return {k: move_to_device(v, device) for k, v in obj.items()}
            elif hasattr(obj, "__dataclass_fields__"):
                for field in obj.__dataclass_fields__:
                    val = getattr(obj, field)
                    setattr(obj, field, move_to_device(val, device))
                return obj
            return obj

        # Create output directory
        out_dir = Path(self.config["output"]["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        # Archive the executed config into output_dir (timestamped, rank-0 only)
        if is_main_process() and self.config_path.exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dst = out_dir / f"{self.config_path.stem}_{stamp}.yaml"
            shutil.copy2(self.config_path, dst)
            print_rank0(f"Config archived to: {dst}")

        # Mixed precision (bf16) to cut activation memory ~2x (box-prompted
        # training has many queries per image, so memory is tight).
        use_amp = self.device.type == "cuda"

        # LR schedule: linear warmup -> cosine decay to ~0 (per optimizer step).
        # Implements config keys lr_scheduler/warmup_steps (previously ignored).
        warmup_steps = int(self.config["training"].get("warmup_steps", 0))
        total_steps = max(1, len(train_loader) * epochs)
        sched_name = self.config["training"].get("lr_scheduler", "cosine")
        if sched_name == "cosine":
            def lr_lambda(step):
                if step < warmup_steps:
                    return (step + 1) / max(1, warmup_steps)
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        else:
            lr_lambda = None
        if lr_lambda is not None:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
            print_rank0(f"LR schedule: {sched_name}, warmup={warmup_steps} steps, "
                        f"total={total_steps} steps")
        else:
            self.scheduler = None

        # Gradient clipping (previously ignored config key max_grad_norm)
        max_grad_norm = float(self.config["training"].get("max_grad_norm", 0.0))
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        # Early stopping (previously ignored config block evaluation.early_stopping)
        es_cfg = self.config.get("evaluation", {}).get("early_stopping", {})
        es_enable = bool(es_cfg.get("enable", False)) and has_validation
        es_patience = int(es_cfg.get("patience", 5))
        es_threshold = float(es_cfg.get("threshold", 0.0))
        epochs_no_improve = 0
        if es_enable:
            print_rank0(f"Early stopping: patience={es_patience}, threshold={es_threshold}")

        # Fresh stats file per run (was append-mode, mixing concurrent runs)
        stats_path = out_dir / "val_stats.json"
        if is_main_process():
            stats_path.write_text("")

        for epoch in range(epochs):
            # Set epoch for distributed sampler (required for proper shuffling)
            if self.multi_gpu and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            # Track training losses for this epoch
            train_losses = []

            # Only show progress bar on rank 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not is_main_process())
            for batch_dict in pbar:
                input_batch = batch_dict["input"]

                # Move to device
                input_batch = move_to_device(input_batch, self.device)

                # 负样本权重码图 [B, H, W] uint8
                neg_codes = batch_dict.get("neg_weight_codes", None)
                if neg_codes is not None:
                    neg_codes = neg_codes.to(self.device, non_blocking=True)

                # Forward pass
                # outputs_list is SAM3Output, we need to pass the whole thing to loss_wrapper
                if use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        outputs_list = self.model(input_batch)
                else:
                    outputs_list = self.model(input_batch)

                # Prepare targets for loss
                # input_batch.find_targets is a list of BatchedFindTarget (one per stage)
                find_targets = [self._unwrapped_model.back_convert(target) for target in input_batch.find_targets]

                # 把负样本权重码图挂到 targets（WeightedMasks 会读取）
                if neg_codes is not None:
                    for targets in find_targets:
                        targets["neg_weight_codes"] = neg_codes

                # Move targets to device
                for targets in find_targets:
                    for k, v in targets.items():
                        if isinstance(v, torch.Tensor):
                            targets[k] = v.to(self.device)

                # Add matcher indices to outputs (required by Sam3LossWrapper)
                # Use SAM3Output.iteration_mode to properly iterate over outputs
                with SAM3Output.iteration_mode(
                    outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
                ) as outputs_iter:
                    for stage_outputs, stage_targets in zip(outputs_iter, find_targets):
                        # stage_targets is a single target dict, replicate for all steps
                        stage_targets_list = [stage_targets] * len(stage_outputs)
                        for outputs, targets in zip(stage_outputs, stage_targets_list):
                            # Compute indices for main output
                            outputs["indices"] = self.matcher(outputs, targets)

                            # Also add indices to auxiliary outputs if they exist
                            if "aux_outputs" in outputs:
                                for aux_out in outputs["aux_outputs"]:
                                    aux_out["indices"] = self.matcher(aux_out, targets)

                # Compute loss using Sam3LossWrapper
                # This handles num_boxes calculation and proper weighting
                if use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        loss_dict = self.loss_wrapper(outputs_list, find_targets)
                else:
                    loss_dict = self.loss_wrapper(outputs_list, find_targets)

                # Extract total loss
                total_loss = loss_dict[CORE_LOSS_KEY]

                # Backward
                self.optimizer.zero_grad()
                total_loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                # Track training loss
                train_losses.append(total_loss.item())
                cur_lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix({"loss": total_loss.item(), "lr": f"{cur_lr:.2e}"})

            # Calculate average training loss for this epoch
            avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0

            # Release the training-phase cache before validation so the val
            # forward doesn't stack on top of the full training peak.
            torch.cuda.empty_cache()

            # Validation (only compute loss - no metrics, like SAM3)
            if has_validation and val_loader is not None:
                self.model.eval()
                val_losses = []

                with torch.no_grad():
                    val_pbar = tqdm(val_loader, desc=f"Validation", disable=not is_main_process())

                    for batch_dict in val_pbar:
                        input_batch = batch_dict["input"]
                        input_batch = move_to_device(input_batch, self.device)

                        neg_codes = batch_dict.get("neg_weight_codes", None)
                        if neg_codes is not None:
                            neg_codes = neg_codes.to(self.device, non_blocking=True)

                        # Forward pass
                        if use_amp:
                            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                                outputs_list = self.model(input_batch)
                        else:
                            outputs_list = self.model(input_batch)

                        # Prepare targets
                        find_targets = [self._unwrapped_model.back_convert(target) for target in input_batch.find_targets]

                        # 把负样本权重码图挂到 targets（WeightedMasks 会读取）
                        if neg_codes is not None:
                            for targets in find_targets:
                                targets["neg_weight_codes"] = neg_codes

                        # Move targets to device
                        for targets in find_targets:
                            for k, v in targets.items():
                                if isinstance(v, torch.Tensor):
                                    targets[k] = v.to(self.device)

                        # Add matcher indices to outputs (required by Sam3LossWrapper)
                        with SAM3Output.iteration_mode(
                            outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
                        ) as outputs_iter:
                            for stage_outputs, stage_targets in zip(outputs_iter, find_targets):
                                stage_targets_list = [stage_targets] * len(stage_outputs)
                                for outputs, targets in zip(stage_outputs, stage_targets_list):
                                    outputs["indices"] = self.matcher(outputs, targets)
                                    if "aux_outputs" in outputs:
                                        for aux_out in outputs["aux_outputs"]:
                                            aux_out["indices"] = self.matcher(aux_out, targets)

                        # Compute loss using Sam3LossWrapper
                        loss_dict = self.loss_wrapper(outputs_list, find_targets)
                        total_loss = loss_dict[CORE_LOSS_KEY]

                        val_losses.append(total_loss.item())
                        val_pbar.set_postfix({"val_loss": total_loss.item()})

                avg_val_loss = sum(val_losses) / len(val_losses)

                # Synchronize val_loss across all processes for consistent best model selection
                if self.multi_gpu:
                    val_loss_tensor = torch.tensor([avg_val_loss], device=self.device)
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                    avg_val_loss = val_loss_tensor.item()

                print_rank0(f"\nEpoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

                # Capture pre-update best for early-stopping bookkeeping
                prev_best = best_val_loss

                # Save models based on validation loss (only on rank 0)
                if is_main_process():
                    # Get underlying model from DDP wrapper
                    model_to_save = self.model.module if self.multi_gpu else self.model
                    save_lora_weights(model_to_save, str(out_dir / "last_lora_weights.pt"))

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        save_lora_weights(model_to_save, str(out_dir / "best_lora_weights.pt"))
                        print(f"✓ New best model saved (val_loss: {avg_val_loss:.6f})")

                    # Log to file
                    with open(stats_path, "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch + 1,
                            "train_loss": avg_train_loss,
                            "val_loss": avg_val_loss
                        }) + "\n")

                # Early stopping: reset counter only on meaningful improvement.
                # Compares against the PRE-update best; avg_val_loss is
                # all-reduced, so all ranks decide identically.
                if avg_val_loss < prev_best - es_threshold:
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if es_enable and epochs_no_improve >= es_patience:
                        print_rank0(f"\n⏹ Early stopping triggered at epoch {epoch+1} "
                                    f"(no improvement > {es_threshold} for {es_patience} epochs)")
                        break

                torch.cuda.empty_cache()

                # Back to training mode
                self.model.train()
            else:
                # No validation - just save model each epoch (only on rank 0)
                if is_main_process():
                    model_to_save = self.model.module if self.multi_gpu else self.model
                    save_lora_weights(model_to_save, str(out_dir / "last_lora_weights.pt"))
                torch.cuda.empty_cache()

        # Synchronize before final save
        if self.multi_gpu:
            dist.barrier()

        # Final save (only on rank 0)
        if is_main_process():
            if has_validation:
                print(f"\n{'='*80}")
                print(f"✅ Training complete!")
                print(f"{'='*80}")
                print(f"Best validation loss: {best_val_loss:.6f}")
                print(f"\nModels saved to {out_dir}:")
                print(f"  - best_lora_weights.pt (best validation loss)")
                print(f"  - last_lora_weights.pt (last epoch)")
                print(f"\n📊 To compute full metrics (mAP, cgF1) with NMS:")
                print(f"   python validate_sam3_lora.py \\")
                print(f"     --config <config_path> \\")
                print(f"     --weights {out_dir}/best_lora_weights.pt \\")
                print(f"     --val_data_dir <data_dir>/valid")
                print(f"{'='*80}")
            else:
                # If no validation, copy last to best
                last_path = out_dir / "last_lora_weights.pt"
                best_path = out_dir / "best_lora_weights.pt"
                if last_path.exists():
                    shutil.copy(last_path, best_path)

                print(f"\n{'='*80}")
                print(f"✅ Training complete!")
                print(f"{'='*80}")
                print(f"\nModels saved to {out_dir}:")
                print(f"  - best_lora_weights.pt (copy of last epoch)")
                print(f"  - last_lora_weights.pt (last epoch)")
                print(f"\nℹ️  No validation data - consider adding data/valid/ for better model selection")
                print(f"{'='*80}")

        # Cleanup distributed training
        if self.multi_gpu:
            cleanup_distributed()

def launch_distributed_training(args):
    """Launch training with multiple GPUs using torchrun subprocess."""
    import subprocess
    import sys

    devices = args.device
    num_gpus = len(devices)
    device_str = ",".join(map(str, devices))

    print(f"Launching distributed training on GPUs: {devices}")
    print(f"Number of processes: {num_gpus}")

    # Build the command
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
        "--master_port", str(args.master_port),
        sys.argv[0],  # This script
        "--config", args.config,
        "--device", *map(str, devices),
        "--_launched_by_torchrun"  # Internal flag to indicate we're in subprocess
    ]
    
    # Add checkpoint argument if provided
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])

    # Set environment variable for visible devices
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device_str

    # Run the subprocess
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SAM3 with LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single GPU (default GPU 0):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml

  Single GPU (specific GPU):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml --device 1

  Multi-GPU (GPUs 0 and 1):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml --device 0 1

  Multi-GPU (GPUs 0, 2, 3):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml --device 0 2 3

  Multi-GPU (all 4 GPUs):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml --device 0 1 2 3
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/home/model/work/SAM3_LoRA/configs/full_lora_config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to SAM3 checkpoint file (e.g., sam3.pt). "
             "If not provided, will try: 1) config file (model.name), 2) /home/data-1/yjs/niu/code/sam3/pth/sam3.pt, 3) download from HuggingFace"
    )
    parser.add_argument(
        "--device",
        type=int,
        nargs="+",
        default=[0],
        help="GPU device ID(s) to use. Single value for single GPU, multiple values for multi-GPU. "
             "Example: --device 0 (single GPU), --device 0 1 2 (3 GPUs)"
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=29500,
        help="Master port for distributed training (default: 29500)"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (set automatically by torchrun)"
    )
    parser.add_argument(
        "--_launched_by_torchrun",
        action="store_true",
        help=argparse.SUPPRESS  # Hidden argument for internal use
    )
    args = parser.parse_args()

    # Determine if multi-GPU training is requested
    num_devices = len(args.device)
    is_torchrun_subprocess = args._launched_by_torchrun or "LOCAL_RANK" in os.environ

    if num_devices > 1 and not is_torchrun_subprocess:
        # Multi-GPU requested but not yet in torchrun - launch it
        launch_distributed_training(args)
    else:
        # Single GPU or already in torchrun subprocess
        multi_gpu = num_devices > 1 and is_torchrun_subprocess

        if not multi_gpu and num_devices == 1:
            # Single GPU mode - set the device
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device[0])
            print(f"Using single GPU: {args.device[0]}")

        trainer = SAM3TrainerNative(args.config, multi_gpu=multi_gpu, checkpoint_path=args.checkpoint)
        trainer.train()
