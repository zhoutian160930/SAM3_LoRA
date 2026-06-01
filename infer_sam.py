#!/usr/bin/env python3
"""
SAM3 + LoRA Inference Script

Based on official SAM3 batched inference patterns.
Supports text prompts and visual prompts with LoRA fine-tuned weights.

Usage:
    # Text prompt inference
    python3 infer_sam.py \
        --config configs/full_lora_config.yaml \
        --image path/to/image.jpg \
        --prompt "crack" \
        --output output.png

    # Multiple prompts
    python3 infer_sam.py \
        --config configs/full_lora_config.yaml \
        --image path/to/image.jpg \
        --prompt "crack" "defect" "damage" \
        --output output.png
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
from torchvision.ops import nms

# SAM3 imports
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    Image as SAMImage,
    FindQueryLoaded,
    InferenceMetadata
)
from sam3.train.data.collator import collate_fn_api
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    RandomResizeAPI,
    ToTensorAPI,
    NormalizeAPI,
)
from sam3.eval.postprocessors import PostProcessImage

# LoRA imports
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights


class SAM3LoRAInference:
    """SAM3 model with LoRA for inference."""

    def __init__(
        self,
        config_path: str,
        weights_path: Optional[str] = None,
        resolution: int = 1008,
        detection_threshold: float = 0.5,
        nms_iou_threshold: float = 0.5,
        device: str = "cuda",
        checkpoint_path: Optional[str] = None
    ):
        """
        Initialize SAM3 with LoRA.

        Args:
            config_path: Path to training config YAML
            weights_path: Path to LoRA weights (optional, auto-detected from config)
            resolution: Input image resolution (default: 1008)
            detection_threshold: Confidence threshold for detections (default: 0.5)
            nms_iou_threshold: IoU threshold for NMS (default: 0.5)
            device: Device to run on (default: "cuda")
            checkpoint_path: Path to SAM3 base model checkpoint (optional)
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Auto-detect weights if not provided
        if weights_path is None:
            output_dir = self.config.get('output', {}).get('output_dir', 'outputs/sam3_lora_full')
            weights_path = os.path.join(output_dir, 'best_lora_weights.pt')
            print(f"ℹ️  Auto-detected weights: {weights_path}")

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"LoRA weights not found: {weights_path}")

        self.weights_path = weights_path
        self.resolution = resolution
        self.detection_threshold = detection_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        print(f"🔧 Initializing SAM3 + LoRA...")
        print(f"   Device: {self.device}")
        print(f"   Resolution: {resolution}x{resolution}")
        print(f"   Confidence threshold: {detection_threshold}")
        print(f"   NMS IoU threshold: {nms_iou_threshold}")

        # Determine checkpoint path (same logic as training script)
        if checkpoint_path is None:
            checkpoint_path = self.config.get("model", {}).get("checkpoint_path")
        
        if checkpoint_path is None:
            # Try default local path first
            default_local_path = "/home/data-1/yjs/niu/code/sam3/pth/sam3.pt"
            if os.path.exists(default_local_path):
                checkpoint_path = default_local_path
                print(f"   Using local checkpoint: {checkpoint_path}")

        # Build base model
        print("\n📦 Building SAM3 model...")
        if checkpoint_path is not None:
            print(f"   Loading from: {checkpoint_path}")
            self.model = build_sam3_image_model(
                device=self.device.type,
                compile=False,
                load_from_HF=False,  # Use local checkpoint
                checkpoint_path=checkpoint_path,
                bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
                eval_mode=True
            )
        else:
            print("   No local checkpoint found, downloading from HuggingFace...")
            self.model = build_sam3_image_model(
                device=self.device.type,
                compile=False,
                load_from_HF=False,
                bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
                eval_mode=True
            )

        # Apply LoRA configuration
        print("🔗 Applying LoRA configuration...")
        lora_cfg = self.config["lora"]
        lora_config = LoRAConfig(
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=0.0,  # No dropout during inference
            target_modules=lora_cfg["target_modules"],
            apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
            apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
            apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
            apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
            apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
            apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
        )
        self.model = apply_lora_to_model(self.model, lora_config)

        # Load LoRA weights
        print(f"💾 Loading LoRA weights from {weights_path}...")
        load_lora_weights(self.model, weights_path)

        self.model.to(self.device)
        self.model.eval()

        # Setup transforms (official SAM3 pattern)
        self.transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(
                    sizes=resolution,
                    max_size=resolution,
                    square=True,
                    consistent_transform=False
                ),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # Setup postprocessor
        # Note: Using simpler manual postprocessing instead of PostProcessImage
        # because PostProcessImage may have additional filtering logic
        self.use_manual_postprocess = True

        print("✅ SAM3 + LoRA ready for inference!\n")

    def create_datapoint(self, pil_image: PILImage.Image, text_prompts: List[str]) -> Datapoint:
        """
        Create a SAM3 datapoint from image and text prompts.

        Args:
            pil_image: PIL Image
            text_prompts: List of text queries

        Returns:
            Datapoint with image and queries
        """
        w, h = pil_image.size

        # Create SAM Image
        sam_image = SAMImage(
            data=pil_image,
            objects=[],
            size=[h, w]
        )

        # Create queries for each text prompt
        queries = []
        for idx, text_query in enumerate(text_prompts):
            query = FindQueryLoaded(
                query_text=text_query,
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=idx,
                inference_metadata=InferenceMetadata(
                    coco_image_id=idx,
                    original_image_id=idx,
                    original_category_id=1,
                    original_size=[w, h],
                    object_id=0,
                    frame_index=0,
                )
            )
            queries.append(query)

        return Datapoint(
            find_queries=queries,
            images=[sam_image]
        )

    def create_datapoint_with_boxes(self, pil_image: PILImage.Image, boxes: List[List[float]], text_prompts: Optional[List[str]] = None) -> Datapoint:
        """
        Create a SAM3 datapoint from image and bounding boxes.

        Args:
            pil_image: PIL Image
            boxes: List of bounding boxes in format [x1, y1, x2, y2] (pixel coordinates)
            text_prompts: Optional list of text labels for each box

        Returns:
            Datapoint with image and box prompts
        """
        from sam3.train.data.sam3_image_dataset import Object
        import torch
        
        w, h = pil_image.size
        
        # Create objects from boxes
        objects = []
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            # Normalize to [0, 1]
            bbox_norm = torch.tensor([x1/w, y1/h, x2/w, y2/h], dtype=torch.float32)
            
            # Calculate area for Object
            area = (x2 - x1) * (y2 - y1)
            
            obj = Object(
                bbox=bbox_norm,
                area=area,
                object_id=idx,
                frame_index=0,
                segment=None,  # No mask for box prompt
                is_crowd=False
            )
            objects.append(obj)
        
        # Create SAM Image with objects
        sam_image = SAMImage(
            data=pil_image,
            objects=objects,
            size=[h, w]
        )
        
        # Create query
        if text_prompts is None:
            text_prompts = ["object"] * len(boxes)
        elif len(text_prompts) == 1 and len(boxes) > 1:
            # Repeat single prompt for all boxes
            text_prompts = text_prompts * len(boxes)
        
        queries = []
        for idx, text_query in enumerate(text_prompts[:len(boxes)]):
            query = FindQueryLoaded(
                query_text=text_query,
                image_id=0,
                object_ids_output=[idx],  # Link to the object
                is_exhaustive=True,
                query_processing_order=idx,
                inference_metadata=InferenceMetadata(
                    coco_image_id=idx,
                    original_image_id=idx,
                    original_category_id=1,
                    original_size=[w, h],
                    object_id=idx,
                    frame_index=0,
                )
            )
            queries.append(query)
        
        return Datapoint(
            find_queries=queries,
            images=[sam_image]
        )

    @torch.no_grad()
    def predict(
        self,
        image_path: str,
        text_prompts: List[str]
    ) -> dict:
        """
        Run inference on an image with text prompts.

        Args:
            image_path: Path to input image
            text_prompts: List of text queries (e.g., ["crack", "defect"])

        Returns:
            Dictionary mapping prompt index to predictions:
            {
                0: {'boxes': [...], 'scores': [...], 'masks': [...]},
                1: {'boxes': [...], 'scores': [...], 'masks': [...]}
            }
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        pil_image = PILImage.open(image_path).convert("RGB")
        print(f"📷 Loaded image: {image_path}")
        print(f"   Size: {pil_image.size}")
        print(f"   Prompts: {text_prompts}")

        print("\n🔮 Running inference...")

        results = {}

        # Process each prompt separately (SAM3 expects one query per forward pass)
        for query_idx, prompt in enumerate(text_prompts):
            # Create datapoint with single prompt
            datapoint = self.create_datapoint(pil_image, [prompt])

            # Apply transforms
            datapoint = self.transform(datapoint)

            # Collate into batch
            batch = collate_fn_api([datapoint], dict_key="input")["input"]

            # Move to device
            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            # Forward pass
            outputs = self.model(batch)

            # Manual post-processing
            last_output = outputs[-1]
            pred_logits = last_output['pred_logits']  # [batch, num_queries, num_classes]
            pred_boxes = last_output['pred_boxes']    # [batch, num_queries, 4]
            pred_masks = last_output.get('pred_masks', None)  # [batch, num_queries, H, W]

            # Get probabilities
            out_probs = pred_logits.sigmoid()  # [batch, num_queries, num_classes]

            # Get scores for this query
            scores = out_probs[0, :, :].max(dim=-1)[0]  # [num_queries]

            # Filter by threshold
            keep = scores > self.detection_threshold
            num_keep = keep.sum().item()

            if num_keep > 0:
                # Get boxes and convert from cxcywh to xyxy
                boxes_cxcywh = pred_boxes[0, keep]  # [num_keep, 4]
                kept_scores = scores[keep]
                cx, cy, w, h = boxes_cxcywh.unbind(-1)

                # Convert to xyxy and scale to original image size
                orig_w, orig_h = pil_image.size
                x1 = (cx - w / 2) * orig_w
                y1 = (cy - h / 2) * orig_h
                x2 = (cx + w / 2) * orig_w
                y2 = (cy + h / 2) * orig_h

                boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

                # Apply NMS to remove overlapping boxes
                keep_nms = nms(boxes_xyxy, kept_scores, self.nms_iou_threshold)
                boxes_xyxy = boxes_xyxy[keep_nms]
                kept_scores = kept_scores[keep_nms]
                num_keep = len(keep_nms)

                # Get masks and resize to original size
                if pred_masks is not None:
                    # Apply NMS filtering to masks too
                    masks_small = pred_masks[0, keep][keep_nms].sigmoid() > 0.5  # [num_keep_nms, H, W]

                    # Resize masks to original image size
                    import torch.nn.functional as F
                    masks_resized = F.interpolate(
                        masks_small.unsqueeze(0).float(),
                        size=(orig_h, orig_w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0) > 0.5

                    masks_np = masks_resized.cpu().numpy()
                else:
                    masks_np = None

                results[query_idx] = {
                    'prompt': prompt,
                    'boxes': boxes_xyxy.cpu().numpy(),
                    'scores': kept_scores.cpu().numpy(),
                    'masks': masks_np,
                    'num_detections': num_keep
                }
                print(f"   '{prompt}': {num_keep} detections after NMS (max score: {kept_scores.max().item():.3f})")
            else:
                results[query_idx] = {
                    'prompt': prompt,
                    'boxes': None,
                    'scores': None,
                    'masks': None,
                    'num_detections': 0
                }
                print(f"   '{prompt}': 0 detections")

        # Store original image for visualization
        results['_image'] = pil_image

        return results

    @torch.no_grad()
    def predict_with_boxes(
        self,
        image_path: str,
        boxes: List[List[float]],
        text_prompts: Optional[List[str]] = None
    ) -> dict:
        """
        Run inference on an image with bounding box prompts.

        Args:
            image_path: Path to input image
            boxes: List of bounding boxes in format [[x1, y1, x2, y2], ...]
                   Coordinates are in pixels (not normalized)
            text_prompts: Optional list of text labels for each box

        Returns:
            Dictionary with predictions for each box
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        pil_image = PILImage.open(image_path).convert("RGB")
        print(f"📷 Loaded image: {image_path}")
        print(f"   Size: {pil_image.size}")
        print(f"   Number of box prompts: {len(boxes)}")
        if text_prompts:
            print(f"   Text labels: {text_prompts}")

        print("\n🔮 Running inference with box prompts...")

        # Create datapoint with boxes
        datapoint = self.create_datapoint_with_boxes(pil_image, boxes, text_prompts)

        # Apply transforms
        datapoint = self.transform(datapoint)

        # Collate into batch
        batch = collate_fn_api([datapoint], dict_key="input")["input"]

        # Move to device
        batch = copy_data_to_device(batch, self.device, non_blocking=True)

        # Forward pass
        outputs = self.model(batch)

        # Manual post-processing
        last_output = outputs[-1]
        pred_logits = last_output['pred_logits']  # [batch, num_queries, num_classes]
        pred_boxes = last_output['pred_boxes']    # [batch, num_queries, 4]
        pred_masks = last_output.get('pred_masks', None)  # [batch, num_queries, H, W]

        # Get probabilities
        out_probs = pred_logits.sigmoid()  # [batch, num_queries, num_classes]

        # Get scores
        scores = out_probs[0, :, :].max(dim=-1)[0]  # [num_queries]

        # Filter by threshold
        keep = scores > self.detection_threshold
        num_keep = keep.sum().item()

        results = {}

        if num_keep > 0:
            # Get boxes and convert from cxcywh to xyxy
            boxes_cxcywh = pred_boxes[0, keep]  # [num_keep, 4]
            kept_scores = scores[keep]
            cx, cy, w, h = boxes_cxcywh.unbind(-1)

            # Convert to xyxy and scale to original image size
            orig_w, orig_h = pil_image.size
            x1 = (cx - w / 2) * orig_w
            y1 = (cy - h / 2) * orig_h
            x2 = (cx + w / 2) * orig_w
            y2 = (cy + h / 2) * orig_h

            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

            # Apply NMS to remove overlapping boxes
            keep_nms = nms(boxes_xyxy, kept_scores, self.nms_iou_threshold)
            boxes_xyxy = boxes_xyxy[keep_nms]
            kept_scores = kept_scores[keep_nms]
            num_keep = len(keep_nms)

            # Get masks and resize to original size
            if pred_masks is not None:
                masks_small = pred_masks[0, keep][keep_nms].sigmoid() > 0.5  # [num_keep_nms, H, W]

                # Resize masks to original image size
                import torch.nn.functional as F
                masks_resized = F.interpolate(
                    masks_small.unsqueeze(0).float(),
                    size=(orig_h, orig_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0) > 0.5

                masks_np = masks_resized.cpu().numpy()
            else:
                masks_np = None

            results[0] = {
                'prompt': 'box_prompts',
                'boxes': boxes_xyxy.cpu().numpy(),
                'scores': kept_scores.cpu().numpy(),
                'masks': masks_np,
                'num_detections': num_keep
            }
            print(f"   {num_keep} detections after NMS (max score: {kept_scores.max().item():.3f})")
        else:
            results[0] = {
                'prompt': 'box_prompts',
                'boxes': None,
                'scores': None,
                'masks': None,
                'num_detections': 0
            }
            print(f"   0 detections")

        # Store original image and input boxes for visualization
        results['_image'] = pil_image
        results['_input_boxes'] = boxes

        return results

    def visualize(
        self,
        results: dict,
        output_path: str,
        show_boxes: bool = True,
        show_masks: bool = True
    ):
        """
        Visualize predictions on image.

        Args:
            results: Results from predict()
            output_path: Where to save visualization
            show_boxes: Whether to show bounding boxes
            show_masks: Whether to show segmentation masks
        """
        pil_image = results['_image']

        # Create figure
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(pil_image)

        # Colors for different prompts
        colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta']

        # Draw input boxes if available (for box prompt mode)
        if '_input_boxes' in results:
            input_boxes = results['_input_boxes']
            for box in input_boxes:
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                rect = patches.Rectangle(
                    (x1, y1), width, height,
                    linewidth=2,
                    edgecolor='yellow',
                    facecolor='none',
                    linestyle='--',
                    label='Input Box'
                )
                ax.add_patch(rect)

        total_detections = 0

        # Draw results for each prompt (filter out special keys that start with '_')
        result_keys = [k for k in results.keys() if not str(k).startswith('_')]
        for idx in sorted(result_keys):
            result = results[idx]
            prompt = result['prompt']
            color = colors[idx % len(colors)]

            if result['num_detections'] == 0:
                continue

            total_detections += result['num_detections']

            boxes = result['boxes']
            scores = result['scores']
            masks = result['masks']

            for i in range(result['num_detections']):
                # Draw mask
                if show_masks and masks is not None:
                    mask = masks[i]
                    colored_mask = np.zeros((*mask.shape, 4))
                    # Use different colors for different prompts
                    if color == 'red':
                        colored_mask[mask] = [1, 0, 0, 0.4]
                    elif color == 'blue':
                        colored_mask[mask] = [0, 0, 1, 0.4]
                    elif color == 'green':
                        colored_mask[mask] = [0, 1, 0, 0.4]
                    else:
                        colored_mask[mask] = [1, 1, 0, 0.4]
                    ax.imshow(colored_mask)

                # Draw box
                if show_boxes and boxes is not None:
                    box = boxes[i]  # [x1, y1, x2, y2]
                    x1, y1, x2, y2 = box

                    # Clamp to image bounds
                    img_w, img_h = pil_image.size
                    x1 = max(0, min(img_w, x1))
                    y1 = max(0, min(img_h, y1))
                    x2 = max(0, min(img_w, x2))
                    y2 = max(0, min(img_h, y2))

                    width = x2 - x1
                    height = y2 - y1

                    # Draw rectangle
                    rect = patches.Rectangle(
                        (x1, y1), width, height,
                        linewidth=2,
                        edgecolor=color,
                        facecolor='none'
                    )
                    ax.add_patch(rect)

                    # Add label
                    score = scores[i] if scores is not None else 0
                    label = f"{prompt}: {score:.2f}"
                    ax.text(
                        x1, y1 - 5,
                        label,
                        bbox=dict(facecolor=color, alpha=0.5),
                        fontsize=10,
                        color='white'
                    )

        ax.axis('off')

        # Add title with all prompts
        result_keys = [k for k in results.keys() if not str(k).startswith('_')]
        prompts_str = ", ".join([f'"{results[k]["prompt"]}"' for k in sorted(result_keys)])
        plt.suptitle(f'Prompts: {prompts_str}', fontsize=12, y=0.98)

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()

        print(f"\n✅ Saved visualization to {output_path}")
        print(f"   Total detections: {total_detections}")


def main():
    parser = argparse.ArgumentParser(description="SAM3 + LoRA Inference")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to LoRA weights (auto-detected if not provided)"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs='+',
        default=["object"],
        help='Text prompt(s) to guide segmentation (e.g., "crack" or "crack" "defect")'
    )
    parser.add_argument(
        "--box",
        type=float,
        nargs='+',
        default=None,
        help='Bounding box prompt(s) in format: x1 y1 x2 y2 [x1 y1 x2 y2 ...] '
             'Multiple boxes are separated by spaces. Example: --box 100 100 200 200 300 300 400 400'
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Output visualization path"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection confidence threshold"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1008,
        help="Input resolution (default: 1008)"
    )
    parser.add_argument(
        "--boundingbox",
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=False,
        help="Show bounding boxes: True or False (default: False)"
    )
    parser.add_argument(
        "--no-masks",
        action="store_true",
        help="Don't show segmentation masks"
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.5,
        help="NMS IoU threshold (default: 0.5, lower = fewer overlapping boxes)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/model/sam3_pth/sam3pt/sam3.pt",
        help="Path to SAM3 base model checkpoint (optional, will use local path if available)"
    )

    args = parser.parse_args()

    # Initialize model
    inferencer = SAM3LoRAInference(
        config_path=args.config,
        weights_path=args.weights,
        resolution=args.resolution,
        detection_threshold=args.threshold,
        nms_iou_threshold=args.nms_iou,
        checkpoint_path=args.checkpoint
    )

    # Run inference - choose mode based on input
    if args.box is not None:
        # Box prompt mode
        if len(args.box) % 4 != 0:
            raise ValueError(f"Box coordinates must be in groups of 4 (x1 y1 x2 y2). Got {len(args.box)} values.")
        
        # Parse boxes
        boxes = []
        for i in range(0, len(args.box), 4):
            boxes.append(args.box[i:i+4])
        
        print(f"\n📦 Box Prompt Mode: {len(boxes)} box(es)")
        results = inferencer.predict_with_boxes(args.image, boxes, args.prompt)
    else:
        # Text prompt mode
        print(f"\n💬 Text Prompt Mode")
        results = inferencer.predict(args.image, args.prompt)

    # Visualize
    inferencer.visualize(
        results,
        args.output,
        show_boxes=args.boundingbox,
        show_masks=not args.no_masks
    )

    # Print summary
    print("\n" + "="*60)
    print("📊 Summary:")
    result_keys = [k for k in results.keys() if not str(k).startswith('_')]
    for idx in sorted(result_keys):
        result = results[idx]
        print(f"   Prompt '{result['prompt']}': {result['num_detections']} detections")
        if result['num_detections'] > 0 and result['scores'] is not None:
            print(f"      Max confidence: {result['scores'].max():.3f}")
    print("="*60)


if __name__ == "__main__":
    main()
