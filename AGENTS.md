# AGENTS.md

## Project overview

SAM3-LoRA fine-tunes Meta's SAM3 segmentation model using Low-Rank Adaptation. The SAM3 model source lives in `sam3/` (not pip-installed). Two parallel LoRA implementations exist; use `lora_layers.py` (the primary one).

## Entrypoint scripts

- **Training**: `train_sam3_lora_native.py` (canonical; many other `train_*.py` files are stale)
- **Inference**: `infer_sam.py`
- **Full evaluation**: `validate_sam3_lora.py` (mAP, cgF1, NMS)
- **Data prep**: `prepare_data.py`, `prepare_data_split.py`, `convert_roboflow_to_coco.py`

## Installation & model access

```bash
pip install -e .
hf auth login  # must have facebook/sam3 access granted
```

The SAM3 checkpoint must exist locally or be downloadable from HuggingFace. The trainer's fallback chain: `--checkpoint` arg → `config["model"]["checkpoint_path"]` → hardcoded `/home/data-1/yjs/niu/code/sam3/pth/sam3.pt` → HF download. The BPE file at `sam3/assets/bpe_simple_vocab_16e6.txt.gz` is required for model building.

## Config system

Configs live in `configs/` (YAML). The only config directory used by the main trainer is `configs/`; `sam3_lora_configs/` is for the alternative `sam3_lora/` package trainer and should be avoided.

Key config sections: `model`, `lora`, `training`, `output`, `evaluation`, `hardware`.

**Schema gotcha**: The main trainer reads `training.data_dir` (path to root with train/valid subdirs). Some older configs use `training.train_data_path` / `training.val_data_path` instead — those won't work with `train_sam3_lora_native.py`.

## Data format

COCO JSON in subdirectories:
```
data_dir/
├── train/
│   ├── image001.jpg
│   └── _annotations.coco.json
├── valid/
│   └── ...
└── test/
    └── ...
```

Each split folder must contain `_annotations.coco.json` alongside images.

## Training commands

```bash
# Single GPU
python3 train_sam3_lora_native.py --config configs/full_lora_config.yaml

# Multi-GPU (auto-launches torchrun subprocess)
python3 train_sam3_lora_native.py --config configs/full_lora_config.yaml --device 0 1

# With explicit checkpoint path
python3 train_sam3_lora_native.py --config configs/full_lora_config.yaml --checkpoint /path/to/sam3.pt
```

Validation uses a local checkpoint by preference. Set `num_workers: 0` to avoid shared memory issues. Use `bf16` mixed precision on Ampere+ GPUs; `fp16` on older hardware.

## Loss values are normal

Training losses of 100–160 are expected due to heavily weighted loss components (loss_mask=200x, loss_ce=20x, loss_dice=10x, etc.). Monitor the trend (should decrease), not the absolute value.

## Validation: two-phase design

- **During training**: Only validation loss is computed (fast).
- **After training**: `validate_sam3_lora.py` computes mAP, cgF1 metrics with proper NMS.

```bash
python3 validate_sam3_lora.py \
  --config configs/full_lora_config.yaml \
  --weights outputs/sam3_lora_full/best_lora_weights.pt \
  --val_data_dir /path/to/data/valid
```

## Checkpoint outputs

Training saves two files: `best_lora_weights.pt` (lowest val loss) and `last_lora_weights.pt` (latest epoch). If no validation data exists, both files are copies of the last epoch.

## Inference

```bash
python3 infer_sam.py \
  --config configs/full_lora_config.yaml \
  --image test.jpg \
  --prompt "crack" \
  --output result.png
```

Supports multiple text prompts: `--prompt "crack" "defect" "damage"`. NMS is applied at `--nms-iou` (default 0.5). Use `--boundingbox True` to show boxes.

## Common gotchas

- `num_workers` should stay at 0 unless you have plenty of shared memory.
- `gradient_accumulation_steps` compounds with batch_size; effective batch = batch_size × accumulation × num_gpus.
- `lora.rank`, `lora.alpha`, and `lora.target_modules` must match the actual layer names in the SAM3 architecture (they vary: `qkv`/`proj` for ViT, `q_proj`/`k_proj`/`v_proj` for transformer, `c_fc`/`c_proj` for CLIP text, `linear1`/`linear2` for FFN).
- The model's `build_sam3_image_model()` sets TF32 on Ampere GPUs automatically.
- `.gitignore` excludes `*.pt`, `*.png`, `*.jpg`, and `data/` — checkpoint and data files won't be tracked.

## Git conventions

- **Always push to `upstream`** (not `origin`): `git push upstream main`
- Remote info: `origin` = `https://github.com/Sompote/SAM3_LoRA.git`, `upstream` = `git@github.com:zhoutian160930/SAM3_LoRA.git`
- Commit messages should be in Chinese for this project
