# AGENTS.md

**Generated:** 2026-07-13 02:11 UTC
**Commit:** 664bb03
**Branch:** main

## Project overview

SAM3-LoRA fine-tunes Meta's SAM3 segmentation model using Low-Rank Adaptation. The SAM3 model source lives in `sam3/` (not pip-installed). Two parallel LoRA implementations exist; use `lora_layers.py` (the primary one).

## Structure

```
SAM3_LoRA/
├── train_sam3_lora_native.py   # CANONICAL trainer (1336 lines)
├── validate_sam3_lora.py       # Full eval: mAP, cgF1, NMS (1257 lines)
├── infer_sam.py                # Inference CLI (266 lines)
├── lora_layers.py              # PRIMARY LoRA impl (541 lines)
├── prepare_data.py             # COCO data prep
├── convert_roboflow_to_coco.py # Roboflow → COCO
├── configs/                    # YAML configs (6 presets)
├── sam3/                       # SAM3 model source (vendored, not pip)
├── sam3_lora/                  # ALTERNATIVE trainer package (avoid)
├── src/                        # LEGACY alternative lora layout (avoid)
├── export/                     # ONNX/HF model export pipeline
├── script/                     # ONNX eval scripts
├── dataset/                    # COCO datasets (liupo, lajiaoyou, 透明勺子)
├── outputs/                    # Training outputs (weights, viz)
└── weight_output/              # LoRA weight checkpoints
```

## Where to look

| Task | Location | Notes |
|------|----------|-------|
| Train model | `train_sam3_lora_native.py` | Canonical; other `train_*.py` are stale |
| Evaluate mAP/cgF1 | `validate_sam3_lora.py` | Post-training; uses NMS |
| Run inference | `infer_sam.py` | Supports multi-prompt, NMS, bounding boxes |
| LoRA injection logic | `lora_layers.py` | `apply_lora_to_model()`, `LoRAConfig` |
| Config presets | `configs/` | Use `data_dir` schema, not `train_data_path` |
| Export to ONNX/HF | `export/` | 3-step: merge → convert → export |
| SAM3 model internals | `sam3/` | Vendored source; `build_sam3_image_model()` |
| Alternative trainer | `sam3_lora/` | Self-contained package; AVOID, use root scripts |

## Code map

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `SAM3TrainerNative` | class | `train_sam3_lora_native.py` | Main trainer; DDP-aware |
| `LoRAConfig` | dataclass | `lora_layers.py` | rank, alpha, target_modules, component flags |
| `apply_lora_to_model()` | func | `lora_layers.py` | Injects LoRA into SAM3 modules |
| `MultiheadAttentionLoRA` | class | `lora_layers.py` | Replaces nn.MHA for Q/K/V LoRA |
| `save_lora_weights()` | func | `lora_layers.py` | Saves only LoRA params (~10-50MB) |
| `load_lora_weights()` | func | `lora_layers.py` | Loads LoRA params into model |
| `count_parameters()` | func | `lora_layers.py` | Trainable/total stats |
| `SAM3LoRAInference` | class | `infer_sam.py` | Inference engine with NMS |
| `COCOSegmentDataset` | class | `validate_sam3_lora.py` | COCO dataset for eval |
| `setup_distributed()` | func | `train_sam3_lora_native.py` | DDP init |

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

## Python environment

**Always activate the conda env before running any Python command** (torch/sam3/vllm etc. are not in the base shell):

```bash
conda activate sam3_6000d
```

All `python`/`python3` invocations for training, validation, inference, and data scripts in this repo must run inside `sam3_6000d`. If you see `ModuleNotFoundError: No module named 'torch'`, you forgot to activate it.

## Config system

Configs live in `configs/` (YAML). The only config directory used by the main trainer is `configs/`; `sam3_lora_configs/` is for the alternative `sam3_lora/` package trainer and should be avoided.

Key config sections: `model`, `lora`, `training`, `output`, `evaluation`, `hardware`.

**Schema gotcha**: The main trainer reads `training.data_dir` (path to root with train/valid subdirs). Some older configs use `training.train_data_path` / `training.val_data_path` instead — those won't work with `train_sam3_lora_native.py`.

Config preset matrix:

| Config | rank | target_modules | Components enabled | Schema | Status |
|--------|------|----------------|--------------------|--------|--------|
| `full_lora_config.yaml` | 16 | qkv, proj | vision only | `data_dir` | ✓ Works |
| `light_lora_config.yaml` | 16 | q/k/v/out_proj | DETR enc+dec, mask | `data_dir` | ✓ Works |
| `base_config.yaml` | 8 | q/k/v/out_proj | vision, text, DETR enc+dec | `train_data_path` | ✗ Legacy |
| `minimal_lora_config.yaml` | 4 | q/k/v_proj | DETR decoder only | `train_data_path` | ✗ Legacy |
| `crack_detection_config.yaml` | varies | varies | varies | check file | Domain-specific |
| `sam3_lora_standalone.yaml` | varies | varies | varies | for `sam3_lora/` package | Alternative only |

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

## Anti-patterns (this project)

- **DO NOT** use `sam3_lora/` package trainer — use root `train_sam3_lora_native.py` instead.
- **DO NOT** use `src/` directory lora — use root `lora_layers.py` instead.
- **DO NOT** use configs with `train_data_path`/`val_data_path` schema with the native trainer — only `data_dir` works.
- **DO NOT** use `sam3_lora_configs/` — use `configs/` only.
- **NEVER** set `num_workers > 0` unless you have plenty of shared memory.
- **NEVER** expect loss < 1.0 — SAM3 uses weighted multi-component loss; 100-160 is normal.
- **NEVER** push to `origin` — always push to `upstream`.
- **NEVER** modify `sam3/` — it's vendored SAM3 source, not our code.

## Unique styles

- Commit messages in Chinese.
- Configs use Chinese comments for production-tuned parameters (see active section of `full_lora_config.yaml`).
- LoRA target module names vary by component: `qkv`/`proj` (ViT), `q_proj`/`k_proj`/`v_proj` (transformer), `c_fc`/`c_proj` (CLIP), `linear1`/`linear2` (FFN).
- `MultiheadAttentionLoRA` replaces `nn.MultiheadAttention` to enable LoRA on separate Q/K/V projections (fused in_proj split into 3).
- Multi-GPU auto-launches `torchrun` subprocess via `--device` arg (no manual torchrun needed).
- `full_lora_config.yaml` contains a large commented-out "full" config + an active minimal config optimized for small datasets (vision encoder only).

## Common gotchas

- `gradient_accumulation_steps` compounds with batch_size; effective batch = batch_size × accumulation × num_gpus.
- `lora.rank`, `lora.alpha`, and `lora.target_modules` must match the actual SAM3 layer names (see Unique styles for the naming map).
- The model's `build_sam3_image_model()` sets TF32 on Ampere GPUs automatically.
- `.gitignore` excludes `*.pt`, `*.png`, `*.jpg`, and `data/` — checkpoint and data files won't be tracked.

## Git conventions

- **Always push to `upstream`** (not `origin`): `git push upstream main`
- Remote info: `origin` = `https://github.com/Sompote/SAM3_LoRA.git`, `upstream` = `git@github.com:zhoutian160930/SAM3_LoRA.git`
- Commit messages should be in Chinese for this project
