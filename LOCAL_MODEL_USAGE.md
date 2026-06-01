# 使用本地模型训练 SAM3 LoRA

## ✅ 修改完成

已成功修改 `train_sam3_lora_native.py`，现在支持使用本地模型文件而不是从 HuggingFace 下载。

---

## 🎯 模型加载优先级

程序会按以下优先级查找模型：

1. **命令行参数** `--checkpoint`（最高优先级）
2. **配置文件** 中的 `model.checkpoint_path`
3. **默认本地路径** `/home/data-1/yjs/niu/code/sam3/pth/sam3.pt`（自动检测）
4. **HuggingFace 下载**（最后备用方案）

---

## 🚀 使用方法

### 方式1：自动使用本地模型（推荐）⭐

由于你的模型已在默认路径 `/home/data-1/yjs/niu/code/sam3/pth/sam3.pt`，直接运行即可：

```bash
cd /home/data-1/yjs/niu/code/SAM3_LoRA

# 单GPU训练（自动使用本地模型）
python train_sam3_lora_native.py --config configs/full_lora_config.yaml

# 多GPU训练
python train_sam3_lora_native.py --config configs/full_lora_config.yaml --device 0 1
```

**输出示例：**
```
Using local checkpoint: /home/data-1/yjs/niu/code/sam3/pth/sam3.pt
Building SAM3 model...
Loading model from: /home/data-1/yjs/niu/code/sam3/pth/sam3.pt
```

---

### 方式2：显式指定模型路径

如果模型在其他位置，使用 `--checkpoint` 参数：

```bash
python train_sam3_lora_native.py \
    --config configs/full_lora_config.yaml \
    --checkpoint /path/to/your/sam3.pt
```

---

### 方式3：在配置文件中指定

编辑 `configs/full_lora_config.yaml`：

```yaml
model:
  name: "facebook/sam3"
  cache_dir: null
  checkpoint_path: "/home/data-1/yjs/niu/code/sam3/pth/sam3.pt"  # 添加这一行
```

然后运行：
```bash
python train_sam3_lora_native.py --config configs/full_lora_config.yaml
```

---

## 📦 你的本地模型文件

路径：`/home/data-1/yjs/niu/code/sam3/pth/`

文件列表：
```
✓ sam3.pt                    # 主模型文件（约3GB）
✓ config.json                # 模型配置
✓ tokenizer.json             # 分词器
✓ vocab.json                 # 词汇表
✓ merges.txt                 # BPE合并规则
✓ model.safetensors          # SafeTensors格式
✓ processor_config.json      # 处理器配置
✓ special_tokens_map.json    # 特殊token
✓ tokenizer_config.json      # 分词器配置
```

---

## 🔍 验证模型文件

### 检查文件存在性
```bash
ls -lh /home/data-1/yjs/niu/code/sam3/pth/sam3.pt
```

预期输出：
```
-rw-r--r-- 1 user user 3.0G ... sam3.pt
```

### 测试加载
```python
import torch
import os

checkpoint_path = "/home/data-1/yjs/niu/code/sam3/pth/sam3.pt"

if os.path.exists(checkpoint_path):
    print(f"✓ 文件存在: {checkpoint_path}")
    
    # 尝试加载
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print(f"✓ 模型加载成功！")
    print(f"  包含的键: {list(checkpoint.keys())[:5]}...")
else:
    print(f"✗ 文件不存在: {checkpoint_path}")
```

---

## 🎓 完整训练示例

### 单GPU训练
```bash
cd /home/data-1/yjs/niu/code/SAM3_LoRA

# 使用默认本地模型
python train_sam3_lora_native.py \
    --config configs/full_lora_config.yaml \
    --device 0

# 或显式指定
python train_sam3_lora_native.py \
    --config configs/full_lora_config.yaml \
    --checkpoint /home/data-1/yjs/niu/code/sam3/pth/sam3.pt \
    --device 0
```

### 多GPU训练
```bash
# 使用2个GPU（0和1）
python train_sam3_lora_native.py \
    --config configs/full_lora_config.yaml \
    --device 0 1

# 使用4个GPU
python train_sam3_lora_native.py \
    --config configs/full_lora_config.yaml \
    --device 0 1 2 3
```

### 使用不同配置
```bash
# 最小LoRA配置（更快）
python train_sam3_lora_native.py \
    --config configs/minimal_lora_config.yaml

# 完整LoRA配置（更强）
python train_sam3_lora_native.py \
    --config configs/full_lora_config.yaml

# 裂缝检测配置
python train_sam3_lora_native.py \
    --config configs/crack_detection_config.yaml
```

---

## 🐛 常见问题

### Q1: 提示 "No local checkpoint found"
**原因**: 文件路径不正确或文件不存在

**解决**: 
```bash
# 检查文件
ls -lh /home/data-1/yjs/niu/code/sam3/pth/sam3.pt

# 如果路径不同，使用 --checkpoint 指定
python train_sam3_lora_native.py \
    --config configs/full_lora_config.yaml \
    --checkpoint /actual/path/to/sam3.pt
```

### Q2: 仍然尝试从 HuggingFace 下载
**原因**: 检测到本地文件但加载失败

**解决**: 检查文件完整性
```bash
# 检查文件大小（应该约3GB）
du -h /home/data-1/yjs/niu/code/sam3/pth/sam3.pt
```

### Q3: CUDA out of memory
**原因**: GPU显存不足

**解决**: 减小batch_size
```yaml
# 编辑 configs/full_lora_config.yaml
training:
  batch_size: 2  # 从4改为2
```

---

## 📊 预期输出

成功加载本地模型时的输出：

```
Using single GPU: 0
Using local checkpoint: /home/data-1/yjs/niu/code/sam3/pth/sam3.pt
Building SAM3 model...
Loading model from: /home/data-1/yjs/niu/code/sam3/pth/sam3.pt
Applying LoRA...
Trainable params: 42,123,456 (1.23%)
Loading training data...
Loaded COCO dataset: train split
  Images: 778
  Annotations: 1234
  Categories: {...}
Epoch 1/100 [━━━━━━━━━━] 100%
```

---

## ✨ 优势

✅ **无需访问 HuggingFace** - 使用本地模型文件  
✅ **自动检测** - 智能查找默认路径  
✅ **灵活配置** - 支持多种指定方式  
✅ **后备方案** - 找不到本地文件时才下载  
✅ **无需修改代码** - 通过参数控制  

---

## 📝 注意事项

1. **模型文件大小**: 确保 `sam3.pt` 约3GB（完整模型）
2. **文件权限**: 确保有读取权限
3. **磁盘空间**: 训练过程会保存checkpoints，预留足够空间
4. **GPU显存**: 至少16GB（单GPU + batch_size=4）

---

## 🎉 开始训练

现在你可以直接开始训练了：

```bash
cd /home/data-1/yjs/niu/code/SAM3_LoRA
python train_sam3_lora_native.py --config configs/full_lora_config.yaml
```

祝训练顺利！🚀
