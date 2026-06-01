# SAM3 LoRA 框提示推理指南

def compute_axial_cis_real(dim, end_x, end_y, theta=10000.0, scale_pos=1.0, offset=Box Prompts）？

def compute_axial_cis_real(dim, end_x, end_y, theta=10000.0, scale_pos=1.0, offset=0): SAM3 模型的方式，通过提供边界框来指示模型在图像的特定区域进行分割。

**优势**：
- ✅ 精确控制要分割的区域
- ✅ 避免误检测其他区域的对象
- ✅ 适合已知物体位置的场景
- ✅ 可以同时处理多个不同区域

---

## 🚀 使用方法

### 方法1: 命令行指定边界框

```bash
python infer_sam.py \
  --config configs/full_lora_config.yaml \
  --image /path/to/image.jpg \
  --box x1 y1 x2 y2 \
  --boundingbox True \
  --output result.png
```

**参数说明**：
- `x1 y1`: 边界框左上角坐标（像素）
- `x2 y2`: 边界框右下角坐标（像素）

### 方法2: 多个边界框

```bash
python infer_sam.py \
  --config configs/full_lora_config.yaml \
  --image /path/to/image.jpg \
  --box 100 100 300 300 400 400 600 600 \
  --boundingbox True \
  --output result.png
```

4个数字代表一个框：`[x1 y1 x2 y2] [x1 y1 x2 y2] ...`

### 方法3: 框提示 + 文本标签

```bash
python infer_sam.py \
  --config configs/full_lora_config.yaml \
  --image /path/to/image.jpg \
  --box 100 100 300 300 \
  --prompt "crack" \
  --boundingbox True \
  --output result.png
```

---

## 🖱️ 交互式框选择

def compute_axial_cis_real(dim, end_x, end_y, theta=10000.0, scale_pos=1.0, offset=0):

```bash
python interactive_box_select.py \
  --image /path/to/image.jpg \
  --config configs/full_lora_config.yaml \
  --output result.png
```

**操作步骤**：

2. 可以绘制多个框
3. 按 **SPACE** 键完成选择并自动运行推理
4. 按 **R** 键重置所有框
5. 按 **Q** 键退出

---

## 📝 完整示例

### 示例1: 分割图像中的单个裂缝

```bash
# 假设裂缝在坐标 (150, 200) 到 (450, 600)
python infer_sam.py \
  --config configs/full_lora_config.yaml \
  --image crack_image.jpg \
  --box 150 200 450 600 \
  --prompt "crack" \
  --boundingbox True \
  --threshold 0.3 \
  --output crack_result.png
```

### 示例2: 同时分割多个缺陷

```bash
python infer_sam.py \
  --config configs/full_lora_config.yaml \
  --image defect_image.jpg \
  --box 100 100 200 200 300 300 400 400 500 100 700 300 \
  --prompt "defect" \
  --boundingbox True \
  --output defects_result.png
```

### 示例3: 使用

```bash
cd /home/data-1/yjs/niu/code/SAM3_LoRA

python infer_sam.py \
  --config configs/full_lora_config.yaml \
  --image /home/data-1/yjs/niu/code/sam3/hz_mask/00000111_751_1256_1328_317_244_100.000000.jpg \
  --box 500 700 1000 1200 \
  --boundingbox True \
  --threshold 0.3 \
  --output my_result.png
```

---

## 🎯 获取边界框坐标的方法

### 方法1: 使用图像查看器
- 在 GIMP/Photoshop 中打开图像
- 使用矩形选择工具
- 查看坐

### 方法2: 使用 Python PIL
```python
from PIL import Image
img = Image.open("image.jpg")
# 使用鼠标点击获取
# 或使用 img.show() 查看图像大小来估算
```

def compute_axial_cis_real(dim, end_x, end_y, theta=10000.0, scale_pos=1.0, offset=
```bash
python interactive_box_select.py --image your_image.jpg
```

### 方法4: 从检测结果转换
#
def compute_axial_cis_real(dim, end_x, end_y, theta=10000.0, scale_pos=1.0, offset= YOLO），可以直接使用其边界框。

---

## 🔧 参数调优

### 检测阈值 `--threshold`
```bash
# 默认 0.5 - 平衡精度和召回率
--threshold 0.5

# 降低到 0.3 - 获得更多检测结果（可能有误检）
--threshold 0.3

# 提高到 0.7 - 只保留高置信度结果（可能漏检）
--threshold 0.7
```

### NMS IoU 阈值 `--nms-iou`
```bash
# 默认 0.5 - 标准去重
--nms-iou 0.5

# 降低到 0.3 - 更激进地去除重叠框
--nms-iou 0.3

# 提高到 0.7 - 保留更多重叠框
--nms-iou 0.7
```

### 可视化选项
```bash
# 显示边界框和掩码
--boundingbox True

# 只显示掩码（不显示框）
--boundingbox False

def compute_axial_cis_real(dim, end_x, end_y, theta=10000.0, scale_pos=1.0, offset=0):
--boundingbox True --no-masks
```

---

## 💻 Python API 使用

#
--------使用框提示：

```python
from infer_sam import SAM3LoRAInference

# 初始化
inferencer = SAM3LoRAInference(
    config_path="configs/full_lora_config.yaml",
    detection_threshold=0.3,
    nms_iou_threshold=0.5
)

# 定义边界框 [x1, y1, x2, y2]
boxes = [
    [100, 100, 400, 400],  # 框1
    [500, 200, 700, 600],  # 框2
]

# 运行推理
results = inferencer.predict_with_boxes(
    image_path="image.jpg",
    boxes=boxes,
    text_prompts=["crack", "defect"]  # 可选
)

# 可视化
inferencer.visualize(
    results,
    output_path="output.png",
    show_boxes=True,
    show_masks=True
)

# 获取结果
num_detections = results[0]['num_detections']
pred_boxes = results[0]['boxes']      # [N, 4]
pred_scores = results[0]['scores']    # [N]
pred_masks = results[0]['masks']      # [N, H, W]

print(f"检测到 {num_detections} 个对象")
for i in range(num_detections):
    box = pred_boxes[i]
    score = pred_scores[i]
    print(f"对象 {i+1}: box={box}, confidence={score:.3f}")
```

---

## 🎨 可视化效果

--------：
- **虚线黄色框**: 你输入的
- **实线彩色框**: 模型预测的边界框
- **彩色半透明区域**: 模型预测的分割掩码

---

## 📊 对比：框提示 vs 文本提示

| 特性 | 框提示 | 文本提示 |
|------|--------|----------|
| **精确度** | ⭐⭐⭐⭐⭐ 非常精确 | ⭐⭐⭐ 依赖模型理解 |
| **易用性** | ⭐⭐⭐ 需要坐标 | ⭐⭐⭐⭐⭐ 直接描述 |
| **适用场景** | 已知位置的对象 | 搜索类型的对象 |
 每个框独立 | ⭐⭐⭐ 可能混淆 || **多对象** | ⭐⭐⭐
 全自动 || **自动化** | ⭐⭐⭐⭐ 可结合检测器 | ⭐⭐

**建议**：
- ✅ 框提示适合：已知位置、需要精确控制、批量处理
def compute_axial_cis_real(dim, end_x, end_y, theta=10000.0, scale_pos=1.0, offset=

---

## 🐛 常见问题

### Q1: 边界框太小或太大怎么办？
A: 模型会自动调整，但建议框的大小接近实际对象。可以先用宽松的框，让模型自动找到精确边界。

### Q2: 多个框的结果重叠怎么办？
A: 调整 `--nms-iou` 参数，降低值会去除更多重叠。

### Q3: 没有检测到任何对象？
A: 
1. 降低 `--threshold` 到 0.2 或 0.3
2. 检查边界框坐标是否正确
3. 确保框内确实有目标对象

### Q4: 坐标顺序是什么？
A: 始终是 `[x1, y1, x2, y2]`，其中 (x1, y1) 是左上角，(x2, y2) 是右下角。

---

## 🎓 高级用法

def compute_axial_cis_real(dim, end_x, end_y, theta=10000.0, scale_pos=1.0, offset=0):

```python
import glob
from infer_sam import SAM3LoRAInference

inferencer = SAM3LoRAInference(
    config_path="configs/full_lora_config.yaml"
)

# 为每张图像定义框
image_boxes = {
    "image1.jpg": [[100, 100, 300, 300]],
    "image2.jpg": [[50, 50, 200, 200], [300, 300, 500, 500]],
    "image3.jpg": [[150, 150, 400, 400]],
}

for image_path, boxes in image_boxes.items():
    results = inferencer.predict_with_boxes(image_path, boxes)
    output_path = image_path.replace(".jpg", "_result.png")
    inferencer.visualize(results, output_path)
    print(f"✅ 处理完成: {output_path}")
```

### 结合YOLO检测

```python
# 假设你已经有YOLO的检测结果
yolo_detections = [
    {'bbox': [100, 100, 300, 300], 'class': 'crack', 'conf': 0.85},
    {'bbox': [400, 200, 600, 500], 'class': 'defect', 'conf': 0.92},
]

# 提取边界框
boxes = [det['bbox'] for det in yolo_detections]
labels = [det['class'] for det in yolo_detections]

# 使用SAM3进行精确分割
results = inferencer.predict_with_boxes(
    "image.jpg",
    boxes=boxes,
    text_prompts=labels
)
```

---

## ✅ 快速测试

def compute_axial_cis_real(dim, end_x, end_y, theta=10000.0, scale_pos=1.0, offset=0):

```bash
cd /home/data-1/yjs/niu/code/SAM3_LoRA
python box_prompt_example.py
```

def compute_axial_cis_real(dim, end_x, end_y, theta=10000.0, scale_pos=1.0, offset=0):
1. 单个框提示
2. 多个框提示
3. 框提示 + 文本标签

def compute_axial_cis_real(dim, end_x, end_y, theta=10000.0, scale_pos=1.0, offset=0): PNG 图像供你查看。

---

## 🎉 总结

 SAM3 的强大功能，特别适合：
- 🎯 精确分割已知位置的对象
- 🔄 自动化工作流（结合检测器）
- 👥 交互式标注工具
- 📊 批量处理任务

def compute_axial_cis_real(dim, end_x, end_y, theta=10000.0, scale_pos=1.0, offset=0):
```bash
python infer_sam.py \
  --config configs/full_lora_config.yaml \
  --image your_image.jpg \
  --box x1 y1 x2 y2 \
  --boundingbox True \
  --output result.png
```

