#!/usr/bin/env python3
"""
SAM3 LoRA 框提示推理示例

演示如何使用边界框提示进行推理
"""

import subprocess
import sys

def print_section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def run_command(description, command):
    print(f"\n{description}")
    print(f"命令: {' '.join(command)}")
    print("-"*70)
    
    result = subprocess.run(command)
    return result.returncode == 0

def main():
    print_section("SAM3 LoRA 框提示推理示例")
    
    # 检查测试图像
    test_image = "/home/data-1/yjs/niu/code/sam3/hz_mask/00000111_751_1256_1328_317_244_100.000000.jpg"
    
    print(f"\n使用测试图像: {test_image}")
    print("\n边界框格式: x1 y1 x2 y2 (左上角和右下角坐标，单位：像素)")
    
    # 示例1: 单个框提示
    print_section("示例 1: 单个边界框提示")
    print("""
假设图像中有一个物体在区域 (100, 100) 到 (400, 400)
我们使用这个边界框来提示模型分割该区域的对象
    """)
    
    cmd1 = [
        "python", "infer_sam.py",
        "--config", "configs/full_lora_config.yaml",
        "--image", test_image,
        "--box", "100", "100", "400", "400",
        "--boundingbox", "True",
        "--output", "box_prompt_single.png",
        "--threshold", "0.3"
    ]
    
    if run_command("运行单框推理:", cmd1):
        print("✅ 成功！结果保存为: box_prompt_single.png")
    
    # 示例2: 多个框提示
    print_section("示例 2: 多个边界框提示")
    print("""
同时使用多个边界框来分割不同区域的对象
框1: (50, 50, 200, 200)
框2: (300, 300, 500, 500)
框3: (600, 100, 800, 300)
    """)
    
    cmd2 = [
        "python", "infer_sam.py",
        "--config", "configs/full_lora_config.yaml",
        "--image", test_image,
        "--box", "50", "50", "200", "200", "300", "300", "500", "500", "600", "100", "800", "300",
        "--boundingbox", "True",
        "--output", "box_prompt_multiple.png",
        "--threshold", "0.3"
    ]
    
    if run_command("运行多框推理:", cmd2):
        print("✅ 成功！结果保存为: box_prompt_multiple.png")
    
    # 示例3: 框提示 + 文本标签
    print_section("示例 3: 边界框 + 文本标签")
    print("""
结合边界框和文本提示词
框提示告诉模型在哪里找对象
文本提示告诉模型找什么类型的对象
    """)
    
    cmd3 = [
        "python", "infer_sam.py",
        "--config", "configs/full_lora_config.yaml",
        "--image", test_image,
        "--box", "100", "100", "400", "400",
        "--prompt", "crack",
        "--boundingbox", "True",
        "--output", "box_prompt_with_text.png",
        "--threshold", "0.3"
    ]
    
    if run_command("运行框+文本推理:", cmd3):
        print("✅ 成功！结果保存为: box_prompt_with_text.png")
    
    # 使用技巧
    print_section("使用技巧")
    print("""
1. 获取边界框坐标的方法:
   - 使用图像查看器/编辑器测量像素坐标
   - 使用 labelme 等标注工具
   - 使用 Python PIL/OpenCV 交互式选择
   - 从已有的检测结果转换

2. 边界框格式:
   --box x1 y1 x2 y2 [x1 y1 x2 y2 ...]
   - (x1, y1): 左上角坐标
   - (x2, y2): 右下角坐标
   - 单位: 像素（不是归一化的）
   - 多个框用空格分隔

3. 参数调整:
   --threshold 0.3    # 降低阈值获得更多检测结果
   --threshold 0.7    # 提高阈值获得更精确的结果
   --nms-iou 0.3      # 降低IoU阈值去除更多重叠框
   
4. 可视化选项:
   --boundingbox True # 显示预测的边界框（虚线黄色为输入框）
   --no-masks         # 只显示框，不显示掩码

5. 适用场景:
   ✓ 已知物体大致位置，需要精确分割
   ✓ 多个物体需要分别处理
   ✓ 交互式标注/编辑工作流
   ✓ 结合其他检测器的结果
    """)
    
    # Python API 示例
    print_section("Python API 使用示例")
    print("""
如果你想在代码中使用框提示推理：

```python
from infer_sam import SAM3LoRAInference

# 初始化模型
inferencer = SAM3LoRAInference(
    config_path="configs/full_lora_config.yaml",
    detection_threshold=0.3,
    nms_iou_threshold=0.5
)

# 定义边界框 (x1, y1, x2, y2)
boxes = [
    [100, 100, 400, 400],  # 框1
    [500, 200, 700, 600],  # 框2
]

# 可选：为每个框添加文本标签
text_labels = ["crack", "defect"]

# 运行推理
results = inferencer.predict_with_boxes(
    image_path="image.jpg",
    boxes=boxes,
    text_prompts=text_labels  # 可选
)

# 可视化结果
inferencer.visualize(
    results,
    output_path="output.png",
    show_boxes=True,
    show_masks=True
)

# 访问结果
num_detections = results[0]['num_detections']
pred_boxes = results[0]['boxes']      # numpy array [N, 4]
pred_scores = results[0]['scores']    # numpy array [N]
pred_masks = results[0]['masks']      # numpy array [N, H, W]
```
    """)
    
    print_section("完成")
    print("""
✅ 所有示例演示完成！

生成的文件:
- box_prompt_single.png     : 单框提示结果
- box_prompt_multiple.png   : 多框提示结果  
- box_prompt_with_text.png  : 框+文本提示结果

查看这些图像来了解框提示的效果。
虚线黄色框是你输入的提示框。
实线框和彩色掩码是模型的预测结果。
    """)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        sys.exit(1)
