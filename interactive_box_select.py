#!/usr/bin/env python3
"""
交互式选择边界框工具

使用鼠标在图像上选择边界框，然后用这些框进行推理
需要安装: pip install opencv-python
"""

import cv2
import numpy as np
import argparse
import sys

class BoxSelector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        self.display_image = self.image.copy()
        self.boxes = []
        self.current_box = None
        self.drawing = False
        self.start_point = None
        
        # 创建窗口
        cv2.namedWindow('Select Boxes (Press SPACE to finish, R to reset, Q to quit)')
        cv2.setMouseCallback('Select Boxes (Press SPACE to finish, R to reset, Q to quit)', self.mouse_callback)
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.current_box = [x, y, x, y]
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_box[2] = x
                self.current_box[3] = y
                self.update_display()
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.current_box[2] = x
            self.current_box[3] = y
            
            # 确保x1 < x2, y1 < y2
            x1, y1, x2, y2 = self.current_box
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # 至少要有一些大小
            if x2 - x1 > 10 and y2 - y1 > 10:
                self.boxes.append([x1, y1, x2, y2])
                print(f"添加边界框 #{len(self.boxes)}: [{x1}, {y1}, {x2}, {y2}]")
            
            self.current_box = None
            self.update_display()
    
    def update_display(self):
        self.display_image = self.image.copy()
        
        # 绘制已确认的框
        for idx, box in enumerate(self.boxes):
            x1, y1, x2, y2 = box
            cv2.rectangle(self.display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(self.display_image, f"Box {idx+1}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 绘制当前正在画的框
        if self.current_box is not None:
            x1, y1, x2, y2 = self.current_box
            cv2.rectangle(self.display_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # 添加说明文字
        help_text = [
            "Instructions:",
            "- Click and drag to draw boxes",
            "- Press SPACE to finish and run inference",
            "- Press R to reset all boxes",
            "- Press Q to quit without inference"
        ]
        y_offset = 30
        for text in help_text:
            cv2.putText(self.display_image, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        cv2.imshow('Select Boxes (Press SPACE to finish, R to reset, Q to quit)', self.display_image)
    
    def run(self):
        print("="*70)
        print("交互式边界框选择")
        print("="*70)
        print("\n操作说明:")
        print("  1. 用鼠标点击并拖动来绘制边界框")
        print("  2. 可以绘制多个边界框")
        print("  3. 按 SPACE 键完成选择并运行推理")
        print("  4. 按 R 键重置所有框")
        print("  5. 按 Q 键退出")
        print("\n")
        
        self.update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # SPACE - 完成选择
                if len(self.boxes) > 0:
                    print(f"\n✅ 已选择 {len(self.boxes)} 个边界框")
                    break
                else:
                    print("⚠️  至少需要选择一个边界框")
            
            elif key == ord('r') or key == ord('R'):  # R - 重置
                self.boxes = []
                self.current_box = None
                print("已重置所有边界框")
                self.update_display()
            
            elif key == ord('q') or key == ord('Q'):  # Q - 退出
                print("用户取消")
                cv2.destroyAllWindows()
                return None
        
        cv2.destroyAllWindows()
        return self.boxes


def main():
    parser = argparse.ArgumentParser(description="交互式选择边界框并推理")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--config", type=str, default="configs/full_lora_config.yaml", help="配置文件")
    parser.add_argument("--output", type=str, default="box_interactive_result.png", help="输出图像路径")
    parser.add_argument("--threshold", type=float, default=0.3, help="检测阈值")
    parser.add_argument("--prompt", type=str, nargs="+", default=["object"], help="文本提示词")
    
    args = parser.parse_args()
    
    try:
        # 选择边界框
        selector = BoxSelector(args.image)
        boxes = selector.run()
        
        if boxes is None or len(boxes) == 0:
            print("未选择边界框，退出")
            return
        
        # 构建推理命令
        print("\n" + "="*70)
        print("运行 SAM3 LoRA 推理")
        print("="*70)
        
        box_args = []
        for box in boxes:
            box_args.extend([str(int(coord)) for coord in box])
        
        cmd = [
            "python", "infer_sam.py",
            "--config", args.config,
            "--image", args.image,
            "--box", *box_args,
            "--prompt", *args.prompt,
            "--boundingbox", "True",
            "--threshold", str(args.threshold),
            "--output", args.output
        ]
        
        print(f"\n执行命令:")
        print(" ".join(cmd))
        print()
        
        import subprocess
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print(f"\n✅ 推理完成！结果保存为: {args.output}")
            
            # 显示结果
            print("\n是否显示结果图像？(y/n): ", end='')
            response = input().strip().lower()
            if response == 'y':
                result_image = cv2.imread(args.output)
                if result_image is not None:
                    cv2.imshow('Inference Result (Press any key to close)', result_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        else:
            print(f"\n❌ 推理失败，返回码: {result.returncode}")
    
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
