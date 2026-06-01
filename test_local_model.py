#!/usr/bin/env python3
"""
快速测试脚本 - 验证本地 SAM3 模型加载
"""

import os
import torch

def test_local_model():
    """测试本地模型文件"""
    
    checkpoint_path = "/home/data-1/yjs/niu/code/sam3/pth/sam3.pt"
    
    print("=" * 60)
    print("SAM3 本地模型测试")
    print("=" * 60)
    
    # 1. 检查文件存在
    print(f"\n1. 检查文件存在性")
    print(f"   路径: {checkpoint_path}")
    
    if os.path.exists(checkpoint_path):
        print(f"   ✓ 文件存在")
        
        # 获取文件大小
        file_size = os.path.getsize(checkpoint_path)
        file_size_gb = file_size / (1024**3)
        print(f"   文件大小: {file_size_gb:.2f} GB")
        
        if file_size_gb < 0.1:
            print(f"   ⚠️  警告: 文件太小，可能不完整")
            return False
    else:
        print(f"   ✗ 文件不存在")
        print(f"\n请确保模型文件在正确位置：")
        print(f"   {checkpoint_path}")
        return False
    
    # 2. 尝试加载模型
    print(f"\n2. 尝试加载模型...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        print(f"   ✓ 模型加载成功")
        
        # 显示模型信息
        if isinstance(checkpoint, dict):
            print(f"\n3. 模型信息:")
            print(f"   包含的键: {list(checkpoint.keys())[:10]}")
            
            # 计算参数量
            if 'model' in checkpoint:
                model_state = checkpoint['model']
            else:
                model_state = checkpoint
                
            total_params = 0
            for key, value in model_state.items():
                if isinstance(value, torch.Tensor):
                    total_params += value.numel()
            
            print(f"   总参数量: {total_params:,} ({total_params/1e6:.1f}M)")
        
        print(f"\n✓ 所有测试通过！")
        print(f"\n你可以开始训练了：")
        print(f"   cd /home/data-1/yjs/niu/code/SAM3_LoRA")
        print(f"   python train_sam3_lora_native.py --config configs/full_lora_config.yaml")
        
        return True
        
    except Exception as e:
        print(f"   ✗ 模型加载失败")
        print(f"   错误: {e}")
        return False

if __name__ == "__main__":
    success = test_local_model()
    print("\n" + "=" * 60)
    exit(0 if success else 1)
