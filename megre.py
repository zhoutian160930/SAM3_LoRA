#!/usr/bin/env python3
"""
将 hz_mask 文件夹中的 JSON 标注转换为 COCO 格式
"""
import json
import os
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np


def polygon_to_bbox(polygon_points):
    """将多边形转换为边界框 [x, y, width, height]"""
    x_coords = [p[0] for p in polygon_points]
    y_coords = [p[1] for p in polygon_points]
    
    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)
    
    width = x_max - x_min
    height = y_max - y_min
    
    return [x_min, y_min, width, height]


def polygon_to_area(polygon_points):
    """计算多边形面积"""
    polygon_points = np.array(polygon_points, dtype=np.float32)
    area = cv2.contourArea(polygon_points)
    return float(area)


def convert_hz_to_coco(input_dir, output_file):
    """
    将 hz_mask 格式转换为 COCO 格式
    
    Args:
        input_dir: 包含 .json 和对应 .jpg 文件的目录
        output_file: 输出的 COCO 格式 JSON 文件路径
    """
    input_path = Path(input_dir)
    
    # 初始化 COCO 格式结构
    coco_format = {
        "info": {
            "description": "Hz Mask Dataset in COCO Format",
            "url": "",
            "version": "1.0",
            "year": 2026,
            "contributor": "",
            "date_created": datetime.now().strftime("%Y/%m/%d")
        },
        "licenses": [{
            "id": 1,
            "name": "Unknown",
            "url": ""
        }],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # 收集所有类别
    categories_set = set()
    json_files = sorted(input_path.glob("*.json"))
    
    print(f"Found {len(json_files)} JSON files")
    
    # 第一遍扫描：收集所有类别
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for shape in data.get('shapes', []):
            label = shape.get('label', '1')
            categories_set.add(label)
    
    # 创建类别映射
    categories_list = sorted(list(categories_set))
    category_id_map = {}
    
    for idx, category_name in enumerate(categories_list, start=1):
        coco_format['categories'].append({
            "id": idx,
            "name": category_name,
            "supercategory": "object"
        })
        category_id_map[category_name] = idx
    
    print(f"Categories: {categories_list}")
    
    # 第二遍扫描：处理图像和标注
    image_id = 1
    annotation_id = 1
    
    for json_file in json_files:
        # 查找对应的图像文件
        image_name = json_file.stem + '.jpg'
        image_path = json_file.parent / image_name
        
        if not image_path.exists():
            print(f"Warning: Image not found for {json_file.name}, skipping...")
            continue
        
        # 读取图像获取尺寸
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Warning: Cannot read image {image_path}, skipping...")
                continue
            height, width = img.shape[:2]
        except Exception as e:
            print(f"Error reading {image_path}: {e}")
            continue
        
        # 添加图像信息
        image_info = {
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": image_name,
            "license": 1,
            "date_captured": ""
        }
        coco_format['images'].append(image_info)
        
        # 读取标注
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 处理每个标注对象
        for shape in data.get('shapes', []):
            label = shape.get('label', '1')
            points = shape.get('points', [])
            
            if len(points) < 3:
                print(f"Warning: Shape in {json_file.name} has less than 3 points, skipping...")
                continue
            
            # 计算边界框和面积
            bbox = polygon_to_bbox(points)
            area = polygon_to_area(points)
            
            # 将多边形转换为 COCO segmentation 格式 (flat list)
            segmentation = []
            for point in points:
                segmentation.extend(point)
            
            # 添加标注
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id_map[label],
                "segmentation": [segmentation],
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            }
            coco_format['annotations'].append(annotation)
            annotation_id += 1
        
        print(f"Processed: {image_name} (image_id={image_id}, {len(data.get('shapes', []))} annotations)")
        image_id += 1
    
    # 保存 COCO 格式文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(coco_format, f, indent=2, ensure_ascii=False)
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"Conversion completed!")
    print(f"{'='*60}")
    print(f"Output file: {output_file}")
    print(f"Total images: {len(coco_format['images'])}")
    print(f"Total annotations: {len(coco_format['annotations'])}")
    print(f"Total categories: {len(coco_format['categories'])}")
    print(f"\nCategories:")
    for cat in coco_format['categories']:
        count = sum(1 for ann in coco_format['annotations'] if ann['category_id'] == cat['id'])
        print(f"  - {cat['name']} (id={cat['id']}): {count} annotations")


if __name__ == "__main__":
    import sys
    
    # 默认参数
    input_dir = "/home/model/work/SAM3_LoRA/data/valid"
    output_file = "/home/model/work/SAM3_LoRA/data/valid/_annotations.coco.json"
    
    # 支持命令行参数
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}\n")
    
    convert_hz_to_coco(input_dir, output_file)
