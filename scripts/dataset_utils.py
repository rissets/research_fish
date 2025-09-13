#!/usr/bin/env python3
"""
Utilities untuk mengelola fish dataset dan training
Author: AI Assistant
Date: 2025-09-13
"""

import os
import sys
import yaml
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class DatasetUtils:
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    def analyze_dataset(self, dataset_path):
        """Analyze dataset structure and statistics"""
        dataset_path = Path(dataset_path)
        
        print(f"📊 Analyzing dataset: {dataset_path}")
        print("-" * 50)
        
        # Find data.yaml
        data_yaml = dataset_path / "data.yaml"
        if not data_yaml.exists():
            print("❌ data.yaml not found")
            return None
        
        with open(data_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"Dataset configuration:")
        print(f"  Classes: {config.get('nc', 'Unknown')}")
        print(f"  Names: {config.get('names', 'Unknown')}")
        
        # Analyze splits
        splits = ['train', 'val', 'test']
        stats = {}
        
        for split in splits:
            if split in config:
                split_path = dataset_path / config[split]
                if split_path.exists():
                    images = []
                    for ext in self.supported_formats:
                        images.extend(list(split_path.glob(f"*{ext}")))
                    
                    labels_path = split_path.parent / split_path.name.replace('images', 'labels')
                    labels = list(labels_path.glob("*.txt")) if labels_path.exists() else []
                    
                    stats[split] = {
                        'images': len(images),
                        'labels': len(labels),
                        'path': str(split_path)
                    }
                    
                    print(f"  {split.capitalize()}:")
                    print(f"    Images: {len(images)}")
                    print(f"    Labels: {len(labels)}")
                    print(f"    Path: {split_path}")
        
        return stats
    
    def create_sample_visualization(self, dataset_path, output_dir="./dataset_preview", num_samples=9):
        """Create visualization of dataset samples"""
        dataset_path = Path(dataset_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Load config
        data_yaml = dataset_path / "data.yaml"
        with open(data_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get train images
        train_path = dataset_path / config.get('train', 'train/images')
        if not train_path.exists():
            print(f"❌ Train path not found: {train_path}")
            return
        
        images = []
        for ext in self.supported_formats:
            images.extend(list(train_path.glob(f"*{ext}")))
        
        if len(images) < num_samples:
            num_samples = len(images)
        
        # Create grid
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle(f'Dataset Preview: {dataset_path.name}', fontsize=16)
        
        selected_images = np.random.choice(images, num_samples, replace=False)
        
        for i, img_path in enumerate(selected_images):
            row = i // 3
            col = i % 3
            
            # Load and display image
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[row, col].imshow(img)
            axes[row, col].set_title(img_path.name, fontsize=10)
            axes[row, col].axis('off')
        
        # Hide unused subplots
        for i in range(num_samples, 9):
            row = i // 3
            col = i % 3
            axes[row, col].axis('off')
        
        plt.tight_layout()
        preview_path = output_dir / f"{dataset_path.name}_preview.png"
        plt.savefig(preview_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Dataset preview saved: {preview_path}")
        return str(preview_path)
    
    def convert_to_yolo_format(self, source_dir, output_dir, class_names=None):
        """Convert dataset to YOLO format"""
        source_dir = Path(source_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        if not class_names:
            class_names = ['fish']  # Default
        
        # Create directory structure
        for split in ['train', 'val', 'test']:
            (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Create data.yaml
        data_yaml_content = {
            'path': str(output_dir),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(class_names),
            'names': class_names
        }
        
        with open(output_dir / 'data.yaml', 'w') as f:
            yaml.dump(data_yaml_content, f, default_flow_style=False)
        
        print(f"✅ YOLO format structure created at: {output_dir}")
        return str(output_dir / 'data.yaml')
    
    def split_dataset(self, dataset_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """Split dataset into train/val/test"""
        dataset_path = Path(dataset_path)
        
        # Get all images
        images = []
        for ext in self.supported_formats:
            images.extend(list(dataset_path.glob(f"**/*{ext}")))
        
        # Shuffle
        np.random.shuffle(images)
        
        # Calculate splits
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        print(f"Dataset split:")
        print(f"  Total: {n_total}")
        print(f"  Train: {len(train_images)} ({len(train_images)/n_total:.1%})")
        print(f"  Val: {len(val_images)} ({len(val_images)/n_total:.1%})")
        print(f"  Test: {len(test_images)} ({len(test_images)/n_total:.1%})")
        
        return {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }

class ModelUtils:
    def __init__(self):
        pass
    
    def compare_models(self, model_paths, test_images_dir):
        """Compare multiple trained models"""
        from ultralytics import YOLO
        
        results = {}
        test_images_dir = Path(test_images_dir)
        
        # Get test images
        test_images = []
        for ext in ['.jpg', '.jpeg', '.png']:
            test_images.extend(list(test_images_dir.glob(f"*{ext}")))
        
        if not test_images:
            print("❌ No test images found")
            return results
        
        test_images = test_images[:5]  # Limit to 5 images
        
        for model_path in model_paths:
            print(f"\n🔍 Testing model: {model_path}")
            model = YOLO(model_path)
            
            model_results = []
            for img_path in test_images:
                result = model(str(img_path))
                model_results.append(result)
            
            results[model_path] = model_results
            print(f"✅ Completed testing {model_path}")
        
        return results
    
    def export_model(self, model_path, export_formats=['onnx', 'tflite']):
        """Export model to different formats"""
        from ultralytics import YOLO
        
        model = YOLO(model_path)
        exported = {}
        
        for fmt in export_formats:
            try:
                print(f"📤 Exporting to {fmt}...")
                exported_path = model.export(format=fmt)
                exported[fmt] = exported_path
                print(f"✅ Exported to {fmt}: {exported_path}")
            except Exception as e:
                print(f"❌ Failed to export to {fmt}: {e}")
        
        return exported

def create_training_script(dataset_path, output_script="train_custom.py"):
    """Create custom training script for specific dataset"""
    dataset_path = Path(dataset_path)
    
    # Analyze dataset
    utils = DatasetUtils()
    stats = utils.analyze_dataset(dataset_path)
    
    if not stats:
        print("❌ Could not analyze dataset")
        return
    
    # Load config
    data_yaml = dataset_path / "data.yaml"
    with open(data_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    # Calculate recommended batch size based on dataset size
    total_images = sum(split_data['images'] for split_data in stats.values())
    
    if total_images < 100:
        batch_size = 4
        epochs = 50
    elif total_images < 1000:
        batch_size = 8
        epochs = 100
    else:
        batch_size = 16
        epochs = 200
    
    # Generate script content
    script_content = f'''#!/usr/bin/env python3
"""
Custom training script for {dataset_path.name}
Generated automatically
"""

from train_fish_segmentation import FishSegmentationTrainer

def main():
    # Dataset info
    dataset_path = "{dataset_path}"
    
    # Training parameters optimized for this dataset
    trainer = FishSegmentationTrainer(model_name="yolov8n-seg.pt")
    
    results = trainer.train(
        data_path=dataset_path,
        epochs={epochs},
        batch_size={batch_size},
        imgsz=640,
        patience=30,
        # Custom parameters for fish detection
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
    )
    
    print("🎉 Training completed!")

if __name__ == "__main__":
    main()
'''
    
    with open(output_script, 'w') as f:
        f.write(script_content)
    
    print(f"✅ Custom training script created: {output_script}")
    print(f"   Dataset: {dataset_path.name}")
    print(f"   Total images: {total_images}")
    print(f"   Recommended batch size: {batch_size}")
    print(f"   Recommended epochs: {epochs}")
    
    return output_script

def main():
    print("🛠 Dataset and Model Utilities")
    print("=" * 50)
    
    import argparse
    parser = argparse.ArgumentParser(description='Dataset and Model Utilities')
    parser.add_argument('--analyze', type=str, help='Analyze dataset')
    parser.add_argument('--preview', type=str, help='Create dataset preview')
    parser.add_argument('--create-script', type=str, help='Create custom training script')
    parser.add_argument('--convert', type=str, help='Convert dataset to YOLO format')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    
    args = parser.parse_args()
    
    utils = DatasetUtils()
    
    if args.analyze:
        utils.analyze_dataset(args.analyze)
    
    if args.preview:
        utils.create_sample_visualization(args.preview, args.output)
    
    if args.create_script:
        create_training_script(args.create_script)
    
    if args.convert:
        utils.convert_to_yolo_format(args.convert, args.output)

if __name__ == "__main__":
    main()
