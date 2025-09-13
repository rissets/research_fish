#!/usr/bin/env python3
"""
Script untuk mengkonversi FishImgDataset ke format YOLO
Author: AI Assistant
Date: 2025-09-13
"""

import os
import shutil
import yaml
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict

class FishDatasetConverter:
    def __init__(self, source_path, output_path):
        self.source_path = Path(source_path)
        self.output_path = Path(output_path)
        self.class_names = []
        self.class_to_id = {}
        
    def scan_classes(self):
        """Scan untuk mendapatkan semua nama kelas"""
        train_dir = self.source_path / "train"
        
        if not train_dir.exists():
            raise FileNotFoundError(f"Train directory tidak ditemukan: {train_dir}")
        
        classes = []
        for class_dir in sorted(train_dir.iterdir()):
            if class_dir.is_dir():
                classes.append(class_dir.name)
        
        self.class_names = classes
        self.class_to_id = {name: idx for idx, name in enumerate(classes)}
        
        print(f"✅ Ditemukan {len(classes)} kelas:")
        for idx, name in enumerate(classes):
            print(f"  {idx}: {name}")
        
        return classes
    
    def convert_classification_to_detection(self):
        """Convert classification dataset ke detection format"""
        print(f"\n🔧 Converting dataset dari {self.source_path} ke {self.output_path}")
        
        # Create output directory structure
        self.output_path.mkdir(exist_ok=True)
        
        splits = ['train', 'val', 'test']
        for split in splits:
            (self.output_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Process each split
        total_processed = 0
        stats = defaultdict(lambda: defaultdict(int))
        
        for split in splits:
            source_split_dir = self.source_path / split
            if not source_split_dir.exists():
                print(f"⚠️  {split} directory tidak ditemukan, skip")
                continue
            
            target_images_dir = self.output_path / split / 'images'
            target_labels_dir = self.output_path / split / 'labels'
            
            split_count = 0
            
            # Process each class
            for class_dir in source_split_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                
                class_name = class_dir.name
                if class_name not in self.class_to_id:
                    print(f"⚠️  Unknown class: {class_name}, skip")
                    continue
                
                class_id = self.class_to_id[class_name]
                class_images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                
                print(f"  Processing {split}/{class_name}: {len(class_images)} images")
                
                for img_path in class_images:
                    # Copy image dengan nama yang unique
                    new_img_name = f"{class_name}_{img_path.name}"
                    new_img_path = target_images_dir / new_img_name
                    
                    try:
                        # Copy image
                        shutil.copy2(img_path, new_img_path)
                        
                        # Create label file
                        # Untuk classification to detection, kita buat bounding box yang cover seluruh image
                        img = cv2.imread(str(img_path))
                        if img is None:
                            print(f"❌ Gagal membaca image: {img_path}")
                            continue
                        
                        height, width = img.shape[:2]
                        
                        # Create bounding box annotation (normalized coordinates)
                        # Format YOLO: class_id center_x center_y width height
                        center_x = 0.5  # Center of image
                        center_y = 0.5  # Center of image
                        bbox_width = 0.9  # 90% of image width
                        bbox_height = 0.9  # 90% of image height
                        
                        # Create label file
                        label_name = new_img_name.rsplit('.', 1)[0] + '.txt'
                        label_path = target_labels_dir / label_name
                        
                        with open(label_path, 'w') as f:
                            f.write(f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}\n")
                        
                        split_count += 1
                        stats[split][class_name] += 1
                        
                    except Exception as e:
                        print(f"❌ Error processing {img_path}: {e}")
                        continue
            
            total_processed += split_count
            print(f"✅ {split}: {split_count} images processed")
        
        print(f"\n📊 Dataset conversion completed!")
        print(f"Total images processed: {total_processed}")
        
        # Print statistics
        print(f"\nStatistics per split:")
        for split in splits:
            if split in stats:
                total_split = sum(stats[split].values())
                print(f"  {split}: {total_split} images")
                for class_name, count in sorted(stats[split].items()):
                    print(f"    {class_name}: {count}")
        
        return stats
    
    def create_data_yaml(self):
        """Create data.yaml configuration file"""
        data_config = {
            'path': str(self.output_path),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        yaml_path = self.output_path / 'data.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ data.yaml created: {yaml_path}")
        print(f"   Classes: {len(self.class_names)}")
        print(f"   Names: {self.class_names[:5]}..." if len(self.class_names) > 5 else f"   Names: {self.class_names}")
        
        return str(yaml_path)
    
    def create_visualization(self, num_samples=9):
        """Create dataset visualization"""
        import matplotlib.pyplot as plt
        
        print(f"\n📊 Creating dataset visualization...")
        
        # Get sample images from each class
        train_images_dir = self.output_path / 'train' / 'images'
        if not train_images_dir.exists():
            print("❌ Train images directory tidak ditemukan")
            return
        
        # Get random samples
        all_images = list(train_images_dir.glob("*.jpg")) + list(train_images_dir.glob("*.png"))
        if len(all_images) < num_samples:
            num_samples = len(all_images)
        
        selected_images = np.random.choice(all_images, num_samples, replace=False)
        
        # Create visualization
        rows = int(np.sqrt(num_samples))
        cols = int(np.ceil(num_samples / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        fig.suptitle(f'Fish Dataset Preview - {len(self.class_names)} Classes', fontsize=16)
        
        if rows == 1:
            axes = [axes]
        if cols == 1:
            axes = [[ax] for ax in axes]
        
        for i, img_path in enumerate(selected_images):
            row = i // cols
            col = i % cols
            
            # Load and display image
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Extract class name from filename
                class_name = img_path.name.split('_')[0]
                
                axes[row][col].imshow(img)
                axes[row][col].set_title(f'{class_name}', fontsize=10)
                axes[row][col].axis('off')
        
        # Hide unused subplots
        for i in range(num_samples, rows * cols):
            row = i // cols
            col = i % cols
            axes[row][col].axis('off')
        
        plt.tight_layout()
        preview_path = self.output_path / 'dataset_preview.png'
        plt.savefig(preview_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Dataset preview saved: {preview_path}")
        return str(preview_path)
    
    def analyze_dataset(self):
        """Analyze converted dataset"""
        print(f"\n📈 Dataset Analysis")
        print("-" * 50)
        
        splits = ['train', 'val', 'test']
        total_images = 0
        
        for split in splits:
            images_dir = self.output_path / split / 'images'
            labels_dir = self.output_path / split / 'labels'
            
            if images_dir.exists():
                images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
                labels = list(labels_dir.glob("*.txt")) if labels_dir.exists() else []
                
                print(f"{split.capitalize()}:")
                print(f"  Images: {len(images)}")
                print(f"  Labels: {len(labels)}")
                print(f"  Match: {'✅' if len(images) == len(labels) else '❌'}")
                
                total_images += len(images)
        
        print(f"\nTotal images: {total_images}")
        print(f"Total classes: {len(self.class_names)}")
        print(f"Average images per class: {total_images / len(self.class_names):.1f}")

def main():
    print("🐟 Fish Dataset Converter - Classification to YOLO Detection")
    print("=" * 70)
    
    # Paths
    source_path = "/mnt/arch-data/data/research_od/datasets/fish_dataset/FishImgDataset"
    output_path = "/mnt/arch-data/data/research_od/datasets/fish_dataset_yolo"
    
    # Initialize converter
    converter = FishDatasetConverter(source_path, output_path)
    
    try:
        # Scan classes
        classes = converter.scan_classes()
        
        # Convert dataset
        stats = converter.convert_classification_to_detection()
        
        # Create data.yaml
        yaml_path = converter.create_data_yaml()
        
        # Create visualization
        preview_path = converter.create_visualization()
        
        # Analyze dataset
        converter.analyze_dataset()
        
        print(f"\n🎉 Conversion completed successfully!")
        print(f"📁 Output dataset: {output_path}")
        print(f"📄 Configuration: {yaml_path}")
        print(f"🖼️  Preview: {preview_path}")
        print(f"\n📝 Ready for training:")
        print(f"   python train_fish_segmentation.py --data {yaml_path} --epochs 100")
        
        return yaml_path
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return None

if __name__ == "__main__":
    main()
