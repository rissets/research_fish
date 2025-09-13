#!/usr/bin/env python3
"""
Alternative Fish Dataset Downloader - menggunakan sumber dataset terbuka
Author: AI Assistant
Date: 2025-09-13
"""

import os
import sys
import requests
import zipfile
import json
from pathlib import Path
import yaml
import cv2
import numpy as np
from urllib.parse import urlparse
import shutil

class SimpleFishDatasetDownloader:
    def __init__(self, base_dir="./datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Dataset URLs yang tersedia bebas
        self.datasets = {
            "fish_sample": {
                "name": "Fish Sample Dataset",
                "description": "Small sample fish dataset for testing",
                "url": None,  # Will be created locally
                "type": "sample"
            },
            "open_fish": {
                "name": "Open Fish Dataset", 
                "description": "Open source fish classification dataset",
                "url": "https://github.com/fishbase/fishbase-data/archive/refs/heads/main.zip",
                "type": "github"
            }
        }
    
    def create_sample_dataset(self):
        """Create a small sample dataset for object detection testing"""
        print("🔧 Creating sample fish object detection dataset...")
        
        sample_dir = self.base_dir / "fish_sample_detection"
        sample_dir.mkdir(exist_ok=True)
        
        # Create directory structure
        for split in ['train', 'val', 'test']:
            (sample_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (sample_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Create sample images (colored rectangles representing fish)
        colors = [
            (255, 0, 0),    # Red fish - Tuna
            (0, 255, 0),    # Green fish - Salmon
            (0, 0, 255),    # Blue fish - Cod
            (255, 255, 0),  # Yellow fish - Bass
            (255, 0, 255),  # Magenta fish - Trout
        ]
        
        fish_types = ['tuna', 'salmon', 'cod', 'bass', 'trout']
        fish_names_id = ['tuna', 'salmon', 'kod', 'bass', 'trout']
        
        # Generate sample images and labels
        for split_idx, split in enumerate(['train', 'val', 'test']):
            n_images = [20, 8, 5][split_idx]  # 20 train, 8 val, 5 test
            
            for i in range(n_images):
                # Create synthetic fish image
                img = np.ones((640, 640, 3), dtype=np.uint8) * 50  # Dark background
                
                # Add 1-3 fish per image
                n_fish = np.random.randint(1, 4)
                label_lines = []
                
                for fish_idx in range(n_fish):
                    # Random fish properties
                    fish_type = np.random.randint(0, len(fish_types))
                    color = colors[fish_type]
                    
                    # Random fish size and position
                    width = np.random.randint(50, 150)
                    height = np.random.randint(30, 100)
                    x = np.random.randint(width//2, 640 - width//2)
                    y = np.random.randint(height//2, 640 - height//2)
                    
                    # Draw fish (ellipse)
                    cv2.ellipse(img, (x, y), (width//2, height//2), 0, 0, 360, color, -1)
                    
                    # Add some texture
                    overlay = img.copy()
                    cv2.ellipse(overlay, (x, y), (width//3, height//3), 0, 0, 360, 
                              (min(255, color[0]+50), min(255, color[1]+50), min(255, color[2]+50)), -1)
                    img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
                    
                    # Create YOLO format label for object detection (normalized coordinates)
                    center_x = x / 640
                    center_y = y / 640
                    norm_width = width / 640
                    norm_height = height / 640
                    
                    # YOLO format: class_id center_x center_y width height
                    label_lines.append(f"{fish_type} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}")
                
                # Save image
                img_path = sample_dir / split / 'images' / f'fish_{split}_{i:03d}.jpg'
                cv2.imwrite(str(img_path), img)
                
                # Save label
                label_path = sample_dir / split / 'labels' / f'fish_{split}_{i:03d}.txt'
                with open(label_path, 'w') as f:
                    f.write('\n'.join(label_lines))
        
        # Create data.yaml for object detection
        data_yaml = {
            'path': str(sample_dir),
            'train': 'train/images',
            'val': 'val/images', 
            'test': 'test/images',
            'nc': len(fish_types),
            'names': fish_names_id  # Use Indonesian names
        }
        
        with open(sample_dir / 'data.yaml', 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        print(f"✅ Sample object detection dataset created at: {sample_dir}")
        print(f"   Train images: 20")
        print(f"   Val images: 8") 
        print(f"   Test images: 5")
        print(f"   Classes: {len(fish_types)} ({', '.join(fish_names_id)})")
        
        return str(sample_dir / 'data.yaml')
    
    def download_from_url(self, url, dataset_name):
        """Download dataset from URL"""
        dataset_dir = self.base_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        print(f"📥 Downloading {dataset_name} from {url}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get filename from URL
            parsed_url = urlparse(url)
            filename = Path(parsed_url.path).name
            if not filename:
                filename = f"{dataset_name}.zip"
            
            zip_path = dataset_dir / filename
            
            # Download with progress
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r   Progress: {progress:.1f}%", end='', flush=True)
            
            print(f"\n✅ Downloaded: {zip_path}")
            
            # Extract if it's a zip file
            if zip_path.suffix.lower() == '.zip':
                print("📦 Extracting...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(dataset_dir)
                
                # Remove zip file
                zip_path.unlink()
                print(f"✅ Extracted to: {dataset_dir}")
            
            return str(dataset_dir)
            
        except Exception as e:
            print(f"❌ Failed to download {dataset_name}: {e}")
            return None
    
    def convert_to_yolo_detection(self, dataset_path):
        """Convert downloaded dataset to YOLO object detection format"""
        dataset_path = Path(dataset_path)
        
        # Look for images in the dataset
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(list(dataset_path.rglob(f"*{ext}")))
        
        if not image_files:
            print(f"❌ No images found in {dataset_path}")
            return None
        
        print(f"📸 Found {len(image_files)} images")
        
        # Create YOLO structure
        yolo_dir = dataset_path / "yolo_format"
        yolo_dir.mkdir(exist_ok=True)
        
        for split in ['train', 'val', 'test']:
            (yolo_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (yolo_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Split images
        n_total = len(image_files)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.2)
        
        np.random.shuffle(image_files)
        
        splits = {
            'train': image_files[:n_train],
            'val': image_files[n_train:n_train+n_val],
            'test': image_files[n_train+n_val:]
        }
        
        for split, images in splits.items():
            print(f"Processing {split}: {len(images)} images")
            
            for img_path in images:
                # Copy image
                new_img_path = yolo_dir / split / 'images' / img_path.name
                shutil.copy2(img_path, new_img_path)
                
                # Create dummy label for object detection
                # In real scenario, you'd convert existing annotations
                label_path = yolo_dir / split / 'labels' / (img_path.stem + '.txt')
                
                # Create a dummy bounding box covering most of the image
                # This is just for demonstration - real labels would come from annotations
                with open(label_path, 'w') as f:
                    f.write("0 0.5 0.5 0.8 0.6")  # class_id center_x center_y width height
        
        # Create data.yaml for object detection
        data_yaml = {
            'path': str(yolo_dir),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images', 
            'nc': 1,
            'names': ['ikan']  # Indonesian name
        }
        
        with open(yolo_dir / 'data.yaml', 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        print(f"✅ YOLO object detection format dataset created: {yolo_dir}")
        return str(yolo_dir / 'data.yaml')
    
    def download_all_datasets(self):
        """Download all available datasets"""
        downloaded = []
        
        # Create sample dataset first
        sample_config = self.create_sample_dataset()
        if sample_config:
            downloaded.append(sample_config)
        
        # Download from URLs
        for dataset_key, dataset_info in self.datasets.items():
            if dataset_info['type'] == 'sample':
                continue  # Already created
            
            if dataset_info['url']:
                dataset_path = self.download_from_url(dataset_info['url'], dataset_key)
                if dataset_path:
                    # Convert to YOLO detection format
                    yolo_config = self.convert_to_yolo_detection(dataset_path)
                    if yolo_config:
                        downloaded.append(yolo_config)
        
        return downloaded
    
    def list_datasets(self):
        """List available datasets"""
        print("\n📊 Available Datasets:")
        print("-" * 50)
        
        configs = []
        for dataset_dir in self.base_dir.iterdir():
            if dataset_dir.is_dir():
                # Look for data.yaml files
                yaml_files = list(dataset_dir.rglob("data.yaml"))
                for yaml_file in yaml_files:
                    configs.append(str(yaml_file))
                    
                    with open(yaml_file, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    print(f"✅ {dataset_dir.name}")
                    print(f"   Config: {yaml_file}")
                    print(f"   Classes: {config.get('nc', 'Unknown')}")
                    print(f"   Names: {config.get('names', 'Unknown')}")
                    print()
        
        return configs

def main():
    print("🐟 Simple Fish Dataset Downloader")
    print("=" * 50)
    
    downloader = SimpleFishDatasetDownloader()
    
    print("Available options:")
    print("1. Create sample dataset (for testing)")
    print("2. Download all datasets")
    print("3. List existing datasets")
    print("4. Download specific dataset")
    
    try:
        choice = int(input("\nSelect option (1-4): "))
    except ValueError:
        choice = 1
    
    if choice == 1:
        config = downloader.create_sample_dataset()
        print(f"\n🎉 Sample dataset ready!")
        print(f"Use: python train_fish_segmentation.py --data {config}")
    
    elif choice == 2:
        configs = downloader.download_all_datasets()
        print(f"\n🎉 Downloaded {len(configs)} datasets!")
        for config in configs:
            print(f"  - {config}")
    
    elif choice == 3:
        configs = downloader.list_datasets()
        if configs:
            print(f"Ready for training: {len(configs)} datasets")
        else:
            print("No datasets found. Run download first.")
    
    elif choice == 4:
        print("\nAvailable datasets:")
        for i, (key, info) in enumerate(downloader.datasets.items(), 1):
            print(f"  {i}. {info['name']} - {info['description']}")
        
        try:
            dataset_choice = int(input(f"\nSelect dataset (1-{len(downloader.datasets)}): ")) - 1
            dataset_key = list(downloader.datasets.keys())[dataset_choice]
            dataset_info = downloader.datasets[dataset_key]
            
            if dataset_info['type'] == 'sample':
                config = downloader.create_sample_dataset()
            else:
                dataset_path = downloader.download_from_url(dataset_info['url'], dataset_key)
                if dataset_path:
                    config = downloader.convert_to_yolo_detection(dataset_path)
            
            if config:
                print(f"\n🎉 Dataset ready: {config}")
        except (ValueError, IndexError):
            print("❌ Invalid selection")

if __name__ == "__main__":
    main()
