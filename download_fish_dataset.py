#!/usr/bin/env python3
"""
Script untuk mendownload fish dataset dari Kaggle dan mempersiapkan untuk training YOLO segmentation
Author: AI Assistant
Date: 2025-09-13
"""

import os
import sys
import zipfile
import json
import shutil
from pathlib import Path
import subprocess
import requests
from urllib.parse import urlparse

class FishDatasetDownloader:
    def __init__(self, base_dir="./datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    def setup_kaggle_api(self):
        """Setup Kaggle API credentials"""
        try:
            import kaggle
            print("✓ Kaggle API sudah terinstall")
            return True
        except ImportError:
            print("Installing Kaggle API...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
            import kaggle
            return True
        except Exception as e:
            print(f"Error setting up Kaggle API: {e}")
            print("Please make sure you have kaggle.json in ~/.kaggle/ directory")
            print("Download it from: https://www.kaggle.com/settings -> API -> Create New API Token")
            return False
    
    def download_fish_species_dataset(self):
        """Download fish species dataset from Kaggle"""
        try:
            import kaggle
            
            # Dataset fish species yang populer di Kaggle
            datasets = [
                {
                    "name": "markdaniellampa/fish-dataset",
                    "folder": "fish_dataset",
                    "description": "Fish Species Dataset"
                },
                # {
                #     "name": "crowww/a-large-scale-fish-dataset",
                #     "folder": "fish_large_scale",
                #     "description": "Large Scale Fish Dataset"
                # },
                # {
                #     "name": "sriramr/2020-fish-object-detection-dataset",
                #     "folder": "fish_detection_2020", 
                #     "description": "Fish Object Detection Dataset 2020"
                # },
                # {
                #     "name": "yasirabdaali/fish-detection",
                #     "folder": "fish_detection_yasir",
                #     "description": "Fish Detection Dataset"
                # }
            ]
            
            downloaded_datasets = []
            
            for dataset in datasets:
                try:
                    print(f"\n🎣 Downloading {dataset['description']}...")
                    dataset_path = self.base_dir / dataset['folder']
                    dataset_path.mkdir(exist_ok=True)
                    
                    # Download dataset
                    kaggle.api.dataset_download_files(
                        dataset['name'], 
                        path=str(dataset_path), 
                        unzip=True
                    )
                    
                    print(f"✓ Successfully downloaded {dataset['description']} to {dataset_path}")
                    downloaded_datasets.append(dataset)
                    
                except Exception as e:
                    print(f"❌ Failed to download {dataset['description']}: {e}")
                    continue
            
            return downloaded_datasets
            
        except Exception as e:
            print(f"Error downloading datasets: {e}")
            return []
    
    def download_roboflow_fish_dataset(self):
        """Download fish dataset from Roboflow (alternative)"""
        try:
            import roboflow
            
            # Initialize Roboflow
            rf = roboflow.Roboflow()
            
            # Fish datasets available on Roboflow (public)
            roboflow_datasets = [
                {
                    "workspace": "university-bswxt",
                    "project": "fish-market-ggjso",
                    "version": 2,
                    "folder": "fish_market_roboflow"
                },
                {
                    "workspace": "roboflow-jvuqo",
                    "project": "aquarium-combined",
                    "version": 2,
                    "folder": "aquarium_roboflow"
                }
            ]
            
            downloaded_datasets = []
            
            for dataset_info in roboflow_datasets:
                try:
                    print(f"\n🐠 Downloading {dataset_info['project']} from Roboflow...")
                    
                    project = rf.workspace(dataset_info['workspace']).project(dataset_info['project'])
                    dataset = project.version(dataset_info['version']).download("yolov8")
                    
                    # Move to our datasets directory
                    dataset_path = self.base_dir / dataset_info['folder']
                    if os.path.exists(dataset.location):
                        if dataset_path.exists():
                            shutil.rmtree(dataset_path)
                        shutil.move(dataset.location, dataset_path)
                        
                        print(f"✓ Successfully downloaded {dataset_info['project']} to {dataset_path}")
                        downloaded_datasets.append(dataset_info)
                        
                except Exception as e:
                    print(f"❌ Failed to download {dataset_info['project']}: {e}")
                    continue
            
            return downloaded_datasets
            
        except Exception as e:
            print(f"Error with Roboflow download: {e}")
            return []
    
    def prepare_yolo_format(self, dataset_path):
        """Convert dataset to YOLO format if needed"""
        dataset_path = Path(dataset_path)
        
        # Check if already in YOLO format
        yolo_files = list(dataset_path.glob("data.yaml"))
        if yolo_files:
            print(f"✓ Dataset {dataset_path.name} already in YOLO format")
            return str(dataset_path / "data.yaml")
        
        # Look for common dataset structures
        image_dirs = []
        label_dirs = []
        
        for subdir in dataset_path.iterdir():
            if subdir.is_dir():
                if any(keyword in subdir.name.lower() for keyword in ['image', 'img', 'picture']):
                    image_dirs.append(subdir)
                elif any(keyword in subdir.name.lower() for keyword in ['label', 'annotation', 'ann']):
                    label_dirs.append(subdir)
        
        if image_dirs:
            print(f"Found image directories: {[d.name for d in image_dirs]}")
            print(f"Found label directories: {[d.name for d in label_dirs]}")
            
            # Create basic data.yaml
            data_yaml_content = f"""
# Fish Dataset Configuration
path: {dataset_path}
train: train/images
val: valid/images
test: test/images

# Classes (akan diupdate berdasarkan dataset)
nc: 1  # number of classes
names: ['fish']  # class names
"""
            
            data_yaml_path = dataset_path / "data.yaml"
            with open(data_yaml_path, 'w') as f:
                f.write(data_yaml_content.strip())
            
            print(f"✓ Created data.yaml for {dataset_path.name}")
            return str(data_yaml_path)
        
        return None
    
    def list_downloaded_datasets(self):
        """List all downloaded datasets"""
        print("\n📊 Downloaded Datasets:")
        print("-" * 50)
        
        dataset_configs = []
        for dataset_dir in self.base_dir.iterdir():
            if dataset_dir.is_dir():
                data_yaml = dataset_dir / "data.yaml"
                if data_yaml.exists():
                    print(f"✓ {dataset_dir.name} - YOLO Ready")
                    dataset_configs.append(str(data_yaml))
                else:
                    print(f"⚠ {dataset_dir.name} - Needs formatting")
                    config = self.prepare_yolo_format(dataset_dir)
                    if config:
                        dataset_configs.append(config)
        
        return dataset_configs

def main():
    print("🐟 Fish Dataset Downloader untuk YOLO Training")
    print("=" * 60)
    
    downloader = FishDatasetDownloader()
    
    # Setup Kaggle API
    if not downloader.setup_kaggle_api():
        print("⚠ Kaggle API setup failed. Will try Roboflow only.")
    
    # Download datasets
    print("\n1. Downloading from Kaggle...")
    kaggle_datasets = downloader.download_fish_species_dataset()
    
    print("\n2. Downloading from Roboflow...")
    roboflow_datasets = downloader.download_roboflow_fish_dataset()
    
    # Prepare YOLO format
    print("\n3. Preparing YOLO format...")
    dataset_configs = downloader.list_downloaded_datasets()
    
    print(f"\n🎉 Download complete! Found {len(dataset_configs)} ready datasets.")
    
    if dataset_configs:
        print("\nDataset configurations ready for training:")
        for i, config in enumerate(dataset_configs, 1):
            print(f"  {i}. {config}")
        
        print(f"\n📝 Next step: Run training script with one of these configurations")
        print(f"   Example: python train_fish_segmentation.py --data {dataset_configs[0]}")
    
    return dataset_configs

if __name__ == "__main__":
    main()
