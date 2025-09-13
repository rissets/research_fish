#!/usr/bin/env python3
"""
Quick Start Script untuk Fish Dataset Training
Author: AI Assistant
Date: 2025-09-13
"""

import os
import sys
import subprocess
from pathlib import Path

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_kaggle():
    """Setup Kaggle API"""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_json.exists():
        print(f"\n⚠️  Kaggle API setup required!")
        print(f"1. Go to https://www.kaggle.com/settings")
        print(f"2. Scroll down to 'API' section")
        print(f"3. Click 'Create New API Token'")
        print(f"4. Download kaggle.json")
        print(f"5. Place it at: {kaggle_json}")
        print(f"6. Run: chmod 600 {kaggle_json}")
        
        response = input("\nHave you setup kaggle.json? (y/n): ")
        if response.lower() != 'y':
            print("Please setup Kaggle API first, then run this script again.")
            return False
    
    # Set permissions
    try:
        os.chmod(str(kaggle_json), 0o600)
        print("✅ Kaggle API configured")
        return True
    except Exception as e:
        print(f"❌ Failed to set kaggle.json permissions: {e}")
        return False

def download_datasets():
    """Download fish datasets"""
    print("\n🐟 Downloading fish datasets...")
    try:
        subprocess.check_call([sys.executable, "download_fish_dataset.py"])
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download datasets: {e}")
        return False

def list_available_datasets():
    """List available datasets for training"""
    datasets_dir = Path("../datasets")
    if not datasets_dir.exists():
        print("❌ No datasets directory found")
        return []
    
    available_datasets = []
    for dataset_dir in datasets_dir.iterdir():
        if dataset_dir.is_dir():
            data_yaml = dataset_dir / "data.yaml"
            if data_yaml.exists():
                available_datasets.append({
                    'name': dataset_dir.name,
                    'path': str(data_yaml)
                })
    
    return available_datasets

def start_training():
    """Start training process"""
    datasets = list_available_datasets()
    
    if not datasets:
        print("❌ No datasets available for training")
        print("Run download step first or check datasets directory")
        return False
    
    print(f"\n📊 Available datasets:")
    for i, dataset in enumerate(datasets, 1):
        print(f"  {i}. {dataset['name']}")
    
    try:
        choice = int(input(f"\nSelect dataset (1-{len(datasets)}): ")) - 1
        selected_dataset = datasets[choice]
    except (ValueError, IndexError):
        print("❌ Invalid selection")
        return False
    
    print(f"\n🚀 Starting training with {selected_dataset['name']}...")
    
    # Training parameters
    print("\nTraining options:")
    print("1. Quick training (50 epochs, small batch)")
    print("2. Standard training (100 epochs, medium batch)")  
    print("3. Extended training (200 epochs, large batch)")
    print("4. Custom parameters")
    
    try:
        train_choice = int(input("Select training mode (1-4): "))
    except ValueError:
        train_choice = 2  # Default to standard
    
    # Set parameters based on choice
    if train_choice == 1:
        epochs, batch = 20, 8
    elif train_choice == 2:
        epochs, batch = 100, 16
    elif train_choice == 3:
        epochs, batch = 200, 24
    else:
        try:
            epochs = int(input("Epochs (default 100): ") or "100")
            batch = int(input("Batch size (default 16): ") or "16")
        except ValueError:
            epochs, batch = 100, 16
    
    # Start training
    cmd = [
        sys.executable, "train_fish_detection.py",
        "--data", selected_dataset['path'],
        "--epochs", str(epochs),
        "--batch", str(batch),
        "--imgsz", "640",  # image size
        "--model", "yolov8n.pt",  # use YOLOv8 nano detection model for detection
        "--device", "auto",  # automatically select best device
        "--patience", "30",  # early stopping patience
        "--validate"  # run validation after training
    ]
    
    print(f"\n🎯 Training command: {' '.join(cmd)}")
    
    try:
        subprocess.check_call(cmd)
        print("🎉 Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        return False

def main():
    print("🐟 Fish Dataset Training - Quick Start")
    print("=" * 60)
    
    while True:
        print(f"\nWhat would you like to do?")
        print("1. Install dependencies")
        print("2. Setup Kaggle API")
        print("3. Download fish datasets")
        print("4. Analyze available datasets")
        print("5. Start training")
        print("6. Exit")
        
        try:
            choice = int(input("\nSelect option (1-6): "))
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
            continue
        
        if choice == 1:
            install_dependencies()
        
        elif choice == 2:
            setup_kaggle()
        
        elif choice == 3:
            if download_datasets():
                print("✅ Datasets downloaded successfully")
            else:
                print("❌ Dataset download failed")
        
        elif choice == 4:
            datasets = list_available_datasets()
            if datasets:
                print(f"\n📊 Available datasets:")
                for dataset in datasets:
                    print(f"  - {dataset['name']}: {dataset['path']}")
                
                # Create preview
                for dataset in datasets:
                    try:
                        subprocess.check_call([
                            sys.executable, "dataset_utils.py", 
                            "--analyze", str(Path(dataset['path']).parent)
                        ])
                    except subprocess.CalledProcessError:
                        print(f"❌ Failed to analyze {dataset['name']}")
            else:
                print("❌ No datasets found")
        
        elif choice == 5:
            start_training()
        
        elif choice == 6:
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please select 1-6.")

if __name__ == "__main__":
    main()
