#!/usr/bin/env python3
"""
Script untuk training YOLO segmentation model dengan fish dataset
Author: AI Assistant
Date: 2025-09-13
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import json
from datetime import datetime

class FishSegmentationTrainer:
    def __init__(self, model_name="yolov8n-seg.pt", device="auto"):
        self.device = device
        self.model_name = model_name
        self.model = None
        self.results_dir = Path("./training_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Setup device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🔧 Using device: {self.device}")
        print(f"📱 CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🔥 GPU: {torch.cuda.get_device_name()}")
    
    def load_model(self, model_path=None):
        """Load YOLO model"""
        if model_path and os.path.exists(model_path):
            print(f"📥 Loading custom model: {model_path}")
            self.model = YOLO(model_path)
        else:
            print(f"📥 Loading pretrained model: {self.model_name}")
            self.model = YOLO(self.model_name)
        
        return self.model
    
    def validate_dataset(self, data_path):
        """Validate dataset structure"""
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {data_path}")
        
        # Load data.yaml
        if data_path.suffix == '.yaml':
            yaml_path = data_path
            dataset_root = data_path.parent
        else:
            yaml_path = data_path / "data.yaml"
            dataset_root = data_path
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        print(f"📊 Dataset validation:")
        print(f"   Path: {dataset_root}")
        print(f"   Classes: {data_config.get('nc', 'Unknown')}")
        print(f"   Names: {data_config.get('names', 'Unknown')}")
        
        # Check train/val paths
        train_path = dataset_root / data_config.get('train', 'train/images')
        val_path = dataset_root / data_config.get('val', 'valid/images')
        
        if train_path.exists():
            train_images = list(train_path.glob("*.jpg")) + list(train_path.glob("*.png"))
            print(f"   Train images: {len(train_images)}")
        else:
            print(f"   ⚠ Train path not found: {train_path}")
        
        if val_path.exists():
            val_images = list(val_path.glob("*.jpg")) + list(val_path.glob("*.png"))
            print(f"   Validation images: {len(val_images)}")
        else:
            print(f"   ⚠ Validation path not found: {val_path}")
        
        return str(yaml_path), data_config
    
    def train(self, data_path, epochs=100, imgsz=640, batch_size=16, 
              patience=30, save_period=10, **kwargs):
        """Train the segmentation model"""
        
        # Validate dataset
        yaml_path, data_config = self.validate_dataset(data_path)
        
        # Load model
        if not self.model:
            self.load_model()
        
        # Training parameters
        train_params = {
            'data': yaml_path,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'device': self.device,
            'patience': patience,
            'save_period': save_period,
            'project': str(self.results_dir),
            'name': f'fish_segmentation_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'SGD',
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'label_smoothing': 0.0,
            'nbs': 64,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
        }
        
        # Update with custom parameters
        train_params.update(kwargs)
        
        print(f"\n🚀 Starting training with parameters:")
        for key, value in train_params.items():
            print(f"   {key}: {value}")
        
        try:
            # Start training
            results = self.model.train(**train_params)
            
            # Save training info
            training_info = {
                'timestamp': datetime.now().isoformat(),
                'model': self.model_name,
                'dataset': str(data_path),
                'parameters': train_params,
                'results_path': str(results.save_dir) if hasattr(results, 'save_dir') else None
            }
            
            info_path = self.results_dir / f"training_info_{train_params['name']}.json"
            with open(info_path, 'w') as f:
                json.dump(training_info, f, indent=2)
            
            print(f"\n✅ Training completed!")
            print(f"📁 Results saved to: {results.save_dir if hasattr(results, 'save_dir') else 'Check runs folder'}")
            print(f"📄 Training info: {info_path}")
            
            return results
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise e
    
    def validate_model(self, model_path, data_path):
        """Validate trained model"""
        print(f"\n🔍 Validating model: {model_path}")
        
        model = YOLO(model_path)
        yaml_path, _ = self.validate_dataset(data_path)
        
        results = model.val(data=yaml_path, device=self.device)
        
        print(f"✅ Validation completed!")
        return results
    
    def test_inference(self, model_path, image_path, save_dir="./inference_results"):
        """Test inference on sample images"""
        print(f"\n🧪 Testing inference...")
        
        model = YOLO(model_path)
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        if os.path.isfile(image_path):
            images = [image_path]
        else:
            image_dir = Path(image_path)
            images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
            images = images[:5]  # Test on first 5 images
        
        for img_path in images:
            print(f"   Processing: {img_path}")
            results = model(str(img_path), device=self.device)
            
            # Save results
            for i, result in enumerate(results):
                output_path = save_dir / f"{Path(img_path).stem}_result.jpg"
                result.save(str(output_path))
                print(f"   Saved: {output_path}")
        
        print(f"✅ Inference testing completed! Results in: {save_dir}")

def main():
    parser = argparse.ArgumentParser(description='Fish Segmentation Training with YOLO')
    parser.add_argument('--data', type=str, required=True, 
                       help='Path to dataset yaml file or dataset directory')
    parser.add_argument('--model', type=str, default='yolov8n-seg.pt',
                       help='Model to use (default: yolov8n-seg.pt)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs (default: 100)')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size (default: 640)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use: auto, cpu, cuda, or cuda:0 (default: auto)')
    parser.add_argument('--patience', type=int, default=30,
                       help='Early stopping patience (default: 30)')
    parser.add_argument('--validate', action='store_true',
                       help='Run validation after training')
    parser.add_argument('--test-inference', type=str,
                       help='Test inference on image/directory after training')
    
    args = parser.parse_args()
    
    print("🐟 Fish Segmentation Training with YOLO")
    print("=" * 60)
    
    # Initialize trainer
    trainer = FishSegmentationTrainer(model_name=args.model, device=args.device)
    
    # Start training
    results = trainer.train(
        data_path=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        patience=args.patience
    )
    
    # Get best model path
    if hasattr(results, 'save_dir'):
        best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    else:
        # Try to find in runs directory
        runs_dir = Path("runs/segment")
        if runs_dir.exists():
            latest_run = max(runs_dir.iterdir(), key=lambda x: x.stat().st_mtime)
            best_model_path = latest_run / "weights" / "best.pt"
        else:
            best_model_path = None
    
    # Validation
    if args.validate and best_model_path and best_model_path.exists():
        trainer.validate_model(str(best_model_path), args.data)
    
    # Test inference
    if args.test_inference and best_model_path and best_model_path.exists():
        trainer.test_inference(str(best_model_path), args.test_inference)
    
    print(f"\n🎉 Training pipeline completed!")
    if best_model_path:
        print(f"🏆 Best model: {best_model_path}")

if __name__ == "__main__":
    main()
