#!/usr/bin/env python3
"""
Demo script untuk testing fish segmentation training
Author: AI Assistant  
Date: 2025-09-13
"""

import os
import sys
import subprocess
from pathlib import Path

def test_training_pipeline():
    """Test complete training pipeline"""
    print("🧪 Testing Fish Training Pipeline")
    print("=" * 50)
    
    # 1. Create sample dataset
    print("\n1. Creating sample dataset...")
    try:
        result = subprocess.run([
            sys.executable, "simple_fish_dataset.py"
        ], input="1\n", text=True, capture_output=True)
        
        if result.returncode == 0:
            print("✅ Sample dataset created")
        else:
            print(f"❌ Failed to create sample dataset: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error creating sample dataset: {e}")
        return False
    
    # 2. Verify dataset
    dataset_path = Path("./datasets/fish_sample/data.yaml")
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return False
    
    print(f"✅ Dataset verified: {dataset_path}")
    
    # 3. Quick training test
    print("\n2. Starting quick training test...")
    try:
        cmd = [
            sys.executable, "train_fish_detection.py",  # Changed to detection
            "--data", str(dataset_path),
            "--epochs", "5",  # Very short for testing
            "--batch", "2",   # Small batch
            "--imgsz", "320", # Smaller image size
            "--patience", "10"
        ]
        
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
        
        if result.returncode == 0:
            print("✅ Training test completed successfully")
            
            # Check for output files
            runs_dir = Path("runs/detect")  # Changed from segment to detect
            if runs_dir.exists():
                latest_run = max(runs_dir.iterdir(), key=lambda x: x.stat().st_mtime)
                weights_dir = latest_run / "weights"
                
                if (weights_dir / "best.pt").exists():
                    print(f"✅ Model weights created: {weights_dir / 'best.pt'}")
                    return True
                else:
                    print("⚠ Training completed but no weights found")
            else:
                print("⚠ Training completed but no runs directory")
                
        else:
            print(f"❌ Training failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Training test timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Training test error: {e}")
        return False
    
    return True

def test_inference():
    """Test inference with trained model"""
    print("\n3. Testing inference...")
    
    # Find latest model
    runs_dir = Path("runs/detect")  # Changed from segment to detect
    if not runs_dir.exists():
        print("❌ No runs directory found")
        return False
    
    try:
        latest_run = max(runs_dir.iterdir(), key=lambda x: x.stat().st_mtime)
        model_path = latest_run / "weights" / "best.pt"
        
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            return False
        
        # Test inference
        test_images = Path("./datasets/fish_sample_detection/test/images")  # Updated path
        if not test_images.exists():
            print(f"❌ Test images not found: {test_images}")
            return False
        
        # Simple inference test
        inference_script = f'''
from ultralytics import YOLO
import sys

try:
    model = YOLO("{model_path}")
    results = model("{test_images}")
    print("✅ Inference successful")
    
    # Save one result for verification
    if results:
        results[0].save("./test_inference_result.jpg")
        print("✅ Test result saved: ./test_inference_result.jpg")
    
except Exception as e:
    print(f"❌ Inference failed: {{e}}")
    sys.exit(1)
'''
        
        result = subprocess.run([sys.executable, "-c", inference_script], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print(f"❌ Inference test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Inference test error: {e}")
        return False

def cleanup_test_files():
    """Clean up test files"""
    print("\n4. Cleaning up test files...")
    
    # Cleanup
    cleanup_paths = [
        "datasets/fish_sample_detection",  # Updated path
        "runs",
        "training_results",
        "test_inference_result.jpg"
    ]
    
    import shutil
    
    for path in cleanup_paths:
        path_obj = Path(path)
        try:
            if path_obj.exists():
                if path_obj.is_dir():
                    shutil.rmtree(path_obj)
                else:
                    path_obj.unlink()
                print(f"✅ Removed: {path}")
        except Exception as e:
            print(f"⚠ Could not remove {path}: {e}")

def main():
    print("🐟 Fish Training Pipeline Demo")
    print("This will test the complete training pipeline with a small dataset")
    print("=" * 60)
    
    # Check dependencies
    print("Checking dependencies...")
    try:
        import ultralytics
        import cv2
        import numpy as np
        import yaml
        import matplotlib
        print("✅ All dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        return False
    
    # Ask user confirmation
    response = input("\nThis will create test files and run training. Continue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    success = True
    
    # Run tests
    try:
        # Test pipeline
        if not test_training_pipeline():
            success = False
        
        # Test inference
        if success and not test_inference():
            success = False
        
        if success:
            print("\n🎉 All tests passed!")
            print("Training pipeline is working correctly.")
            print("\nNext steps:")
            print("1. Download real fish dataset")
            print("2. Run full training with more epochs")
            print("3. Evaluate model performance")
        else:
            print("\n❌ Some tests failed.")
            print("Check the error messages above.")
        
    except KeyboardInterrupt:
        print("\n⏹ Tests interrupted by user")
        success = False
    
    # Cleanup
    cleanup_choice = input("\nClean up test files? (y/n): ")
    if cleanup_choice.lower() == 'y':
        cleanup_test_files()
    
    return success

if __name__ == "__main__":
    main()
