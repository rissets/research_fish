#!/usr/bin/env python3
"""
Training script for fish object detection using YOLOv8
Author: AI Assistant
Date: 2025-09-13
"""

from ultralytics import YOLO
import argparse
import os
from datetime import datetime

def train(opt):
    # Initialize model
    model = YOLO(opt.weights)
    
    # Training arguments
    training_args = {
        'data': opt.data,
        'epochs': opt.epochs,
        'imgsz': opt.img,
        'batch': opt.batch,
        'name': f'fish_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'device': opt.device,
        'project': 'training_results',
        'val': opt.validate,
        'verbose': True
    }
    
    # Start training
    results = model.train(**training_args)
    
    print("\n🎉 Training completed!")
    print(f"Model saved in: {results.save_dir}")
    
    if opt.validate:
        print("\n📊 Validation Results:")
        metrics = results.results_dict
        print(f"mAP50: {metrics.get('metrics/mAP50(B)', 'N/A'):.3f}")
        print(f"mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.3f}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='initial weights path')
    parser.add_argument('--data', type=str, required=True, help='dataset.yaml path')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--img', type=int, default=640, help='image size')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--task', type=str, default='detect', help='detect, segment, classify, pose')
    parser.add_argument('--validate', action='store_true', help='validate after training')
    
    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_args()
    train(opt)