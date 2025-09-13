"""
Script training YOLO yang lebih cepat untuk dataset kecil
"""

from ultralytics import YOLO
import os

def train_quick():
    """Training cepat dengan parameter yang disesuaikan untuk dataset kecil"""
    print("=== Quick Training YOLO untuk Dataset Kecil ===")
    
    # Load model
    model = YOLO('yolov8n.pt')
    
    # Training dengan parameter yang disesuaikan untuk dataset kecil
    results = model.train(
        data='data.yaml',
        epochs=20,              # Lebih sedikit epoch untuk dataset kecil
        imgsz=320,              # Image size lebih kecil untuk training cepat
        batch=4,                # Batch size kecil
        device='cpu',           # CPU
        project='runs/detect',
        name='bilal_quick',
        save=True,
        plots=True,
        verbose=True,
        patience=5,             # Early stopping jika tidak ada improvement
        lr0=0.01,              # Learning rate
        warmup_epochs=1        # Warmup singkat
    )
    
    print("Quick training selesai!")
    print(f"Model tersimpan di: {results.save_dir}")
    
    return results

if __name__ == "__main__":
    if not os.path.exists('data.yaml'):
        print("Error: File data.yaml tidak ditemukan!")
        exit(1)
    
    if not os.path.exists('datasets/labels'):
        print("Error: Folder datasets/labels tidak ditemukan!")
        print("Jalankan prepare_dataset.py terlebih dahulu.")
        exit(1)
    
    try:
        results = train_quick()
        print("Training berhasil!")
    except Exception as e:
        print(f"Error: {e}")
