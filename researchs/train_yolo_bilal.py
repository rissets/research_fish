"""
Script untuk training YOLO model dengan dataset "bilal"
"""

from ultralytics import YOLO
import os

def train_yolo_model():
    """
    Training YOLO model untuk object detection
    """
    # Load model - bisa gunakan yolov8n.pt, yolov8s.pt, dll
    model = YOLO('yolov8n.pt')  # Ganti dengan model yang ada
    
    # Training parameters
    results = model.train(
        data='data.yaml',           # File konfigurasi dataset
        epochs=20,                 # Jumlah epoch
        imgsz=640,                  # Ukuran gambar input
        batch=16,                   # Batch size (sesuaikan dengan memory)
        device='cuda',              # Gunakan 'cuda' jika ada GPU
        project='runs/detect',      # Folder output
        name='bilal_detection',     # Nama experiment
        save=True,                  # Simpan checkpoint
        plots=True,                 # Buat plot hasil training
        verbose=True                # Output detail
    )
    
    print("Training selesai!")
    print(f"Model terbaik tersimpan di: {results.save_dir}")
    
    return results

def validate_model(model_path):
    """
    Validasi model yang sudah ditraining
    """
    model = YOLO(model_path)
    
    # Validasi
    results = model.val(
        data='data.yaml',
        device='cuda'
    )
    
    print("Validasi selesai!")
    return results

if __name__ == "__main__":
    print("=== Training YOLO untuk Object Detection 'bilal' ===")
    
    # Cek apakah file data.yaml ada
    if not os.path.exists('data.yaml'):
        print("Error: File data.yaml tidak ditemukan!")
        print("Pastikan file data.yaml sudah dibuat dengan benar.")
        exit(1)
    
    # Cek apakah dataset ada
    if not os.path.exists('datasets/images'):
        print("Error: Folder datasets/images tidak ditemukan!")
        exit(1)
    
    if not os.path.exists('datasets/labels'):
        print("Error: Folder datasets/labels tidak ditemukan!")
        print("Jalankan prepare_dataset.py terlebih dahulu.")
        exit(1)
    
    # Mulai training
    print("Memulai training...")
    try:
        results = train_yolo_model()
        print("Training berhasil!")
        
        # Optional: Validasi model terbaik
        best_model_path = results.save_dir / 'weights' / 'best.pt'
        if os.path.exists(best_model_path):
            print("Melakukan validasi model terbaik...")
            validate_model(best_model_path)
            
    except Exception as e:
        print(f"Error saat training: {e}")
        print("Pastikan semua dependency sudah terinstall: pip install ultralytics")
