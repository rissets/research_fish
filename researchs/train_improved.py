"""
Script training YOLO yang diperbaiki dengan parameter yang lebih sesuai untuk dataset 2 class: bilal dan andi
"""

from ultralytics import YOLO
import os

def train_yolo_improved():
    """
    Training YOLO dengan parameter yang disesuaikan untuk dataset kecil
    """
    # Load model - gunakan model yang lebih kecil untuk dataset kecil
    model = YOLO('yolov8n.pt')
    
    print("Memulai training dengan parameter yang diperbaiki...")
    
    # Training parameters yang disesuaikan untuk dataset kecil
    results = model.train(
        data='data.yaml',           # File konfigurasi dataset
        epochs=100,                 # Lebih banyak epoch untuk dataset kecil
        imgsz=416,                  # Ukuran gambar lebih kecil untuk training lebih cepat
        batch=4,                    # Batch size kecil untuk dataset kecil
        device='cuda',               # Gunakan 'cuda' jika ada GPU
        project='runs/detect',      # Folder output
        name='bilal_andi_detection',     # Nama experiment
        save=True,                  # Simpan checkpoint
        plots=True,                 # Buat plot hasil training
        verbose=True,               # Output detail
        patience=50,                # Early stopping patience
        save_period=20,             # Simpan checkpoint setiap 20 epoch
        workers=2,                  # Jumlah worker untuk data loading
        lr0=0.01,                   # Learning rate initial
        lrf=0.01,                   # Learning rate final
        momentum=0.937,             # Momentum
        weight_decay=0.0005,        # Weight decay
        warmup_epochs=3,            # Warmup epochs
        warmup_momentum=0.8,        # Warmup momentum
        box=7.5,                    # Box loss gain
        cls=0.5,                    # Class loss gain
        dfl=1.5,                    # DFL loss gain
        pose=12.0,                  # Pose loss gain (not used for detection)
        kobj=1.0,                   # Keypoint obj loss gain (not used for detection)
        label_smoothing=0.0,        # Label smoothing
        nbs=64,                     # Nominal batch size
        overlap_mask=True,          # Overlap mask
        mask_ratio=4,               # Mask downsample ratio
        dropout=0.0,                # Dropout
        val=True,                   # Validate during training
        split='val',                # Dataset split for validation
        amp=False,                  # Automatic Mixed Precision
        fraction=1.0,               # Dataset fraction
        profile=False,              # Profile ONNX and TensorRT speeds
        freeze=None,                # Freeze layers
        multi_scale=False,          # Multi-scale training
        # overlap=0.0,                # Segment overlap
        copy_paste=0.0,             # Copy-paste augmentation
        auto_augment='randaugment', # Auto augmentation
        erasing=0.4,                # Random erasing
        crop_fraction=1.0,          # Crop fraction
    )
    
    print("Training selesai!")
    print(f"Model terbaik tersimpan di: {results.save_dir}")
    
    # Validasi model terbaik
    best_model_path = f"{results.save_dir}/weights/best.pt"
    if os.path.exists(best_model_path):
        print("\nMelakukan validasi model terbaik...")
        model_best = YOLO(best_model_path)
        
        # Validasi dengan confidence rendah
        val_results = model_best.val(
            data='data.yaml',
            device='cuda',
            conf=0.001,  # Confidence threshold sangat rendah untuk test
            iou=0.6,     # IoU threshold
            max_det=300, # Maximum detections per image
            half=False,  # Half precision
            plots=True,  # Save plots
            verbose=True
        )
        
        print("Validasi selesai!")
        
        # Test prediksi pada satu gambar dataset
        print("\nTesting prediksi pada gambar dataset...")
        test_img = 'datasets/images/1mas_andi.jpeg'
        if os.path.exists(test_img):
            # Prediksi dengan confidence sangat rendah
            pred_results = model_best(test_img, conf=0.001, save=True, project='runs/detect', name='test_prediction')
            
            boxes = pred_results[0].boxes
            detection_count = len(boxes) if boxes is not None else 0
            print(f"Deteksi pada {test_img}: {detection_count} objek")
            
            if detection_count > 0:
                for i, box in enumerate(boxes):
                    conf_score = box.conf[0].item()
                    cls = box.cls[0].item()
                    print(f"  Deteksi {i+1}: confidence={conf_score:.4f}, class={int(cls)}")
        
    return results

def check_dataset():
    """
    Cek dataset dan label sebelum training
    """
    print("=== Checking Dataset ===")
    
    # Cek file konfigurasi
    if not os.path.exists('data.yaml'):
        print("❌ File data.yaml tidak ditemukan!")
        return False
    
    # Cek folder images
    images_dir = 'datasets/images'
    if not os.path.exists(images_dir):
        print(f"❌ Folder {images_dir} tidak ditemukan!")
        return False
    
    # Cek folder labels
    labels_dir = 'datasets/labels'
    if not os.path.exists(labels_dir):
        print(f"❌ Folder {labels_dir} tidak ditemukan!")
        return False
    
    # Hitung jumlah gambar dan label
    images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    labels = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    
    print(f"✅ Jumlah gambar: {len(images)}")
    print(f"✅ Jumlah label: {len(labels)}")
    
    # Cek konsistensi nama file
    image_names = set([os.path.splitext(f)[0] for f in images])
    label_names = set([os.path.splitext(f)[0] for f in labels])
    
    missing_labels = image_names - label_names
    missing_images = label_names - image_names
    
    if missing_labels:
        print(f"⚠️ Gambar tanpa label: {missing_labels}")
    if missing_images:
        print(f"⚠️ Label tanpa gambar: {missing_images}")
    
    # Cek isi beberapa file label
    print("\n📄 Sample label content:")
    for i, label_file in enumerate(labels[:3]):  # Cek 3 label pertama
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, 'r') as f:
            content = f.read().strip()
            print(f"  {label_file}: {content}")
    
    return len(images) > 0 and len(labels) > 0

if __name__ == "__main__":
    print("=== Improved YOLO Training untuk Object Detection 'andi' ===")

    # Cek dataset dulu
    if check_dataset():
        print("\n✅ Dataset valid, memulai training...")
        try:
            results = train_yolo_improved()
            print("\n🎉 Training berhasil!")
        except Exception as e:
            print(f"\n❌ Error saat training: {e}")
    else:
        print("\n❌ Dataset tidak valid. Jalankan prepare_dataset.py terlebih dahulu.")
