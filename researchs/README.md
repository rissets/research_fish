# YOLO Object Detection untuk "bilal"

Project ini berisi implementasi lengkap untuk training dan deteksi objek menggunakan YOLO dengan label "bilal".

## Setup

1. **Install Dependencies**
```bash
python install_dependencies.py
```

2. **Persiapkan Dataset**
```bash
python prepare_dataset.py
```

3. **Training Model**
```bash
python train_yolo_bilal.py
```

4. **Deteksi Objek**
```bash
python detect_objects.py
```

## Struktur File

```
datasets/
  images/          # Gambar training (1bilal.jpeg, 2bilal.jpeg, dst)
  labels/          # File label YOLO (dibuat otomatis)

data.yaml          # Konfigurasi dataset
prepare_dataset.py # Script untuk pre-processing dan labeling
train_yolo_bilal.py # Script training YOLO
detect_objects.py  # Script deteksi objek
install_dependencies.py # Install dependencies
```

## Cara Kerja

### 1. Pre-processing Dataset (prepare_dataset.py)
- Membaca semua gambar di `datasets/images/`
- Menggunakan OpenCV Haar Cascade untuk deteksi wajah/objek
- Mengkonversi bounding box ke format YOLO
- Menyimpan label di `datasets/labels/`

### 2. Training (train_yolo_bilal.py)
- Menggunakan YOLOv8 dari Ultralytics
- Training dengan dataset yang sudah disiapkan
- Menyimpan model terbaik di `runs/detect/bilal_detection/`

### 3. Deteksi (detect_objects.py)
- Menggunakan model yang sudah ditraining
- Fallback ke OpenCV jika model belum ready
- Support deteksi pada gambar tunggal, dataset, atau real-time webcam

## Parameter Training

- **Epochs**: 100 (bisa disesuaikan)
- **Batch size**: 16 (sesuaikan dengan memory)
- **Image size**: 640x640
- **Model**: YOLOv8n (ringan, cepat)

## Customization

### Mengubah Model YOLO
Edit `train_yolo_bilal.py`:
```python
model = YOLO('yolov8s.pt')  # Ganti dengan yolov8m.pt, yolov8l.pt, dst
```

### Mengubah Metode Deteksi Pre-processing
Edit function `detect_face_opencv()` di `prepare_dataset.py` untuk menggunakan metode deteksi yang berbeda.

### Menambah Class
1. Edit `data.yaml`:
```yaml
nc: 2
names: ['bilal', 'class_baru']
```

2. Update script labeling untuk menangani multiple class

## Troubleshooting

### Model tidak ditemukan
- Pastikan training sudah selesai
- Check path model di `detect_objects.py`

### Deteksi tidak akurat
- Tambah lebih banyak data training
- Increase epochs
- Gunakan model yang lebih besar (yolov8m, yolov8l)
- Manual labeling untuk data yang lebih akurat

### Memory error saat training
- Kurangi batch size
- Gunakan model yang lebih kecil
- Kurangi image size

## Dataset Folder Structure

Pastikan struktur folder seperti ini:
```
datasets/
  images/
    1bilal.jpeg
    2bilal.jpeg
    3bilal.jpeg
    4bilal.jpeg
    5bilal.jpeg
    6bilal.jpeg
  labels/          # Akan dibuat otomatis
    1bilal.txt
    2bilal.txt
    ...
```
