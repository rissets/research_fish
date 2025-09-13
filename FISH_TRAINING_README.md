# Fish Dataset Training dengan YOLO Segmentation

Proyek ini menyediakan tools lengkap untuk mendownload dataset ikan dan melakukan training model segmentasi menggunakan YOLO.

## 🚀 Quick Start

### 1. Persiapan Lingkungan

```bash
# Install dependencies
pip install -r requirements.txt

# Atau gunakan quick start script
python quick_start.py
```

### 2. Download Dataset

#### Opsi A: Dataset Sample (Untuk Testing)
```bash
python simple_fish_dataset.py
# Pilih opsi 1 untuk membuat sample dataset
```

#### Opsi B: Dataset dari Kaggle
```bash
# Setup Kaggle API terlebih dahulu
# 1. Download kaggle.json dari https://www.kaggle.com/settings
# 2. Simpan di ~/.kaggle/kaggle.json
# 3. chmod 600 ~/.kaggle/kaggle.json

python download_fish_dataset.py
```

### 3. Training Model

#### Quick Training
```bash
python train_fish_segmentation.py --data ./datasets/fish_sample/data.yaml --epochs 50 --batch 8
```

#### Full Training
```bash
python train_fish_segmentation.py \
    --data ./datasets/fish_sample/data.yaml \
    --epochs 100 \
    --batch 16 \
    --validate \
    --test-inference ./datasets/fish_sample/test/images
```

## 📁 Struktur File

```
research_od/
├── download_fish_dataset.py      # Download dataset dari Kaggle
├── simple_fish_dataset.py        # Download dataset sederhana
├── train_fish_segmentation.py    # Script training utama
├── dataset_utils.py              # Utilities untuk dataset
├── quick_start.py                # Interactive quick start
├── requirements.txt              # Dependencies
└── datasets/                     # Folder dataset
    ├── fish_sample/              # Sample dataset
    ├── fish_large_scale/         # Dataset dari Kaggle
    └── ...
```

## 🛠 Scripts yang Tersedia

### 1. `quick_start.py`
Script interaktif untuk memudahkan setup dan training.

```bash
python quick_start.py
```

### 2. `simple_fish_dataset.py`
Download dan setup dataset sederhana untuk testing.

```bash
python simple_fish_dataset.py
```

**Features:**
- Membuat sample dataset synthetic
- Download dari sumber terbuka
- Konversi ke format YOLO

### 3. `download_fish_dataset.py`
Download dataset ikan dari Kaggle dan Roboflow.

```bash
python download_fish_dataset.py
```

**Requirements:**
- Kaggle API key (`~/.kaggle/kaggle.json`)
- Internet connection

### 4. `train_fish_segmentation.py`
Script training utama untuk model segmentasi.

```bash
python train_fish_segmentation.py --data <path_to_data.yaml> [options]
```

**Parameters:**
- `--data`: Path ke file data.yaml
- `--model`: Model YOLO (default: yolov8n-seg.pt)
- `--epochs`: Jumlah epoch (default: 100)
- `--batch`: Batch size (default: 16)
- `--imgsz`: Image size (default: 640)
- `--device`: Device (auto/cpu/cuda)
- `--validate`: Run validation setelah training
- `--test-inference`: Test inference pada images

### 5. `dataset_utils.py`
Utilities untuk analisis dan manipulasi dataset.

```bash
# Analyze dataset
python dataset_utils.py --analyze ./datasets/fish_sample

# Create preview
python dataset_utils.py --preview ./datasets/fish_sample --output ./previews

# Create custom training script
python dataset_utils.py --create-script ./datasets/fish_sample
```

## 🐟 Dataset yang Didukung

### 1. Sample Dataset (Built-in)
- **Deskripsi**: Dataset synthetic untuk testing
- **Size**: 33 images (20 train, 8 val, 5 test)
- **Format**: YOLO segmentation
- **Cara dapat**: `python simple_fish_dataset.py`

### 2. Kaggle Datasets
- **Large Scale Fish Dataset**: Dataset ikan skala besar
- **Fish Object Detection 2020**: Dataset detection ikan
- **Fish Detection Dataset**: Dataset detection ikan lainnya
- **Cara dapat**: `python download_fish_dataset.py`

### 3. Roboflow Datasets
- **Fish Market Dataset**: Dataset pasar ikan
- **Aquarium Dataset**: Dataset akuarium
- **Cara dapat**: `python download_fish_dataset.py`

## ⚙️ Konfigurasi Training

### Quick Training (Testing)
```bash
python train_fish_segmentation.py \
    --data ./datasets/fish_sample/data.yaml \
    --epochs 50 \
    --batch 4 \
    --imgsz 640
```

### Standard Training
```bash
python train_fish_segmentation.py \
    --data ./datasets/your_dataset/data.yaml \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --patience 30
```

### Extended Training
```bash
python train_fish_segmentation.py \
    --data ./datasets/your_dataset/data.yaml \
    --epochs 200 \
    --batch 24 \
    --imgsz 640 \
    --patience 50
```

## 📊 Monitoring Training

Training results akan disimpan di:
- `./training_results/` - Info training
- `./runs/segment/` - Model weights dan logs

### Melihat Hasil Training
```python
from ultralytics import YOLO

# Load best model
model = YOLO('./runs/segment/train/weights/best.pt')

# Test pada image
results = model('path/to/test/image.jpg')
results[0].show()
```

## 🔧 Troubleshooting

### Error: "Import kaggle could not be resolved"
```bash
pip install kaggle
```

### Error: "Kaggle API credentials not found"
1. Download `kaggle.json` dari https://www.kaggle.com/settings
2. Simpan di `~/.kaggle/kaggle.json`
3. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Error: "CUDA out of memory"
- Turunkan batch size: `--batch 8` atau `--batch 4`
- Turunkan image size: `--imgsz 416`

### Error: "No datasets found"
```bash
# Create sample dataset first
python simple_fish_dataset.py
# Pilih opsi 1
```

## 📈 Tips Optimasi

### 1. Batch Size
- **GPU 4GB**: batch=4-8
- **GPU 8GB**: batch=16-32  
- **GPU 12GB+**: batch=32+

### 2. Image Size
- **Quick training**: 416x416
- **Standard**: 640x640
- **High quality**: 1024x1024

### 3. Epochs
- **Testing**: 50-100 epochs
- **Production**: 200-500 epochs
- **Fine-tuning**: 100-200 epochs

### 4. Model Size
- **yolov8n-seg.pt**: Paling cepat, akurasi standard
- **yolov8s-seg.pt**: Balance speed/accuracy
- **yolov8m-seg.pt**: Akurasi tinggi
- **yolov8l-seg.pt**: Akurasi tertinggi

## 🎯 Contoh Workflow Lengkap

```bash
# 1. Setup
python quick_start.py
# Pilih: 1 (Install dependencies)

# 2. Setup Kaggle (opsional)
python quick_start.py  
# Pilih: 2 (Setup Kaggle)

# 3. Download dataset
python simple_fish_dataset.py
# Pilih: 1 (Create sample)

# 4. Analyze dataset
python dataset_utils.py --analyze ./datasets/fish_sample

# 5. Training
python train_fish_segmentation.py \
    --data ./datasets/fish_sample/data.yaml \
    --epochs 100 \
    --batch 16 \
    --validate

# 6. Test model
python -c "
from ultralytics import YOLO
model = YOLO('./runs/segment/train/weights/best.pt')
results = model('./datasets/fish_sample/test/images/')
"
```

## 📞 Support

Jika mengalami masalah:
1. Cek log error di terminal
2. Pastikan dependencies terinstall
3. Cek format dataset (data.yaml)
4. Coba dengan sample dataset terlebih dahulu

## 🚀 Next Steps

Setelah training berhasil:
1. **Export model**: Konversi ke ONNX, TensorRT
2. **Deploy**: Integrasikan ke aplikasi
3. **Fine-tune**: Training dengan dataset yang lebih spesifik
4. **Optimize**: Pruning dan quantization
