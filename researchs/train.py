# ==============================================================================
# Langkah 1: Instalasi dan Impor Pustaka yang Diperlukan
# ==============================================================================
# Instal pustaka 'ultralytics' yang berisi implementasi YOLOv8 dan YOLOv9.
# Opsi '-q' digunakan untuk instalasi senyap (quiet).
# !pip install -q ultralytics

# Impor kelas YOLO dari pustaka ultralytics.
import os

import yaml
from IPython.display import Image, display
from roboflow import Roboflow
from ultralytics import YOLO

print("✅ Instalasi dan impor selesai.")

# ==============================================================================
# Langkah 2: Mengunduh dan Menyiapkan Dataset
# ==============================================================================
# Dataset ini adalah "Fish Species Classification" dari Roboflow Universe.
# Dataset ini kecil dan cocok untuk demonstrasi.
# Link: https://universe.roboflow.com/research-vpani/fish-species-classification
# Kita akan mengunduh versi yang sudah diproses dalam format YOLO.
# !pip install - q roboflow

# Ganti dengan API Key Roboflow Anda
rf = Roboflow(api_key="pZkh6tKNxrvmMCh2Xdgu")
project = rf.workspace("research-vpani").project("fish-species-classification")
dataset = project.version(1).download("yolov8")

# Lokasi dataset setelah diunduh
dataset_location = dataset.location
print(f"📍 Dataset diunduh ke: {dataset_location}")

# ==============================================================================
# Langkah 3: Membuat File Konfigurasi Dataset (data.yaml)
# ==============================================================================
# File YAML ini memberi tahu skrip pelatihan di mana menemukan data
# dan apa saja kelas-kelas yang ada di dalamnya.

# Definisikan struktur data untuk file YAML
# 'path' akan diisi secara dinamis dengan lokasi dataset yang baru diunduh
# 'train' dan 'val' adalah path relatif ke gambar pelatihan dan validasi
# 'nc' adalah jumlah kelas (3 untuk dataset ini)
# 'names' adalah daftar nama kelas sesuai dengan class_id
config_data = {
    "path": dataset_location,
    "train": "train/images",
    "val": "valid/images",
    "nc": 3,
    "names": ["grouper", "northern-red-snapper", "salmon"],
}

# Tulis kamus Python ke file YAML
yaml_file_path = os.path.join(dataset_location, "data.yaml")
with open(yaml_file_path, "w") as f:
    yaml.dump(config_data, f, sort_keys=False)

print(f"📄 File konfigurasi 'data.yaml' berhasil dibuat di: {yaml_file_path}")
# Tampilkan isi file untuk verifikasi
with open(yaml_file_path, "r") as f:
    print("\nIsi data.yaml:")
    print(f.read())

# ==============================================================================
# Langkah 4: Memuat Model Pre-trained dan Memulai Fine-Tuning
# ==============================================================================
# Kita akan menggunakan YOLOv9-c.pt, model yang kuat namun tetap efisien.
# Bobot pre-trained dari dataset COCO akan diunduh secara otomatis.
# 'c' berarti 'compact', ini adalah salah satu varian ukuran dari YOLOv9.
model = YOLO("yolov8s.pt")
#
# print("\n🚀 Memulai proses fine-tuning model YOLOvv8..")

# Jalankan pelatihan
# data: path ke file data.yaml kita.
# epochs: jumlah iterasi pelatihan. 25 epochs cukup untuk demonstrasi.
#         Untuk hasil yang lebih baik, disarankan 100-300 epochs.[2]
# imgsz: ukuran gambar input. 640 adalah ukuran standar yang baik.
# batch: jumlah gambar yang diproses per iterasi. Sesuaikan jika VRAM GPU terbatas.
# results = model.train(
#     data=yaml_file_path,
#     epochs=25,
#     imgsz=640,
#     batch=8,
#     name="yolov9c_fish_finetuned",  # Nama folder untuk menyimpan hasil
# )
#
# print("🎉 Proses fine-tuning selesai!")

# ==============================================================================
# Langkah 5: Menjalankan Inferensi dengan Model yang Telah Dilatih
# ==============================================================================
# Hasil pelatihan, termasuk bobot terbaik ('best.pt'), disimpan di folder 'runs/detect/'.
# Kita akan menggunakan bobot ini untuk melakukan deteksi pada gambar baru.

# Path ke bobot model terbaik yang telah kita latih
best_model_path = "runs/detect/yolov9c_fish_finetuned/weights/best.pt"

# Muat model yang telah di-fine-tune
finetuned_model = YOLO(best_model_path)

# Path ke salah satu gambar validasi untuk pengujian inferensi
# Anda bisa menggantinya dengan path gambar Anda sendiri
validation_image_path = os.path.join(
    # dataset_location,
    # "valid/images/Gambar-ikan-salmon_500x_jpg.rf.6f06254146c89a50191e853e42763d32.jpg",
    "catla.jpg"
)

print(f"\n🔎 Menjalankan inferensi pada gambar: {validation_image_path}")

# Lakukan prediksi
results = finetuned_model.predict(source=validation_image_path, save=True)

print(
    "\n✅ Inferensi selesai. Gambar hasil prediksi disimpan di folder 'runs/detect/predict/'."
)

# Tampilkan gambar hasil prediksi (jika berjalan di lingkungan notebook)
prediction_image_path = f"runs/detect/predict/{os.path.basename(validation_image_path)}"
if os.path.exists(prediction_image_path):
    display(Image(filename=prediction_image_path))
else:
    print(
        f"Gambar hasil prediksi tidak ditemukan di path yang diharapkan: {
            prediction_image_path
        }"
    )
