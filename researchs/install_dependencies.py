"""
Script untuk instalasi dependencies yang diperlukan
"""

import subprocess
import sys

def install_package(package):
    """Install package menggunakan pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ {package} berhasil diinstall")
    except subprocess.CalledProcessError:
        print(f"✗ Gagal menginstall {package}")

def main():
    print("=== Instalasi Dependencies untuk YOLO Object Detection ===")
    
    packages = [
        "ultralytics",      # YOLO v8/v11
        "opencv-python",    # OpenCV
        "numpy",           # NumPy
        "matplotlib",      # Plotting
        "Pillow",          # Image processing
        "torch",           # PyTorch (untuk YOLO)
        "torchvision"      # PyTorch vision
    ]
    
    print("Menginstall packages yang diperlukan...")
    for package in packages:
        print(f"Installing {package}...")
        install_package(package)
    
    print("\nInstalasi selesai!")
    print("Anda dapat menjalankan:")
    print("1. python prepare_dataset.py - untuk mempersiapkan dataset")
    print("2. python train_yolo_bilal.py - untuk training model")
    print("3. python detect_objects.py - untuk deteksi objek")

if __name__ == "__main__":
    main()
