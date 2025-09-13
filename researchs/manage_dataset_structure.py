"""
Script untuk menyalin gambar dari struktur folder class ke struktur flat yang diperlukan YOLO
dan membuat data.yaml yang sesuai
"""

import os
import shutil
from pathlib import Path

def create_flat_structure():
    """
    Menyalin gambar dari datasets/images/bilal/ dan datasets/images/andi/ 
    ke datasets/images/ (flat structure) untuk YOLO training
    """
    base_dir = 'datasets/images'
    temp_dir = 'datasets/images_temp'
    
    # Buat folder temporary
    os.makedirs(temp_dir, exist_ok=True)
    
    copied_count = 0
    
    # Copy dari folder class ke struktur flat
    for class_folder in ['bilal', 'andi']:
        class_path = os.path.join(base_dir, class_folder)
        
        if os.path.exists(class_path):
            print(f"Menyalin gambar dari folder: {class_folder}")
            
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src_path = os.path.join(class_path, img_file)
                    dst_path = os.path.join(temp_dir, img_file)
                    
                    shutil.copy2(src_path, dst_path)
                    copied_count += 1
                    print(f"  ✓ {img_file}")
    
    # Pindahkan folder asli ke backup
    backup_dir = 'datasets/images_backup'
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    shutil.move(base_dir, backup_dir)
    
    # Pindahkan temp folder ke images
    shutil.move(temp_dir, base_dir)
    
    print(f"\nSelesai! {copied_count} gambar telah disalin ke struktur flat.")
    print(f"Folder asli di-backup ke: {backup_dir}")
    print(f"Struktur flat untuk training di: {base_dir}")

def restore_class_structure():
    """
    Mengembalikan struktur folder class dari backup
    """
    base_dir = 'datasets/images'
    backup_dir = 'datasets/images_backup'
    
    if os.path.exists(backup_dir):
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
        shutil.move(backup_dir, base_dir)
        print("Struktur folder class telah dikembalikan.")
    else:
        print("Backup folder tidak ditemukan!")

if __name__ == "__main__":
    print("=== Script Manajemen Struktur Dataset ===")
    print("1. Buat struktur flat untuk training YOLO")
    print("2. Kembalikan struktur folder class")
    
    choice = input("Pilihan (1-2): ")
    
    if choice == '1':
        create_flat_structure()
    elif choice == '2':
        restore_class_structure()
    else:
        print("Pilihan tidak valid!")
