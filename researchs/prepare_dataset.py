"""
Script untuk mempersiapkan dataset YOLO untuk object detection dengan 2 class: "bilal" dan "andi"
"""

import os
import cv2
import numpy as np
from pathlib import Path

def create_labels_folder():
    """Membuat folder labels jika belum ada"""
    labels_dir = 'datasets/labels'
    os.makedirs(labels_dir, exist_ok=True)
    print(f"Folder {labels_dir} telah dibuat/sudah ada")
    return labels_dir

def detect_face_opencv(image_path):
    """
    Deteksi wajah menggunakan OpenCV Haar Cascade
    Returns: (x, y, w, h) dalam pixel coordinates
    """
    # Load cascade classifier untuk deteksi wajah
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Baca gambar
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Tidak dapat membaca gambar {image_path}")
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        # Ambil wajah terbesar jika ada beberapa
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        return largest_face, img.shape
    
    return None, img.shape

def convert_to_yolo_format(bbox, img_shape):
    """
    Konversi bounding box dari pixel ke format YOLO
    bbox: (x, y, w, h) dalam pixel
    img_shape: (height, width, channels)
    returns: (x_center, y_center, width, height) dalam rasio 0-1
    """
    x, y, w, h = bbox
    img_h, img_w = img_shape[:2]
    
    # Konversi ke center coordinates dan normalize
    x_center = (x + w/2) / img_w
    y_center = (y + h/2) / img_h
    width = w / img_w
    height = h / img_h
    
    return x_center, y_center, width, height

def get_class_id_from_path(img_path):
    """
    Mendapatkan class ID berdasarkan path gambar
    """
    if 'bilal' in img_path.lower():
        return 0, 'bilal'
    elif 'andi' in img_path.lower():
        return 1, 'andi'
    else:
        return 0, 'unknown'  # Default ke class 0

def process_images_and_create_labels():
    """
    Process semua gambar di datasets/images dan buat file label YOLO
    Mendukung folder terpisah untuk setiap class
    """
    images_base_dir = 'datasets/images'
    labels_dir = create_labels_folder()
    
    if not os.path.exists(images_base_dir):
        print(f"Error: Folder {images_base_dir} tidak ditemukan!")
        return
    
    processed_count = 0
    class_counts = {'bilal': 0, 'andi': 0}
    
    # Process folder class (bilal, andi)
    for class_folder in os.listdir(images_base_dir):
        class_path = os.path.join(images_base_dir, class_folder)
        
        if os.path.isdir(class_path):
            print(f"\nMemproses folder: {class_folder}")
            
            # Process setiap gambar dalam folder class
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, img_file)
                    
                    # Deteksi wajah/objek
                    detection_result, img_shape = detect_face_opencv(img_path)
                    
                    # Tentukan class ID berdasarkan folder
                    if class_folder.lower() == 'bilal':
                        class_id = 0
                        class_name = 'bilal'
                    elif class_folder.lower() == 'andi':
                        class_id = 1
                        class_name = 'andi'
                    else:
                        class_id = 0
                        class_name = 'unknown'
                    
                    # Nama file label
                    label_file = os.path.splitext(img_file)[0] + '.txt'
                    label_path = os.path.join(labels_dir, label_file)
                    
                    with open(label_path, 'w') as f:
                        if detection_result is not None:
                            # Konversi ke format YOLO
                            yolo_coords = convert_to_yolo_format(detection_result, img_shape)
                            f.write(f"{class_id} {yolo_coords[0]:.6f} {yolo_coords[1]:.6f} {yolo_coords[2]:.6f} {yolo_coords[3]:.6f}\n")
                            print(f"✓ {img_file}: {class_name} terdeteksi dan label dibuat")
                            class_counts[class_name] += 1
                        else:
                            # File kosong jika tidak ada deteksi
                            print(f"⚠ {img_file}: Tidak ada objek terdeteksi, file label kosong")
                    
                    processed_count += 1
    
    print(f"\nSelesai! {processed_count} gambar telah diproses.")
    print(f"Statistik class:")
    for class_name, count in class_counts.items():
        print(f"  - {class_name}: {count} deteksi")
    print(f"File label tersimpan di: {labels_dir}")

if __name__ == "__main__":
    print("=== Mempersiapkan Dataset YOLO untuk Object Detection ===")
    print("Class: bilal (class_id = 0), andi (class_id = 1)")
    print()
    
    process_images_and_create_labels()
