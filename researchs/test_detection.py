"""
Script untuk test deteksi objek dengan confidence threshold yang bisa disesuaikan
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

def test_detection_on_dataset():
    """
    Test model YOLO pada dataset dengan berbagai confidence threshold
    """
    # Load model terbaik
    model_path = 'runs/detect/bilal_detection2/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"Model tidak ditemukan: {model_path}")
        return
    
    model = YOLO(model_path)
    images_dir = 'datasets/images'
    
    if not os.path.exists(images_dir):
        print(f"Folder {images_dir} tidak ditemukan!")
        return
    
    # Test dengan berbagai confidence threshold
    confidence_levels = [0.1, 0.25, 0.5, 0.75]
    
    for conf in confidence_levels:
        print(f"\n=== Testing dengan confidence = {conf} ===")
        
        for img_file in os.listdir(images_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(images_dir, img_file)
                
                # Prediksi dengan confidence threshold yang disesuaikan
                results = model(img_path, conf=conf)
                
                # Hitung deteksi
                boxes = results[0].boxes
                detection_count = len(boxes) if boxes is not None else 0
                
                print(f"  {img_file}: {detection_count} deteksi")
                
                if detection_count > 0:
                    # Tampilkan hasil jika ada deteksi
                    annotated_img = results[0].plot()
                    cv2.imshow(f'Detection - {img_file} (conf={conf})', annotated_img)
                    
                    # Tampilkan info deteksi detail
                    for i, box in enumerate(boxes):
                        conf_score = box.conf[0].item()
                        cls = box.cls[0].item()
                        print(f"    Deteksi {i+1}: confidence={conf_score:.3f}, class={int(cls)}")
                    
                    key = cv2.waitKey(0) & 0xFF
                    cv2.destroyAllWindows()
                    
                    if key == ord('q'):
                        return

def test_single_image(image_path, conf=0.25):
    """
    Test deteksi pada satu gambar dengan confidence threshold tertentu
    """
    model_path = 'runs/detect/bilal_detection2/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"Model tidak ditemukan: {model_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"Gambar tidak ditemukan: {image_path}")
        return
    
    model = YOLO(model_path)
    
    print(f"Testing pada: {image_path}")
    print(f"Confidence threshold: {conf}")
    
    # Prediksi
    results = model(image_path, conf=conf)
    
    # Hasil
    boxes = results[0].boxes
    detection_count = len(boxes) if boxes is not None else 0
    
    print(f"Jumlah deteksi: {detection_count}")
    
    if detection_count > 0:
        # Tampilkan detail deteksi
        for i, box in enumerate(boxes):
            conf_score = box.conf[0].item()
            cls = box.cls[0].item()
            xyxy = box.xyxy[0].cpu().numpy()
            print(f"Deteksi {i+1}:")
            print(f"  Confidence: {conf_score:.3f}")
            print(f"  Class: {int(cls)} (bilal)")
            print(f"  Bounding box: {xyxy}")
        
        # Tampilkan gambar hasil
        annotated_img = results[0].plot()
        cv2.imshow('Detection Result', annotated_img)
        print("Tekan any key untuk menutup...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Tidak ada deteksi. Coba dengan confidence threshold yang lebih rendah.")

def main():
    print("=== Test Deteksi Objek YOLO ===")
    
    while True:
        print("\nPilih mode test:")
        print("1. Test pada semua gambar dataset")
        print("2. Test pada gambar tertentu")
        print("3. Test dengan OpenCV fallback")
        print("0. Keluar")
        
        choice = input("Pilihan (0-3): ")
        
        if choice == '1':
            test_detection_on_dataset()
        elif choice == '2':
            img_path = input("Masukkan path gambar: ")
            conf = float(input("Confidence threshold (0.1-1.0): ") or "0.25")
            test_single_image(img_path, conf)
        elif choice == '3':
            # Test dengan OpenCV Haar Cascade sebagai pembanding
            img_path = input("Masukkan path gambar: ")
            if os.path.exists(img_path):
                # Load cascade classifier
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                print(f"OpenCV deteksi: {len(faces)} wajah")
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(img, 'bilal', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                cv2.imshow('OpenCV Detection', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("File tidak ditemukan!")
        elif choice == '0':
            break
        else:
            print("Pilihan tidak valid!")

if __name__ == "__main__":
    main()
