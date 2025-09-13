"""
Script untuk deteksi objek real-time menggunakan OpenCV dan model YOLO yang sudah ditraining
Mendukung 2 class: bilal dan andi
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

class ObjectDetector:
    def __init__(self, model_path='runs/detect/bilal_andi_detection/weights/best.pt'):
        """
        Initialize object detector
        Args:
            model_path: Path ke model YOLO yang sudah ditraining
        """
        self.model_path = model_path
        self.model = None
        self.class_names = ['bilal', 'andi']  # Sesuai dengan data.yaml
        
        # Load model jika ada
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"Model loaded: {model_path}")
        else:
            print(f"Model tidak ditemukan: {model_path}")
            print("Gunakan model pre-trained atau train model terlebih dahulu")
    
    def detect_opencv_only(self, image_path):
        """
        Deteksi menggunakan OpenCV saja (tanpa YOLO)
        Untuk testing/fallback jika model YOLO belum ready
        """
        # Load cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Baca gambar
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Tidak dapat membaca gambar {image_path}")
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Deteksi wajah
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Gambar bounding box
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, 'person', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return img, len(faces)
    
    def detect_yolo(self, image_path):
        """
        Deteksi menggunakan model YOLO yang sudah ditraining
        """
        if self.model is None:
            print("Model YOLO tidak tersedia, menggunakan OpenCV...")
            return self.detect_opencv_only(image_path)
        
        # Baca gambar
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Tidak dapat membaca gambar {image_path}")
            return None
        
        # Prediksi dengan YOLO
        results = self.model(image_path)
        
        # Gambar hasil prediksi
        annotated_img = results[0].plot()
        
        # Hitung jumlah deteksi
        boxes = results[0].boxes
        detection_count = len(boxes) if boxes is not None else 0
        
        return annotated_img, detection_count
    
    def detect_webcam(self):
        """
        Deteksi real-time dari webcam
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Tidak dapat mengakses webcam")
            return
        
        print("Tekan 'q' untuk keluar")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if self.model is not None:
                # Gunakan YOLO
                results = self.model(frame)
                annotated_frame = results[0].plot()
            else:
                # Fallback ke OpenCV
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                annotated_frame = frame.copy()
                for (x, y, w, h) in faces:
                    cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, 'person', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow('Object Detection - bilal & andi', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def test_on_dataset(self):
        """
        Test model pada dataset images (mendukung folder terpisah untuk setiap class)
        """
        images_base_dir = 'datasets/images'
        if not os.path.exists(images_base_dir):
            print(f"Folder {images_base_dir} tidak ditemukan!")
            return
        
        # Process folder class (bilal, andi)
        for class_folder in os.listdir(images_base_dir):
            class_path = os.path.join(images_base_dir, class_folder)
            
            if os.path.isdir(class_path):
                print(f"\n=== Testing pada folder: {class_folder} ===")
                
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_path, img_file)
                        
                        print(f"\nMemproses: {class_folder}/{img_file}")
                        
                        # Deteksi
                        if self.model is not None:
                            result_img, count = self.detect_yolo(img_path)
                            method = "YOLO"
                        else:
                            result_img, count = self.detect_opencv_only(img_path)
                            method = "OpenCV"
                        
                        if result_img is not None:
                            print(f"Deteksi {method}: {count} objek ditemukan")
                            
                            # Tampilkan hasil
                            cv2.imshow(f'Detection Result - {class_folder}/{img_file}', result_img)
                            print("Tekan any key untuk lanjut ke gambar berikutnya...")
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()

def main():
    print("=== Object Detection untuk 'bilal' dan 'andi' ===")
    
    # Initialize detector
    detector = ObjectDetector()
    
    while True:
        print("\nPilih mode deteksi:")
        print("1. Test pada dataset images")
        print("2. Deteksi real-time dari webcam")
        print("3. Deteksi pada gambar tertentu")
        print("0. Keluar")
        
        choice = input("Pilihan (0-3): ")
        
        if choice == '1':
            detector.test_on_dataset()
        elif choice == '2':
            detector.detect_webcam()
        elif choice == '3':
            img_path = input("Masukkan path gambar: ")
            if os.path.exists(img_path):
                if detector.model is not None:
                    result_img, count = detector.detect_yolo(img_path)
                    method = "YOLO"
                else:
                    result_img, count = detector.detect_opencv_only(img_path)
                    method = "OpenCV"
                
                if result_img is not None:
                    print(f"Deteksi {method}: {count} objek ditemukan")
                    cv2.imshow('Detection Result', result_img)
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
