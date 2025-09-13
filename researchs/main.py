from urllib.request import urlopen

import cv2
import torch
import torchvision.models as models
import torchvision.transforms as T
from ultralytics import YOLO

# --- 1. INISIALISASI SEMUA MODEL ---

# Muat Model YOLOv8 (untuk deteksi)
print("Memuat model YOLOv8...")
yolo_model = YOLO("yolov8n.pt")

# Muat Model ResNet-50 (untuk klasifikasi)
print("Memuat model ResNet-50...")
resnet_weights = models.ResNet50_Weights.IMAGENET1K_V2
resnet_model = models.resnet50(weights=resnet_weights)
resnet_model.eval()  # Set ke mode evaluasi

# Dapatkan transformasi preprocessing untuk ResNet
preprocess = resnet_weights.transforms()

# Unduh dan muat label kelas ImageNet untuk ResNet
print("Mengunduh label ImageNet...")
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
try:
    response = urlopen(url)
    labels_text = response.read().decode("utf-8")
    resnet_labels = [s.strip() for s in labels_text.split("\n")]
except Exception as e:
    print(f"Gagal mengunduh label: {e}")
    resnet_labels = ["Gagal unduh label"] * 1000

# --- 2. SETUP WEBCAM ---

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak bisa membuka webcam.")
    exit()

print("\nWebcam terbuka. Tekan 'q' untuk keluar.")

# --- 3. LOOP UTAMA (DETEKSI + KLASIFIKASI) ---

while True:
    success, frame = cap.read()
    if not success:
        break

    # Jalankan deteksi objek dengan YOLO
    # verbose=False agar tidak menampilkan log di terminal
    yolo_results = yolo_model(frame, verbose=False)

    # Loop untuk setiap objek yang terdeteksi oleh YOLO
    for result in yolo_results:
        for box in result.boxes:
            # Ambil koordinat kotak (x1, y1, x2, y2)
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords

            # Dapatkan nama kelas dari YOLO
            yolo_class_id = int(box.cls)
            yolo_class_name = yolo_model.names[yolo_class_id]

            # Potong (crop) objek dari frame
            cropped_object = frame[y1:y2, x1:x2]

            # Lakukan klasifikasi dengan ResNet HANYA jika hasil crop valid
            if cropped_object.size > 0:
                # Konversi BGR (OpenCV) ke RGB (PyTorch)
                rgb_crop = cv2.cvtColor(cropped_object, cv2.COLOR_BGR2RGB)
                img_t = T.ToTensor()(rgb_crop)
                batch_t = preprocess(img_t).unsqueeze(0)

                # Inferensi ResNet
                with torch.no_grad():
                    out = resnet_model(batch_t)

                _, index = torch.max(out, 1)
                percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

                resnet_label = resnet_labels[index[0]]
                resnet_score = percentage[index[0]].item()

                # --- Tampilkan Hasil ke Frame ---
                # 1. Gambar kotak dari YOLO
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 2. Tampilkan label YOLO di atas kotak
                label_yolo = f"YOLO: {yolo_class_name}"
                cv2.putText(
                    frame,
                    label_yolo,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                # 3. Tampilkan label ResNet di bawah label YOLO
                label_resnet = f"ResNet: {resnet_label} ({resnet_score:.1f}%)"
                cv2.putText(
                    frame,
                    label_resnet,
                    (x1, y1 - 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 100, 0),
                    2,
                )

    # Tampilkan frame hasil akhir
    cv2.imshow("YOLO Detection + ResNet Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()
