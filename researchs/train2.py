# ===================================================================================
# PANDUAN PELATIHAN DETEKSI IKAN DENGAN YOLOv8 & FASTER R-CNN
# ===================================================================================
#
# Deskripsi:
# Skrip ini menyediakan dua alur kerja untuk melatih model deteksi objek pada
# dataset ikan dari Roboflow:
#
# 1. YOLOv8 (Single-Stage Detector):
#    - Cepat, efisien, dan menggunakan high-level trainer dari library Ultralytics.
#    - Sangat baik untuk aplikasi real-time.
#
# 2. Faster R-CNN dengan ResNet-50 FPN (Two-Stage Detector):
#    - Lebih kompleks, memprioritaskan akurasi.
#    - Memerlukan implementasi manual dari Dataset, DataLoader, dan training loop
#      menggunakan PyTorch dan TorchVision.
#
# Anda dapat memilih model mana yang akan dilatih di bagian bawah skrip.
#
# ===================================================================================

# --- 1. Instalasi & Setup ---
# Pastikan Anda telah menginstal library yang diperlukan.
# Buka terminal atau command prompt Anda dan jalankan perintah berikut:
# pip install ultralytics roboflow torch torchvision pillow matplotlib opencv-python pyyaml

import os

import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
import yaml  # Ditambahkan untuk membaca file data.yaml
from PIL import Image
from roboflow import Roboflow
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ultralytics import YOLO

# --- Helper Functions & Classes untuk Faster R-CNN ---


def yolo_to_pascal_voc(x_center, y_center, w, h, img_width, img_height):
    """
    Mengonversi format anotasi YOLO (relatif) ke Pascal VOC (absolut).
    Faster R-CNN membutuhkan format [xmin, ymin, xmax, ymax].
    """
    x_center *= img_width
    y_center *= img_height
    w *= img_width
    h *= img_height
    xmin = int(x_center - w / 2)
    ymin = int(y_center - h / 2)
    xmax = int(x_center + w / 2)
    ymax = int(y_center + h / 2)
    return [xmin, ymin, xmax, ymax]


class FishDataset(Dataset):
    """
    Custom PyTorch Dataset untuk memuat gambar ikan dan anotasinya.
    """

    def __init__(self, img_dir, label_dir, transforms=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.img_files = sorted(
            [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
        )

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # Baca gambar dan konversi ke RGB
        image = cv2.imread(img_path)
        if image is None:
            print(f"Peringatan: Tidak dapat membaca gambar {img_path}. Melewati.")
            return None, None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width, _ = image.shape

        # Baca anotasi
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_name)

        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue  # Lewati baris yang tidak valid
                    class_id, x_center, y_center, w, h = map(float, parts)
                    box = yolo_to_pascal_voc(
                        x_center, y_center, w, h, img_width, img_height
                    )
                    boxes.append(box)
                    # Class ID ditambah 1, karena 0 adalah background di Faster R-CNN
                    labels.append(int(class_id) + 1)

        # Buat target dictionary
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])

        if self.transforms:
            # Konversi ke PIL Image untuk transformasi torchvision
            image = Image.fromarray(image)
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.img_files)


def get_transform():
    """Definisikan transformasi standar untuk gambar."""
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )


def collate_fn(batch):
    """Fungsi untuk menangani batching data dengan ukuran anotasi yang berbeda."""
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return torch.tensor([]), torch.tensor([])
    return tuple(zip(*batch))


# --- Fungsi Pelatihan untuk Setiap Model ---


def train_yolov8(dataset_yaml_path, device):
    """
    Fungsi untuk melatih model YOLOv8.
    """
    print("\n--- Memulai Pelatihan Model YOLOv8 ---")

    # Memuat model 'yolov8n.pt'
    print("\nMemuat model pretrained YOLOv8n...")
    model = YOLO("yolov8n.pt")
    model.to(device)

    # Pelatihan / Fine-Tuning
    print("\nMemulai proses fine-tuning model...")
    results = model.train(
        data=dataset_yaml_path,
        epochs=25,
        imgsz=640,
        batch=16,
        name="yolov8n_fish_pytorch_ds",
    )

    print("Pelatihan YOLOv8 selesai.")
    best_model_path = os.path.join(
        "runs", "detect", "yolov8n_fish_pytorch_ds", "weights", "best.pt"
    )
    print(f"Model terbaik disimpan di: {best_model_path}")
    return best_model_path


def train_faster_rcnn(dataset_location, num_classes, device):
    """
    Fungsi untuk melatih model Faster R-CNN.
    """
    print("\n--- Memulai Pelatihan Model Faster R-CNN ---")

    # 1. Setup Dataset dan DataLoader
    train_img_dir = os.path.join(dataset_location, "train", "images")
    train_label_dir = os.path.join(dataset_location, "train", "labels")

    dataset = FishDataset(train_img_dir, train_label_dir, get_transform())
    data_loader = DataLoader(
        dataset,
        batch_size=4,  # Kurangi jika terjadi error Out of Memory
        shuffle=True,
        collate_fn=collate_fn,
    )

    # 2. Memuat Model Faster R-CNN Pretrained
    print("\nMemuat model pretrained Faster R-CNN dengan ResNet50...")
    # num_classes + 1 (untuk background)
    model_num_classes = num_classes + 1
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Ganti kepala klasifikasi (box predictor)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, model_num_classes)
    model.to(device)

    # 3. Setup Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # 4. Training Loop
    num_epochs = 10  # Mulai dengan epoch yang lebih sedikit untuk Faster R-CNN
    print(f"\nMemulai fine-tuning untuk {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, (images, targets) in enumerate(data_loader):
            if not images:
                continue  # Lewati batch kosong
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            if (i + 1) % 50 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{
                        len(data_loader)
                    }], Loss: {losses.item()}"
                )

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] selesai. Rata-rata Loss: {
                total_loss / len(data_loader)
            }"
        )

    # Simpan model
    model_save_path = "fasterrcnn_fish_pytorch_ds.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Pelatihan Faster R-CNN selesai. Model disimpan di: {model_save_path}")
    return model_save_path


def main(model_choice="yolov8"):
    """
    Fungsi utama untuk menjalankan seluruh alur kerja.
    Pilih model dengan mengubah argumen: 'yolov8' atau 'faster_rcnn'
    """
    print("Memeriksa ketersediaan GPU...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Menggunakan perangkat: {device}")

    # --- Mengunduh Dataset dari Roboflow ---
    print("\nMengunduh dataset deteksi dari Roboflow...")
    try:
        rf = Roboflow(api_key="pZkh6tKNxrvmMCh2Xdgu")
        project = rf.workspace("daniel-5cnur").project("fish-pytorch")
        dataset = project.version(11).download("yolov8")
        dataset_yaml_path = os.path.join(dataset.location, "data.yaml")

        # --- PERBAIKAN DI SINI ---
        # Membaca informasi kelas langsung dari file data.yaml yang diunduh.
        # Ini lebih andal daripada mengandalkan API .model.classes
        with open(dataset_yaml_path, "r") as f:
            data_yaml = yaml.safe_load(f)

        class_list = data_yaml["names"]
        num_classes = len(class_list)

        print(f"Dataset berhasil diunduh ke: {dataset.location}")
        print(f"Jumlah kelas dalam dataset: {num_classes}")
        print(f"Nama kelas: {class_list}")

    except Exception as e:
        print(f"\n[ERROR] Gagal mengunduh atau memproses dataset dari Roboflow: {e}")
        return

    # --- Memilih dan Melatih Model ---
    if model_choice == "yolov8":
        trained_model_path = train_yolov8(dataset_yaml_path, device)
        model = YOLO(trained_model_path)
        class_names = model.names
    elif model_choice == "faster_rcnn":
        trained_model_path = train_faster_rcnn(dataset.location, num_classes, device)
        # Muat model untuk inferensi
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
        model.load_state_dict(torch.load(trained_model_path, map_location=device))
        model.to(device)
        model.eval()
        # Buat dictionary nama kelas untuk inferensi
        class_names = {i + 1: name for i, name in enumerate(class_list)}

    else:
        print(
            f"[ERROR] Pilihan model tidak valid: {
                model_choice
            }. Pilih 'yolov8' atau 'faster_rcnn'."
        )
        return

    # --- Inferensi pada Gambar Baru ---
    print("\nMenjalankan inferensi pada gambar contoh...")
    validation_image_dir = os.path.join(dataset.location, "valid", "images")
    if not os.path.exists(validation_image_dir) or not os.listdir(validation_image_dir):
        print("[ERROR] Direktori gambar validasi kosong atau tidak ditemukan.")
        return

    sample_image_name = os.listdir(validation_image_dir)[0]
    sample_image_path = os.path.join(validation_image_dir, sample_image_name)

    print(f"Gambar yang digunakan untuk inferensi: {sample_image_path}")

    # Tampilkan hasil
    if model_choice == "yolov8":
        results = model(sample_image_path)
        for r in results:
            im_array = r.plot()
            im = Image.fromarray(im_array[..., ::-1])
    elif model_choice == "faster_rcnn":
        img = Image.open(sample_image_path).convert("RGB")
        img_tensor = torchvision.transforms.ToTensor()(img).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(img_tensor)

        # Gambar anotasi secara manual
        img_cv = cv2.imread(sample_image_path)
        for box, label, score in zip(
            prediction[0]["boxes"], prediction[0]["labels"], prediction[0]["scores"]
        ):
            if score > 0.5:  # Tampilkan hanya deteksi dengan confidence > 0.5
                xmin, ymin, xmax, ymax = map(int, box.cpu().numpy())
                class_name = class_names.get(label.item(), "Unknown")
                cv2.rectangle(img_cv, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(
                    img_cv,
                    f"{class_name}: {score:.2f}",
                    (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
        im = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    plt.figure(figsize=(12, 12))
    plt.imshow(im)
    plt.title(f"Hasil Deteksi Objek ({model_choice})")
    plt.axis("off")
    plt.show()

    print("\nProses selesai.")


if __name__ == "__main__":
    # =================================================================
    # === UBAH PILIHAN ANDA DI SINI ===
    # Pilih antara 'yolov8' atau 'faster_rcnn'
    # =================================================================
    pilihan_model = "yolov8"
    main(model_choice=pilihan_model)
