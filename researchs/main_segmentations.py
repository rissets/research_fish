from urllib.request import urlopen
import numpy as np
import cv2
import torch
import json
from ultralytics import YOLO

# --- 1. INISIALISASI MODEL ---

# Muat Model YOLOv8 Segmentation
print("Memuat model YOLOv8 Segmentation...")
yolo_model = YOLO("yolov8n-seg.pt")  # Model segmentasi

# --- 2. FUNGSI UNTUK MENGGAMBAR SEGMENTASI ---

def draw_segmentation_mask(frame, masks, boxes, class_names, colors=None):
    """
    Menggambar mask segmentasi pada frame
    """
    overlay = frame.copy()
    
    if colors is None:
        # Generate random colors for each class
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype=np.uint8)
    
    for i, (mask, box) in enumerate(zip(masks, boxes)):
        # Dapatkan warna untuk kelas ini
        class_id = int(box.cls)
        color = colors[class_id % len(colors)]
        
        # Konversi mask ke format yang sesuai
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        # Pastikan mask dalam bentuk uint8
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        
        # Resize mask ke ukuran frame jika perlu
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        
        # Buat mask berwarna
        colored_mask = np.zeros_like(frame)
        colored_mask[mask > 0] = color
        
        # Terapkan mask dengan transparansi
        alpha = 0.4
        overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)
    
    return overlay

def calculate_segmentation_area(mask):
    """
    Menghitung area segmentasi dalam pixel
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # Hitung jumlah pixel yang tersegmentasi
    area = np.sum(mask > 0.5)
    return area

def calculate_bounding_area(x1, y1, x2, y2):
    """
    Menghitung area bounding box dalam pixel
    """
    width = x2 - x1
    height = y2 - y1
    area = width * height
    return area, width, height

def calculate_segmentation_coordinates(mask, frame_shape):
    """
    Menghitung koordinat ekstrem dari area segmentasi
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # Resize mask ke ukuran frame jika perlu
    if mask.shape[:2] != frame_shape[:2]:
        mask = cv2.resize(mask, (frame_shape[1], frame_shape[0]))
    
    # Buat mask binary
    binary_mask = mask > 0.5
    
    # Cari koordinat yang tersegmentasi
    y_coords, x_coords = np.where(binary_mask)
    
    if len(y_coords) > 0 and len(x_coords) > 0:
        # Koordinat ekstrem
        min_x, max_x = int(np.min(x_coords)), int(np.max(x_coords))
        min_y, max_y = int(np.min(y_coords)), int(np.max(y_coords))
        
        # Dimensi segmentasi
        seg_width = max_x - min_x
        seg_height = max_y - min_y
        
        return min_x, min_y, max_x, max_y, seg_width, seg_height
    else:
        return 0, 0, 0, 0, 0, 0

def convert_to_yolo_format(x1, y1, x2, y2, frame_width, frame_height):
    """
    Konversi koordinat pixel ke format YOLO (normalized 0-1)
    """
    # Hitung center point
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    
    # Hitung width dan height
    width = x2 - x1
    height = y2 - y1
    
    # Normalisasi ke 0-1
    x_center_norm = x_center / frame_width
    y_center_norm = y_center / frame_height
    width_norm = width / frame_width
    height_norm = height / frame_height
    
    return x_center_norm, y_center_norm, width_norm, height_norm

def extract_segmentation_polygon(mask, frame_shape, simplify_factor=0.02):
    """
    Ekstrak polygon segmentasi dari mask
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # Resize mask ke ukuran frame jika perlu
    if mask.shape[:2] != frame_shape[:2]:
        mask = cv2.resize(mask, (frame_shape[1], frame_shape[0]))
    
    # Buat mask binary
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    # Cari kontur
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        # Simplifikasi kontur untuk mengurangi jumlah titik
        epsilon = simplify_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Konversi ke format list koordinat
        if len(approx) >= 3:  # Minimal 3 titik untuk polygon
            polygon = []
            for point in approx:
                x, y = point[0]
                # Normalisasi koordinat ke 0-1
                x_norm = x / frame_shape[1]
                y_norm = y / frame_shape[0]
                polygon.extend([float(x_norm), float(y_norm)])
            polygons.append(polygon)
    
    return polygons

# --- 3. SETUP WEBCAM ---

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak bisa membuka webcam.")
    exit()

print("\nWebcam terbuka. Tekan 'q' untuk keluar.")

# --- 4. LOOP UTAMA (SEGMENTASI SAJA) ---

frame_count = 0
detection_data = []  # List untuk menyimpan data deteksi
detection_found = False  # Flag untuk mendeteksi apakah sudah ada hasil

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    frame_detections = []  # Data deteksi untuk frame ini
    
    # Jalankan segmentasi dengan YOLO
    yolo_results = yolo_model(frame, verbose=False)

    # Loop untuk setiap hasil segmentasi
    for result in yolo_results:
        if result.masks is not None:  # Pastikan ada mask yang terdeteksi
            masks = result.masks.data
            boxes = result.boxes
            
            # Gambar segmentasi mask
            frame = draw_segmentation_mask(frame, masks, boxes, yolo_model.names)
            
            print(f"\n=== Frame {frame_count} - Deteksi Segmentasi ===")
            
            # Loop untuk setiap objek yang tersegmentasi
            for i, (mask, box) in enumerate(zip(masks, boxes)):
                # Ambil koordinat kotak (x1, y1, x2, y2)
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords

                # Dapatkan nama kelas dari YOLO
                yolo_class_id = int(box.cls)
                yolo_class_name = yolo_model.names[yolo_class_id]
                confidence = float(box.conf)

                # Hitung area segmentasi
                segmentation_area = calculate_segmentation_area(mask)
                
                # Hitung area bounding box
                bounding_area, bbox_width, bbox_height = calculate_bounding_area(x1, y1, x2, y2)
                
                # Hitung koordinat segmentasi
                seg_x1, seg_y1, seg_x2, seg_y2, seg_width, seg_height = calculate_segmentation_coordinates(mask, frame.shape)
                
                # Ekstrak polygon segmentasi
                segmentation_polygons = extract_segmentation_polygon(mask, frame.shape)
                
                # Konversi ke format YOLO (normalized)
                frame_height, frame_width = frame.shape[:2]
                seg_x_center_norm, seg_y_center_norm, seg_width_norm, seg_height_norm = convert_to_yolo_format(
                    seg_x1, seg_y1, seg_x2, seg_y2, frame_width, frame_height
                )
                bbox_x_center_norm, bbox_y_center_norm, bbox_width_norm, bbox_height_norm = convert_to_yolo_format(
                    x1, y1, x2, y2, frame_width, frame_height
                )
                
                # Konversi mask ke numpy array untuk ukuran frame
                mask_np = mask.cpu().numpy()
                if mask_np.shape[:2] != frame.shape[:2]:
                    mask_resized = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
                else:
                    mask_resized = mask_np
                
                # Hitung persentase area dari total frame
                total_frame_pixels = frame.shape[0] * frame.shape[1]
                seg_area_percentage = (segmentation_area / total_frame_pixels) * 100
                bbox_area_percentage = (bounding_area / total_frame_pixels) * 100
                
                # Hitung rasio segmentasi terhadap bounding box
                segmentation_ratio = (segmentation_area / bounding_area) * 100 if bounding_area > 0 else 0

                # Buat data JSON untuk objek ini
                object_data = {
                    "object_id": i + 1,
                    "prediction": yolo_class_name,
                    "class_id": int(yolo_class_id),
                    "confidence": float(confidence),
                    "segmentation": {
                        "area_pixels": int(segmentation_area),
                        "area_percentage": float(seg_area_percentage),
                        "polygons": segmentation_polygons,
                        "bounding_coordinates_pixel": [int(seg_x1), int(seg_y1), int(seg_x2), int(seg_y2)],
                        "yolo_format": [int(yolo_class_id), float(seg_x_center_norm), float(seg_y_center_norm), float(seg_width_norm), float(seg_height_norm)],
                        "size_pixels": [int(seg_width), int(seg_height)]
                    },
                    "bounding_box": {
                        "area_pixels": int(bounding_area),
                        "area_percentage": float(bbox_area_percentage),
                        "coordinates_pixel": [int(x1), int(y1), int(x2), int(y2)],
                        "yolo_format": [int(yolo_class_id), float(bbox_x_center_norm), float(bbox_y_center_norm), float(bbox_width_norm), float(bbox_height_norm)],
                        "size_pixels": [int(bbox_width), int(bbox_height)]
                    },
                    "segmentation_bbox_ratio": float(segmentation_ratio)
                }
                
                frame_detections.append(object_data)

                # Output ke terminal
                print(f"Objek {i+1}:")
                print(f"  - Prediction: {yolo_class_name}")
                print(f"  - Confidence Score: {confidence:.4f}")
                print(f"  - Segmentation Area: {segmentation_area} pixels ({seg_area_percentage:.2f}% dari frame)")
                print(f"  - Segmentation Coordinates (pixel): ({seg_x1}, {seg_y1}) - ({seg_x2}, {seg_y2})")
                print(f"  - Segmentation YOLO Format: {yolo_class_id} {seg_x_center_norm:.6f} {seg_y_center_norm:.6f} {seg_width_norm:.6f} {seg_height_norm:.6f}")
                print(f"  - Segmentation Polygons: {len(segmentation_polygons)} polygon(s)")
                for j, polygon in enumerate(segmentation_polygons):
                    print(f"    Polygon {j+1}: {len(polygon)//2} points")
                print(f"  - Segmentation Size: {seg_width}x{seg_height} pixels")
                print(f"  - Bounding Box Area: {bounding_area} pixels ({bbox_area_percentage:.2f}% dari frame)")
                print(f"  - Bounding Box Coordinates (pixel): ({x1}, {y1}) - ({x2}, {y2})")
                print(f"  - Bounding Box YOLO Format: {yolo_class_id} {bbox_x_center_norm:.6f} {bbox_y_center_norm:.6f} {bbox_width_norm:.6f} {bbox_height_norm:.6f}")
                print(f"  - Bounding Box Size: {bbox_width}x{bbox_height} pixels")
                print(f"  - Segmentation/Bounding Ratio: {segmentation_ratio:.1f}%")
                
                # Tampilkan informasi di frame
                label_info = f"{yolo_class_name}: {confidence:.2f}"
                area_info = f"Seg: {segmentation_area}px | BBox: {bounding_area}px"
                ratio_info = f"Ratio: {segmentation_ratio:.1f}%"
                coord_info = f"SegCoord: ({seg_x1},{seg_y1})-({seg_x2},{seg_y2})"
                
                cv2.putText(
                    frame,
                    label_info,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                
                cv2.putText(
                    frame,
                    area_info,
                    (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 100, 0),
                    1,
                )
                
                cv2.putText(
                    frame,
                    ratio_info,
                    (x1, y1 - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (100, 150, 255),
                    1,
                )
                
                cv2.putText(
                    frame,
                    coord_info,
                    (x1, y1 - 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (200, 200, 0),
                    1,
                )
                
                # Gambar bounding box (hijau)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Gambar segmentation bounding (merah)
                if seg_x1 != seg_x2 and seg_y1 != seg_y2:
                    cv2.rectangle(frame, (seg_x1, seg_y1), (seg_x2, seg_y2), (0, 0, 255), 1)
                
                # Gambar kontur segmentasi (biru)
                mask_uint8 = (mask_resized > 0.5).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, (255, 0, 0), 2)
                
                # Set flag bahwa deteksi sudah ditemukan
                detection_found = True
    
    # Simpan data deteksi frame ini
    if frame_detections:
        frame_data = {
            "frame_number": int(frame_count),
            "frame_size": [int(frame_width), int(frame_height)],
            "detections": frame_detections
        }
        detection_data.append(frame_data)
        
        # Print JSON data untuk frame ini
        print(f"\n=== JSON Data Frame {frame_count} ===")
        print(json.dumps(frame_data, indent=2))
        
        # Simpan data segmentasi ke file JSON langsung
        output_file = "segmentation_results.json"
        with open(output_file, 'w') as f:
            json.dump({
                "total_frames": int(len(detection_data)),
                "frames": detection_data
            }, f, indent=2)
        print(f"\nData segmentasi disimpan ke: {output_file}")
        
        # Tunggu beberapa detik untuk melihat hasil, lalu keluar
        print("\nDeteksi selesai! Tekan 'q' untuk keluar atau tunggu 5 detik...")
        
        # Tampilkan frame hasil selama 5 detik
        for countdown in range(50):  # 50 * 100ms = 5 detik
            cv2.imshow("YOLO Segmentation", frame)
            if cv2.waitKey(100) & 0xFF == ord("q"):
                break
        break

    # Tampilkan frame hasil akhir
    cv2.imshow("YOLO Segmentation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Simpan semua data deteksi ke file JSON (jika belum tersimpan)
if detection_data and not detection_found:
    output_file = "segmentation_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "total_frames": int(len(detection_data)),
            "frames": detection_data
        }, f, indent=2)
    print(f"\nData segmentasi disimpan ke: {output_file}")

print("\nProgram selesai.")

# Bersihkan
cap.release()
cv2.destroyAllWindows()