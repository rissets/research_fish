import cv2
import numpy as np
import torch
from ultralytics import YOLO
import base64
from io import BytesIO
from PIL import Image
import os
from django.conf import settings

class YOLOObjectDetectionService:
    """Service untuk melakukan prediksi YOLO object detection (single object)"""
    
    def __init__(self):
        # Load trained YOLO model from training results
        model_path = os.path.join(settings.BASE_DIR, 'researchs', 'best.pt')
        self.model = YOLO(model_path)
    
    def decode_base64_image(self, base64_string):
        """Decode base64 string ke numpy array image"""
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_string)
            image = Image.open(BytesIO(image_data))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            frame = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            return frame
        except Exception as e:
            print(f"Error decoding image: {e}")
            return None
    
    def calculate_bounding_area(self, x1, y1, x2, y2):
        """Menghitung area bounding box dalam pixel"""
        width = x2 - x1
        height = y2 - y1
        area = width * height
        return int(area), int(width), int(height)
    
    def convert_to_yolo_format(self, x1, y1, x2, y2, frame_width, frame_height):
        """Konversi koordinat pixel ke format YOLO (normalized 0-1)"""
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
        
        return float(x_center_norm), float(y_center_norm), float(width_norm), float(height_norm)
    
    def predict_single_object(self, base64_image, frame_number=1):
        """Melakukan prediksi object detection dan mengembalikan objek dengan confidence tertinggi"""
        # Decode image
        frame = self.decode_base64_image(base64_image)
        if frame is None:
            return None
        
        # Jalankan object detection dengan YOLO
        results = self.model(frame, verbose=False)
        
        best_detection = None
        highest_confidence = 0.0
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                frame_height, frame_width = frame.shape[:2]
                
                for i, box in enumerate(boxes):
                    # Ambil koordinat kotak (x1, y1, x2, y2)
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                    
                    # Dapatkan nama kelas dari YOLO
                    yolo_class_id = int(box.cls.item())
                    yolo_class_name = self.model.names[yolo_class_id]
                    confidence = float(box.conf.item())
                    
                    # Simpan deteksi dengan confidence tertinggi
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        
                        # Hitung area bounding box
                        bounding_area, bbox_width, bbox_height = self.calculate_bounding_area(x1, y1, x2, y2)
                        
                        # Konversi ke format YOLO (normalized)
                        bbox_x_center_norm, bbox_y_center_norm, bbox_width_norm, bbox_height_norm = self.convert_to_yolo_format(
                            x1, y1, x2, y2, frame_width, frame_height
                        )
                        
                        # Hitung persentase area dari total frame
                        total_frame_pixels = frame.shape[0] * frame.shape[1]
                        bbox_area_percentage = float((bounding_area / total_frame_pixels) * 100)
                        
                        # Buat data untuk objek terbaik
                        best_detection = {
                            "object_id": 1,  # Single object, so ID is always 1
                            "prediction": yolo_class_name,
                            "class_id": int(yolo_class_id),
                            "confidence": float(confidence),
                            "detection_type": "detection",
                            "bounding_box": {
                                "area_pixels": int(bounding_area),
                                "area_percentage": float(bbox_area_percentage),
                                "coordinates_pixel": [int(x1), int(y1), int(x2), int(y2)],
                                "yolo_format": [int(yolo_class_id), float(bbox_x_center_norm), float(bbox_y_center_norm), float(bbox_width_norm), float(bbox_height_norm)],
                                "size_pixels": [int(bbox_width), int(bbox_height)]
                            }
                        }
        
        if best_detection is None:
            return {
                "frame_number": int(frame_number),
                "frame_size": [int(frame.shape[1]), int(frame.shape[0])],
                "detection": None,
                "message": "No objects detected"
            }
        
        return {
            "frame_number": int(frame_number),
            "frame_size": [int(frame.shape[1]), int(frame.shape[0])],
            "detection": best_detection
        }

class YOLOSegmentationService:
    """Service untuk melakukan prediksi YOLO segmentation"""
    
    def __init__(self):
        # Load YOLO model
        model_path = os.path.join(settings.BASE_DIR, 'researchs', 'yolov8n-seg.pt')
        self.model = YOLO(model_path)
    
    def decode_base64_image(self, base64_string):
        """Decode base64 string ke numpy array image"""
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_string)
            image = Image.open(BytesIO(image_data))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            frame = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            return frame
        except Exception as e:
            print(f"Error decoding image: {e}")
            return None
    
    def calculate_segmentation_area(self, mask):
        """Menghitung area segmentasi dalam pixel"""
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        area = np.sum(mask > 0.5)
        return int(area)
    
    def calculate_bounding_area(self, x1, y1, x2, y2):
        """Menghitung area bounding box dalam pixel"""
        width = x2 - x1
        height = y2 - y1
        area = width * height
        return int(area), int(width), int(height)
    
    def calculate_segmentation_coordinates(self, mask, frame_shape):
        """Menghitung koordinat ekstrem dari area segmentasi"""
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
    
    def convert_to_yolo_format(self, x1, y1, x2, y2, frame_width, frame_height):
        """Konversi koordinat pixel ke format YOLO (normalized 0-1)"""
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
        
        return float(x_center_norm), float(y_center_norm), float(width_norm), float(height_norm)
    
    def extract_segmentation_polygon(self, mask, frame_shape, simplify_factor=0.02):
        """Ekstrak polygon segmentasi dari mask"""
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
    
    def predict_segmentation(self, base64_image, frame_number=1):
        """Melakukan prediksi segmentasi dari base64 image"""
        # Decode image
        frame = self.decode_base64_image(base64_image)
        if frame is None:
            return None
        
        # Jalankan segmentasi dengan YOLO
        results = self.model(frame, verbose=False)
        
        detections = []
        
        for result in results:
            if result.masks is not None:
                masks = result.masks.data
                boxes = result.boxes
                
                frame_height, frame_width = frame.shape[:2]
                
                for i, (mask, box) in enumerate(zip(masks, boxes)):
                    # Ambil koordinat kotak (x1, y1, x2, y2)
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = coords
                    
                    # Dapatkan nama kelas dari YOLO
                    yolo_class_id = int(box.cls.item())
                    yolo_class_name = self.model.names[yolo_class_id]
                    confidence = float(box.conf.item())
                    
                    # Hitung area segmentasi
                    segmentation_area = self.calculate_segmentation_area(mask)
                    
                    # Hitung area bounding box
                    bounding_area, bbox_width, bbox_height = self.calculate_bounding_area(x1, y1, x2, y2)
                    
                    # Hitung koordinat segmentasi
                    seg_x1, seg_y1, seg_x2, seg_y2, seg_width, seg_height = self.calculate_segmentation_coordinates(mask, frame.shape)
                    
                    # Ekstrak polygon segmentasi
                    segmentation_polygons = self.extract_segmentation_polygon(mask, frame.shape)
                    
                    # Konversi ke format YOLO (normalized)
                    seg_x_center_norm, seg_y_center_norm, seg_width_norm, seg_height_norm = self.convert_to_yolo_format(
                        seg_x1, seg_y1, seg_x2, seg_y2, frame_width, frame_height
                    )
                    bbox_x_center_norm, bbox_y_center_norm, bbox_width_norm, bbox_height_norm = self.convert_to_yolo_format(
                        x1, y1, x2, y2, frame_width, frame_height
                    )
                    
                    # Hitung persentase area dari total frame
                    total_frame_pixels = frame.shape[0] * frame.shape[1]
                    seg_area_percentage = (segmentation_area / total_frame_pixels) * 100
                    bbox_area_percentage = (bounding_area / total_frame_pixels) * 100
                    
                    # Hitung rasio segmentasi terhadap bounding box
                    segmentation_ratio = (segmentation_area / bounding_area) * 100 if bounding_area > 0 else 0
                    
                    # Buat data untuk objek ini
                    object_data = {
                        "object_id": int(i + 1),
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
                    
                    detections.append(object_data)
        
        return {
            "frame_number": int(frame_number),
            "frame_size": [int(frame_width), int(frame_height)],
            "detections": detections
        }
