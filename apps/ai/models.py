from django.db import models
from django.contrib.auth.models import User
import json

class DetectionSession(models.Model):
    """Model untuk menyimpan session deteksi"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    session_id = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        db_table = 'detection_sessions'
        
    def __str__(self):
        return f"Session {self.session_id} - {self.created_at}"

class Detection(models.Model):
    """Model untuk menyimpan hasil deteksi"""
    session = models.ForeignKey(DetectionSession, on_delete=models.CASCADE, related_name='detections')
    frame_number = models.IntegerField()
    frame_width = models.IntegerField()
    frame_height = models.IntegerField()
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'detections'
        
    def __str__(self):
        return f"Detection {self.id} - Frame {self.frame_number}"

class DetectedObject(models.Model):
    """Model untuk menyimpan objek yang terdeteksi"""
    detection = models.ForeignKey(Detection, on_delete=models.CASCADE, related_name='detected_objects')
    object_id = models.IntegerField()  # ID objek dalam frame
    prediction = models.CharField(max_length=100)  # Nama kelas prediksi
    class_id = models.IntegerField()  # ID kelas
    confidence = models.FloatField()  # Confidence score
    
    # Segmentation data
    segmentation_area_pixels = models.IntegerField()
    segmentation_area_percentage = models.FloatField()
    segmentation_polygons = models.JSONField()  # Array of polygon coordinates
    segmentation_bounding_coordinates = models.JSONField()  # [x1, y1, x2, y2]
    segmentation_yolo_format = models.JSONField()  # [class_id, x_center, y_center, width, height]
    segmentation_size = models.JSONField()  # [width, height]
    
    # Bounding box data
    bbox_area_pixels = models.IntegerField()
    bbox_area_percentage = models.FloatField()
    bbox_coordinates = models.JSONField()  # [x1, y1, x2, y2]
    bbox_yolo_format = models.JSONField()  # [class_id, x_center, y_center, width, height]
    bbox_size = models.JSONField()  # [width, height]
    
    # Ratio
    segmentation_bbox_ratio = models.FloatField()
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'detected_objects'
        
    def __str__(self):
        return f"{self.prediction} - {self.confidence:.2f}"
    
    @property
    def segmentation_polygons_formatted(self):
        """Return formatted polygon data"""
        return json.dumps(self.segmentation_polygons, indent=2)
    
    def get_segmentation_summary(self):
        """Get summary of segmentation data"""
        return {
            'prediction': self.prediction,
            'confidence': self.confidence,
            'area_pixels': self.segmentation_area_pixels,
            'area_percentage': self.segmentation_area_percentage,
            'polygon_count': len(self.segmentation_polygons) if self.segmentation_polygons else 0
        }
