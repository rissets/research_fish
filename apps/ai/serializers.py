from rest_framework import serializers
from .models import DetectionSession, Detection, DetectedObject

class DetectedObjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = DetectedObject
        fields = [
            'id', 'object_id', 'prediction', 'class_id', 'confidence',
            'segmentation_area_pixels', 'segmentation_area_percentage', 
            'segmentation_polygons', 'segmentation_bounding_coordinates',
            'segmentation_yolo_format', 'segmentation_size',
            'bbox_area_pixels', 'bbox_area_percentage', 'bbox_coordinates',
            'bbox_yolo_format', 'bbox_size', 'segmentation_bbox_ratio',
            'created_at'
        ]

class DetectionSerializer(serializers.ModelSerializer):
    detected_objects = DetectedObjectSerializer(many=True, read_only=True)
    
    class Meta:
        model = Detection
        fields = [
            'id', 'session', 'frame_number', 'frame_width', 
            'frame_height', 'timestamp', 'detected_objects'
        ]

class DetectionSessionSerializer(serializers.ModelSerializer):
    detections = DetectionSerializer(many=True, read_only=True)
    
    class Meta:
        model = DetectionSession
        fields = [
            'id', 'session_id', 'created_at', 'updated_at', 
            'is_active', 'detections'
        ]

class DetectionSessionCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = DetectionSession
        fields = ['session_id', 'user']

class DetectionResultSerializer(serializers.Serializer):
    """Serializer untuk hasil deteksi yang dikirim melalui WebSocket"""
    session_id = serializers.CharField()
    frame_number = serializers.IntegerField()
    frame_size = serializers.ListField(child=serializers.IntegerField())
    detections = serializers.ListField(child=serializers.DictField())
    timestamp = serializers.DateTimeField()
