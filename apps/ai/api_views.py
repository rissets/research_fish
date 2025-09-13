from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework.views import APIView
from django.shortcuts import get_object_or_404
from django.db import models
from .models import DetectionSession, Detection, DetectedObject
from .serializers import (
    DetectionSessionSerializer, DetectionSerializer, 
    DetectedObjectSerializer, DetectionSessionCreateSerializer
)
from .services import YOLOSegmentationService, YOLOObjectDetectionService
import base64
import json

class DetectionSessionViewSet(viewsets.ModelViewSet):
    """ViewSet untuk mengelola detection sessions"""
    queryset = DetectionSession.objects.all()
    serializer_class = DetectionSessionSerializer
    permission_classes = [AllowAny]
    
    def get_serializer_class(self):
        if self.action == 'create':
            return DetectionSessionCreateSerializer
        return DetectionSessionSerializer
    
    @action(detail=True, methods=['get'])
    def detections(self, request, pk=None):
        """Get all detections for a session"""
        session = self.get_object()
        detections = session.detections.all().order_by('-timestamp')
        serializer = DetectionSerializer(detections, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'])
    def close(self, request, pk=None):
        """Close a detection session"""
        session = self.get_object()
        session.is_active = False
        session.save()
        
        return Response({
            'message': 'Session closed successfully',
            'session_id': session.session_id
        })

class DetectionViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet untuk melihat detection results"""
    queryset = Detection.objects.all()
    serializer_class = DetectionSerializer
    permission_classes = [AllowAny]
    
    def get_queryset(self):
        queryset = Detection.objects.all()
        session_id = self.request.query_params.get('session_id', None)
        if session_id is not None:
            queryset = queryset.filter(session__session_id=session_id)
        return queryset.order_by('-timestamp')
    
    @action(detail=True, methods=['get'])
    def detected_objects(self, request, pk=None):
        """Get all detected objects for a detection"""
        detection = self.get_object()
        objects = detection.detected_objects.all()
        serializer = DetectedObjectSerializer(objects, many=True)
        return Response(serializer.data)

class DetectedObjectViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet untuk melihat detected objects"""
    queryset = DetectedObject.objects.all()
    serializer_class = DetectedObjectSerializer
    permission_classes = [AllowAny]
    
    def get_queryset(self):
        queryset = DetectedObject.objects.all()
        session_id = self.request.query_params.get('session_id', None)
        prediction = self.request.query_params.get('prediction', None)
        
        if session_id is not None:
            queryset = queryset.filter(detection__session__session_id=session_id)
        if prediction is not None:
            queryset = queryset.filter(prediction__icontains=prediction)
            
        return queryset.order_by('-created_at')
    
    @action(detail=False, methods=['get'])
    def statistics(self, request):
        """Get detection statistics"""
        total_objects = DetectedObject.objects.count()
        unique_predictions = DetectedObject.objects.values('prediction').distinct().count()
        avg_confidence = DetectedObject.objects.aggregate(
            avg_confidence=models.Avg('confidence')
        )['avg_confidence'] or 0
        
        # Top predictions
        top_predictions = DetectedObject.objects.values('prediction').annotate(
            count=models.Count('prediction')
        ).order_by('-count')[:10]
        
        return Response({
            'total_objects': total_objects,
            'unique_predictions': unique_predictions,
            'average_confidence': round(avg_confidence, 4),
            'top_predictions': top_predictions
        })

class PredictionAPIView(APIView):
    """API untuk melakukan prediksi segmentasi secara langsung"""
    permission_classes = [AllowAny]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.yolo_service = YOLOSegmentationService()
    
    def post(self, request):
        """Prediksi dari base64 image"""
        try:
            base64_image = request.data.get('image')
            save_to_db = request.data.get('save_to_db', True)
            session_id = request.data.get('session_id', None)
            
            if not base64_image:
                return Response({
                    'error': 'No image data provided'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Perform prediction
            result = self.yolo_service.predict_segmentation(base64_image)
            
            if result is None:
                return Response({
                    'error': 'Failed to process image'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Save to database if requested
            if save_to_db and result['detections']:
                if session_id:
                    try:
                        session = DetectionSession.objects.get(session_id=session_id)
                    except DetectionSession.DoesNotExist:
                        session = DetectionSession.objects.create(session_id=session_id)
                else:
                    import uuid
                    session = DetectionSession.objects.create(
                        session_id=str(uuid.uuid4())
                    )
                
                # Create detection record
                detection = Detection.objects.create(
                    session=session,
                    frame_number=result['frame_number'],
                    frame_width=result['frame_size'][0],
                    frame_height=result['frame_size'][1]
                )
                
                # Save detected objects
                for obj_data in result['detections']:
                    DetectedObject.objects.create(
                        detection=detection,
                        object_id=obj_data['object_id'],
                        prediction=obj_data['prediction'],
                        class_id=obj_data['class_id'],
                        confidence=obj_data['confidence'],
                        
                        # Segmentation data
                        segmentation_area_pixels=obj_data['segmentation']['area_pixels'],
                        segmentation_area_percentage=obj_data['segmentation']['area_percentage'],
                        segmentation_polygons=obj_data['segmentation']['polygons'],
                        segmentation_bounding_coordinates=obj_data['segmentation']['bounding_coordinates_pixel'],
                        segmentation_yolo_format=obj_data['segmentation']['yolo_format'],
                        segmentation_size=obj_data['segmentation']['size_pixels'],
                        
                        # Bounding box data
                        bbox_area_pixels=obj_data['bounding_box']['area_pixels'],
                        bbox_area_percentage=obj_data['bounding_box']['area_percentage'],
                        bbox_coordinates=obj_data['bounding_box']['coordinates_pixel'],
                        bbox_yolo_format=obj_data['bounding_box']['yolo_format'],
                        bbox_size=obj_data['bounding_box']['size_pixels'],
                        
                        # Ratio
                        segmentation_bbox_ratio=obj_data['segmentation_bbox_ratio']
                    )
                
                result['detection_id'] = detection.id
                result['session_id'] = session.session_id
            
            return Response({
                'success': True,
                'result': result
            })
            
        except Exception as e:
            return Response({
                'error': f'Prediction error: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SegmentationModelInfoView(APIView):
    """API untuk mendapatkan informasi model segmentasi"""
    permission_classes = [AllowAny]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.yolo_service = YOLOSegmentationService()

    def get(self, request):
        """Get information about the YOLO segmentation model"""
        try:
            return Response({
                'model_loaded': self.yolo_service.model is not None,
                'model_type': 'YOLOv8n-seg',
                'classes': list(self.yolo_service.model.names.values()) if self.yolo_service.model else []
            })
        except Exception as e:
            return Response({
                'error': f'Error getting model info: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ObjectDetectionAPIView(APIView):
    """API untuk melakukan prediksi object detection (single object) secara langsung"""
    permission_classes = [AllowAny]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.yolo_service = YOLOObjectDetectionService()
    
    def post(self, request):
        """Prediksi object detection dari base64 image"""
        try:
            base64_image = request.data.get('image')
            save_to_db = request.data.get('save_to_db', True)
            session_id = request.data.get('session_id', None)
            
            if not base64_image:
                return Response({
                    'error': 'No image data provided'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Perform prediction
            result = self.yolo_service.predict_single_object(base64_image)
            
            if result is None:
                return Response({
                    'error': 'Failed to process image'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Save to database if requested and detection exists
            if save_to_db and result.get('detection'):
                if session_id:
                    try:
                        session = DetectionSession.objects.get(session_id=session_id)
                    except DetectionSession.DoesNotExist:
                        session = DetectionSession.objects.create(session_id=session_id)
                else:
                    import uuid
                    session = DetectionSession.objects.create(
                        session_id=str(uuid.uuid4())
                    )
                
                # Create detection record
                detection = Detection.objects.create(
                    session=session,
                    frame_number=result['frame_number'],
                    frame_width=result['frame_size'][0],
                    frame_height=result['frame_size'][1]
                )
                
                # Save detected object
                obj_data = result['detection']
                DetectedObject.objects.create(
                    detection=detection,
                    object_id=obj_data['object_id'],
                    prediction=obj_data['prediction'],
                    class_id=obj_data['class_id'],
                    confidence=obj_data['confidence'],
                    detection_type='detection',
                    
                    # Bounding box data
                    bbox_area_pixels=obj_data['bounding_box']['area_pixels'],
                    bbox_area_percentage=obj_data['bounding_box']['area_percentage'],
                    bbox_coordinates=obj_data['bounding_box']['coordinates_pixel'],
                    bbox_yolo_format=obj_data['bounding_box']['yolo_format'],
                    bbox_size=obj_data['bounding_box']['size_pixels']
                )
                
                result['detection_id'] = detection.id
                result['session_id'] = session.session_id
            
            return Response({
                'success': True,
                'result': result
            })
            
        except Exception as e:
            return Response({
                'error': f'Object detection error: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class DetectionModelInfoView(APIView):
    """API untuk mendapatkan informasi model detection"""
    permission_classes = [AllowAny]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.yolo_service = YOLOObjectDetectionService()

    def get(self, request):
        """Get information about the object detection YOLO model"""
        try:
            return Response({
                'model_loaded': self.yolo_service.model is not None,
                'model_type': 'YOLOv8n Detection (Custom Trained)',
                'model_path': 'training_results/fish_detection_20250913_160827/weights/best.pt',
                'classes': list(self.yolo_service.model.names.values()) if self.yolo_service.model else [],
                'task': 'single_object_detection'
            })
        except Exception as e:
            return Response({
                'error': f'Error getting model info: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
