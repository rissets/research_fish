from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .models import DetectionSession, Detection, DetectedObject
import uuid
import json

def index(request):
    """Main page for camera detection"""
    return render(request, 'ai/index.html')

@csrf_exempt
@require_http_methods(["POST"])
def create_session(request):
    """Create a new detection session"""
    try:
        session_id = str(uuid.uuid4())
        session = DetectionSession.objects.create(
            session_id=session_id,
            user=request.user if request.user.is_authenticated else None
        )
        
        return JsonResponse({
            'success': True,
            'session_id': session_id,
            'websocket_url': f'ws://localhost:8000/ws/detection/{session_id}/'
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

@require_http_methods(["GET"])
def get_session_history(request, session_id):
    """Get detection history for a session"""
    try:
        session = DetectionSession.objects.get(session_id=session_id)
        detections = Detection.objects.filter(session=session).order_by('-timestamp')
        
        history = []
        for detection in detections:
            objects = []
            for obj in detection.detected_objects.all():
                objects.append({
                    'prediction': obj.prediction,
                    'confidence': obj.confidence,
                    'segmentation_area': obj.segmentation_area_pixels,
                    'segmentation_polygons': obj.segmentation_polygons,
                    'bbox_coordinates': obj.bbox_coordinates
                })
            
            history.append({
                'detection_id': detection.id,
                'frame_number': detection.frame_number,
                'timestamp': detection.timestamp.isoformat(),
                'objects': objects
            })
        
        return JsonResponse({
            'success': True,
            'session_id': session_id,
            'history': history
        })
    except DetectionSession.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'Session not found'
        }, status=404)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

@require_http_methods(["GET"])
def get_all_sessions(request):
    """Get all detection sessions"""
    try:
        sessions = DetectionSession.objects.all().order_by('-created_at')
        
        sessions_data = []
        for session in sessions:
            detection_count = Detection.objects.filter(session=session).count()
            object_count = DetectedObject.objects.filter(detection__session=session).count()
            
            sessions_data.append({
                'session_id': session.session_id,
                'created_at': session.created_at.isoformat(),
                'is_active': session.is_active,
                'detection_count': detection_count,
                'object_count': object_count
            })
        
        return JsonResponse({
            'success': True,
            'sessions': sessions_data
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

@require_http_methods(["GET"])
def get_detection_stats(request):
    """Get overall detection statistics"""
    try:
        total_sessions = DetectionSession.objects.count()
        total_detections = Detection.objects.count()
        total_objects = DetectedObject.objects.count()
        
        # Most detected classes
        from django.db.models import Count
        top_classes = DetectedObject.objects.values('prediction').annotate(
            count=Count('prediction')
        ).order_by('-count')[:10]
        
        return JsonResponse({
            'success': True,
            'stats': {
                'total_sessions': total_sessions,
                'total_detections': total_detections,
                'total_objects': total_objects,
                'top_classes': list(top_classes)
            }
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)
