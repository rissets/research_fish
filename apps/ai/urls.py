from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .api_views import (
    DetectionSessionViewSet, DetectionViewSet, 
    DetectedObjectViewSet, PredictionAPIView, ObjectDetectionAPIView
)
from .static_views import serve_test_websocket, serve_test_websocket_unified
from .views import SegmentationWebSocketTestView, UnifiedWebSocketTestView

# Create router for API endpoints
router = DefaultRouter()
router.register(r'sessions', DetectionSessionViewSet)
router.register(r'detections', DetectionViewSet)
router.register(r'objects', DetectedObjectViewSet)

app_name = 'ai'

urlpatterns = [
    # API endpoints
    path('api/', include(router.urls)),
    # Direct prediction endpoint
    path('api/predict/', PredictionAPIView.as_view(), name='predict'),
    # Object detection endpoint
    path('api/detect/', ObjectDetectionAPIView.as_view(), name='detect'),
    # Test WebSocket pages
    path('test/', SegmentationWebSocketTestView.as_view(), name='test_websocket'),
    path('test-unified/', UnifiedWebSocketTestView.as_view(), name='test_websocket_unified'),
]
