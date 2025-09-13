from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .api_views import (
    DetectionSessionViewSet, DetectionViewSet, 
    DetectedObjectViewSet, PredictionAPIView, ObjectDetectionAPIView,
    SegmentationModelInfoView, DetectionModelInfoView
)
from .static_views import serve_test_websocket, serve_test_websocket_unified
from .views import SegmentationWebSocketTestView, UnifiedWebSocketTestView, RestAPITestView

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
    # Model info endpoints
    path('api/predict/model_info/', SegmentationModelInfoView.as_view(), name='segmentation_model_info'),
    path('api/detect/model_info/', DetectionModelInfoView.as_view(), name='detection_model_info'),
    # Test WebSocket pages
    path('test/', SegmentationWebSocketTestView.as_view(), name='test_websocket'),
    path('test-unified/', UnifiedWebSocketTestView.as_view(), name='test_websocket_unified'),
    # Test REST API page
    path('test-rest/', RestAPITestView.as_view(), name='test_rest_api'),
]
