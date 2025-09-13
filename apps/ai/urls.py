from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .api_views import (
    DetectionSessionViewSet, DetectionViewSet, 
    DetectedObjectViewSet, PredictionAPIView
)
from .static_views import serve_test_websocket

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
    # Test WebSocket page
    path('test/', serve_test_websocket, name='test_websocket'),
]
