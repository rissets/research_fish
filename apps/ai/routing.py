from django.urls import re_path
from . import consumers
from .consumers_unified import UnifiedPredictionConsumer
from .consumers_unified_new import UnifiedAIPredictionConsumer

websocket_urlpatterns = [
    re_path(r'ws/segmentation/(?P<session_id>\w+)/$', consumers.SegmentationConsumer.as_asgi()),
    re_path(r'ws/unified/(?P<session_id>\w+)/$', UnifiedPredictionConsumer.as_asgi()),
    re_path(r'ws/ai/(?P<session_id>\w+)/$', UnifiedAIPredictionConsumer.as_asgi()),
]
