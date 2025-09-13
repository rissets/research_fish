from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/segmentation/(?P<session_id>\w+)/$', consumers.SegmentationConsumer.as_asgi()),
]
