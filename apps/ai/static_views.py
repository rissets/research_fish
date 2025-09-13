from django.http import FileResponse, Http404
from django.shortcuts import render
from django.conf import settings
import os

def serve_test_websocket(request):
    """Serve the test websocket HTML file using Django template"""
    try:
        return render(request, 'ai/test_websocket.html')
    except Exception as e:
        raise Http404(f"Error serving template: {str(e)}")

def serve_test_websocket_unified(request):
    """Serve the unified test websocket HTML file"""
    try:
        file_path = os.path.join(settings.BASE_DIR, 'static', 'test_websocket_unified.html')
        if os.path.exists(file_path):
            return FileResponse(open(file_path, 'rb'), content_type='text/html')
        else:
            raise Http404("File not found")
    except Exception as e:
        raise Http404(f"Error serving file: {str(e)}")
