from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework.views import APIView
from django.shortcuts import get_object_or_404
from .models import DetectionSession
from .serializers import DetectionSessionSerializer

class DetectionSessionViewSet(viewsets.ModelViewSet):
    """ViewSet untuk mengelola detection sessions"""
    queryset = DetectionSession.objects.all()
    serializer_class = DetectionSessionSerializer
    permission_classes = [AllowAny]

class SimpleAPIView(APIView):
    """Simple test API view"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        return Response({'message': 'API is working'})
