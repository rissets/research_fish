# Fish Detection API Documentation

## Overview

This API provides advanced fish detection and segmentation capabilities using YOLO (You Only Look Once) deep learning models. The system supports both REST API endpoints for single predictions and WebSocket connections for real-time processing.

## Features

- **Fish Segmentation**: Precise polygon-based fish detection with detailed area analysis
- **Fish Detection**: Fast bounding box detection for single fish objects  
- **Session Management**: Track detection sessions and maintain history
- **Real-time Processing**: WebSocket support for live video streams
- **Statistics**: Comprehensive detection analytics and insights
- **Dual Model Support**: Separate models optimized for segmentation and detection tasks

## Quick Links

- **[Complete Frontend Integration Guide](./FRONTEND_INTEGRATION_GUIDE.md)** - Comprehensive guide with code examples
- **[Quick Reference](./API_QUICK_REFERENCE.md)** - Fast lookup for developers
- **Test Endpoints**: 
  - WebSocket Test: `/ai/test/`
  - Unified WebSocket Test: `/ai/test-unified/`

## API Architecture

### REST Endpoints
- **Base URL**: `/ai/api/`
- **Authentication**: Currently open (AllowAny) - implement authentication for production
- **Data Format**: JSON with base64 encoded images
- **Response Format**: Structured JSON with detection results

### WebSocket Endpoints  
- **Base URL**: `ws://your-domain/ws/ai/`
- **Connection**: Session-based with unique session IDs
- **Real-time**: Instant processing and response
- **Bidirectional**: Support for mode switching and live updates

## Model Information

### Segmentation Model
- **Type**: YOLOv8n-seg (Segmentation)
- **Purpose**: Detailed fish analysis with polygon boundaries
- **Output**: Bounding boxes + precise segmentation masks
- **Use Case**: Quality analysis, area measurement, detailed inspection

### Detection Model  
- **Type**: YOLOv8n (Custom Trained)
- **Purpose**: Fast fish detection with bounding boxes
- **Output**: Single object detection with highest confidence
- **Use Case**: Real-time counting, quick identification, live streams

## Data Models

### Core Entities
1. **DetectionSession** - Groups related detections
2. **Detection** - Individual frame analysis
3. **DetectedObject** - Specific fish detection with coordinates

### Detection Results Structure
```json
{
  "object_id": 0,
  "prediction": "fish_species_name", 
  "class_id": 0,
  "confidence": 0.85,
  "segmentation": {
    "area_pixels": 15420,
    "area_percentage": 5.02,
    "polygons": [[x1,y1,x2,y2,...]],
    "bounding_coordinates_pixel": [100,150,300,280]
  },
  "bounding_box": {
    "coordinates_pixel": [100,150,300,280],
    "area_percentage": 8.47
  }
}
```

## Integration Examples

### Simple JavaScript Detection
```javascript
// Basic fish detection
async function detectFish(imageFile) {
    const base64 = await fileToBase64(imageFile);
    
    const response = await fetch('/ai/api/predict/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64 })
    });
    
    const result = await response.json();
    return result.success ? result.result : null;
}
```

### WebSocket Real-time Processing
```javascript
// Connect to real-time detection
const socket = new WebSocket(`ws://localhost:8000/ws/ai/unified/session-123/`);

socket.send(JSON.stringify({
    type: 'predict_segmentation',
    image: base64ImageData
}));

socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'prediction_result') {
        displayDetections(data.result.detections);
    }
};
```

## Performance Characteristics

### Response Times
- **REST API**: ~200-500ms per image (depending on size)
- **WebSocket**: ~100-300ms per frame (optimized for real-time)
- **Model Loading**: ~2-5 seconds on first request

### Throughput
- **Concurrent Requests**: Supports multiple simultaneous sessions
- **Image Size**: Optimized for 640x480 to 1920x1080 images  
- **Batch Processing**: Single image per request (use WebSocket for continuous)

### Resource Usage
- **Memory**: ~1-2GB for model loading
- **CPU**: Efficient inference on CPU or GPU
- **Storage**: Optional database persistence for results

## Error Handling

### Common Error Responses
```json
{
  "success": false,
  "error": "No image data provided"
}
```

### HTTP Status Codes
- `200`: Success
- `400`: Bad Request (invalid image data)
- `500`: Internal Server Error (processing failed)

## Security Considerations

### Current Implementation
- Open API access (AllowAny permissions)
- No rate limiting implemented
- Base64 image processing

### Production Recommendations
1. Implement authentication/authorization
2. Add rate limiting per user/IP
3. Validate image formats and sizes
4. Add request logging and monitoring
5. Use HTTPS for all communications
6. Implement proper session management

## Deployment

### Requirements
- Python 3.8+
- Django 4.0+
- Django Channels (WebSocket support)
- PyTorch + Ultralytics YOLO
- Sufficient GPU/CPU resources

### Configuration
- Model paths configurable in settings
- Database backend configurable
- WebSocket routing in ASGI application

## Monitoring and Analytics

### Available Statistics
- Total objects detected
- Unique species predictions  
- Average confidence scores
- Top detected species
- Session-based analytics

### Performance Metrics
- Processing times per request
- Model accuracy scores
- Error rates and types
- Resource utilization

## Support and Troubleshooting

### Common Issues
1. **Image Upload Fails**: Check image format (JPEG/PNG) and size
2. **WebSocket Disconnects**: Implement reconnection logic
3. **Slow Processing**: Consider image resizing before upload
4. **Memory Issues**: Monitor model memory usage

### Debug Mode
Enable detailed logging for development and troubleshooting.

### Contact
For technical support and feature requests, refer to the development team or project repository.

---

**Version**: 1.0  
**Last Updated**: September 2025  
**API Stability**: Stable (v1)