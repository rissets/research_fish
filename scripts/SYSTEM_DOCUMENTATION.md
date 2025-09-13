# YOLO Segmentation Django System - Configuration Complete

## System Overview
Complete Django REST API + WebSocket system for real-time YOLO image segmentation with database persistence.

## System Components

### 1. Django Models (`apps/ai/models.py`)
- **DetectionSession**: Manages detection sessions with unique session IDs
- **Detection**: Stores frame-level detection results
- **DetectedObject**: Stores individual object segmentation data including:
  - Prediction class and confidence
  - Segmentation polygons and areas
  - Bounding box coordinates
  - YOLO format coordinates
  - Normalized and pixel-based measurements

### 2. REST API Endpoints (`apps/ai/api_views.py`)
- **DetectionSessionViewSet**: `/api/sessions/`
  - CRUD operations for detection sessions
  - Session statistics and management
- **DetectionViewSet**: `/api/detections/`
  - View detection results
  - Get detected objects for specific detections
- **DetectedObjectViewSet**: `/api/objects/`
  - View individual detected objects
  - Statistics and filtering capabilities
- **PredictionAPIView**: `/api/predict/`
  - Direct image prediction endpoint
  - Accepts base64 encoded images
  - Returns segmentation results

### 3. WebSocket Consumer (`apps/ai/consumers.py`)
- **SegmentationConsumer**: `ws/segmentation/{session_id}/`
  - Real-time image streaming
  - Automatic session management
  - Database persistence of results
  - JSON response with segmentation data

### 4. YOLO Service (`apps/ai/services.py`)
- **YOLOSegmentationService**: Core segmentation engine
  - YOLO v8 model integration
  - Base64 image processing
  - Polygon extraction and coordinate normalization
  - Comprehensive output formatting

## API Endpoints

### REST API Base URL: `http://localhost:8000/api/`

#### Detection Sessions
- `GET /api/sessions/` - List all sessions
- `POST /api/sessions/` - Create new session
- `GET /api/sessions/{id}/` - Get session details
- `GET /api/sessions/{id}/detections/` - Get session detections
- `POST /api/sessions/{id}/close/` - Close session

#### Detections
- `GET /api/detections/` - List all detections
- `GET /api/detections/{id}/` - Get detection details
- `GET /api/detections/{id}/detected_objects/` - Get detected objects

#### Detected Objects
- `GET /api/objects/` - List all detected objects
- `GET /api/objects/{id}/` - Get object details
- `GET /api/objects/statistics/` - Get detection statistics

#### Direct Prediction
- `POST /api/predict/` - Direct image prediction
  ```json
  {
    "image": "base64_encoded_image_data",
    "save_to_db": true,
    "session_id": "optional_session_id"
  }
  ```

### WebSocket URL: `ws://localhost:8000/ws/segmentation/{session_id}/`

#### WebSocket Protocol
1. **Connect**: Client connects with session_id
2. **Send Image**: Send JSON with base64 image data
   ```json
   {
     "image": "base64_encoded_image_data"
   }
   ```
3. **Receive Results**: Get segmentation results
   ```json
   {
     "success": true,
     "session_id": "session_123",
     "detection": {
       "id": 1,
       "frame_number": 1,
       "objects": [
         {
           "prediction": "fish",
           "confidence": 0.85,
           "segmentation_polygons": [...],
           "segmentation_area_pixels": 12500,
           "segmentation_area_percentage": 5.2
         }
       ]
     }
   }
   ```

## Database Schema

### DetectionSession
- `id`, `session_id`, `user`, `created_at`, `updated_at`, `is_active`

### Detection
- `id`, `session`, `frame_number`, `frame_width`, `frame_height`, `timestamp`

### DetectedObject
- `id`, `detection`, `object_id`, `prediction`, `class_id`, `confidence`
- `segmentation_area_pixels`, `segmentation_area_percentage`
- `segmentation_polygons`, `segmentation_bounding_coordinates`
- `segmentation_yolo_format`, `segmentation_size`
- `bbox_area_pixels`, `bbox_area_percentage`
- `bbox_coordinates`, `bbox_yolo_format`, `bbox_size`
- `segmentation_bbox_ratio`, `created_at`

## Configuration Files

### URLs Configuration
- Main URLs: `apps/core/urls.py`
- AI URLs: `apps/ai/urls.py`
- WebSocket routing: `apps/ai/routing.py`

### Django Settings
- Database: SQLite (can be changed to PostgreSQL)
- Channels: Redis backend for WebSocket
- REST Framework: JSON API with pagination

### ASGI Configuration
- HTTP: Django ASGI application
- WebSocket: Channels with authentication middleware

## Running the System

### Start Django Server
```bash
cd /mnt/arch-data/data/research_od/apps
python manage.py runserver 0.0.0.0:8000
```

### Start Redis (for WebSocket)
```bash
redis-server
```

## Testing the System

### Test REST API
```bash
# Test API availability
curl http://localhost:8000/api/sessions/

# Create new session
curl -X POST http://localhost:8000/api/sessions/ \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test_session_123"}'
```

### Test WebSocket (JavaScript example)
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/segmentation/test_session/');

ws.onopen = function(event) {
    console.log('WebSocket connected');
    
    // Send base64 image
    ws.send(JSON.stringify({
        image: 'base64_image_data_here'
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Segmentation result:', data);
};
```

## Dependencies
- Django 5.2.6
- Django REST Framework
- Django Channels
- channels-redis
- daphne (ASGI server)
- ultralytics (YOLO)
- opencv-python
- Pillow
- numpy

## Key Features
1. **Real-time Processing**: WebSocket streaming for live segmentation
2. **Comprehensive Data**: Full segmentation polygon data with coordinates
3. **Database Persistence**: All results saved to database
4. **REST API**: Full CRUD operations for all entities
5. **Session Management**: Organized by detection sessions
6. **Statistics**: Built-in analytics endpoints
7. **Flexible Input**: Supports base64 image encoding
8. **Production Ready**: ASGI server with proper error handling

## System Status: ✅ FULLY OPERATIONAL
- Database migrations applied
- All models functional
- REST API endpoints working
- WebSocket consumer ready
- YOLO service integrated
- Django server running on port 8000

The system is now ready for frontend integration and external service consumption.
