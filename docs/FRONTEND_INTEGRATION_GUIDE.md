# Fish Detection API - Frontend Integration Guide

## Overview

This comprehensive guide provides frontend developers with all the necessary information to integrate with the Fish Detection API system. The API supports both REST endpoints and real-time WebSocket connections for fish detection and segmentation using YOLO models.

## Table of Contents

1. [API Base Information](#api-base-information)
2. [Authentication](#authentication)
3. [REST API Endpoints](#rest-api-endpoints)
4. [WebSocket Integration](#websocket-integration)
5. [Data Models](#data-models)
6. [Code Examples](#code-examples)
7. [Error Handling](#error-handling)
8. [Rate Limiting](#rate-limiting)
9. [Best Practices](#best-practices)

## API Base Information

### Base URLs
- **REST API**: `http://your-domain.com/ai/api/`
- **WebSocket**: `ws://your-domain.com/ws/ai/`

### Supported Features
- **Fish Segmentation**: Precise fish detection with polygon segmentation
- **Fish Detection**: Bounding box detection for single fish objects
- **Session Management**: Track detection sessions and history
- **Real-time Processing**: WebSocket support for live video streams
- **Statistics**: Detection analytics and insights

## Authentication

Currently, the API uses `AllowAny` permissions for easy integration. In production, implement proper authentication:

```javascript
// Add authentication headers when implemented
const headers = {
    'Content-Type': 'application/json',
    // 'Authorization': 'Bearer your-token-here'
};
```

## REST API Endpoints

### 1. Detection Sessions

#### Create New Session
```http
POST /ai/api/sessions/
Content-Type: application/json

{
    "session_id": "optional-custom-id"
}
```

**Response:**
```json
{
    "id": 1,
    "session_id": "uuid-generated-id",
    "created_at": "2025-09-13T10:30:00Z",
    "is_active": true,
    "user": null
}
```

#### Get All Sessions
```http
GET /ai/api/sessions/
```

#### Get Session Details
```http
GET /ai/api/sessions/{session_id}/
```

#### Get Session Detections
```http
GET /ai/api/sessions/{session_id}/detections/
```

#### Close Session
```http
POST /ai/api/sessions/{session_id}/close/
```

### 2. Fish Segmentation API

#### Predict with Segmentation
```http
POST /ai/api/predict/
Content-Type: application/json

{
    "image": "base64_encoded_image_data",
    "save_to_db": true,
    "session_id": "optional-session-id"
}
```

**Response:**
```json
{
    "success": true,
    "result": {
        "frame_number": 1,
        "frame_size": [640, 480],
        "detections": [
            {
                "object_id": 0,
                "prediction": "fish",
                "class_id": 0,
                "confidence": 0.85,
                "segmentation": {
                    "area_pixels": 15420,
                    "area_percentage": 5.02,
                    "polygons": [[x1, y1, x2, y2, ...]],
                    "bounding_coordinates_pixel": [100, 150, 300, 280],
                    "yolo_format": [0, 0.3125, 0.4479, 0.3125, 0.2708],
                    "size_pixels": [200, 130]
                },
                "bounding_box": {
                    "area_pixels": 26000,
                    "area_percentage": 8.47,
                    "coordinates_pixel": [100, 150, 300, 280],
                    "yolo_format": [0, 0.3125, 0.4479, 0.3125, 0.2708],
                    "size_pixels": [200, 130]
                },
                "segmentation_bbox_ratio": 0.593
            }
        ],
        "detection_id": 123,
        "session_id": "session-uuid"
    }
}
```

### 3. Fish Detection API

#### Object Detection (Single Fish)
```http
POST /ai/api/detect/
Content-Type: application/json

{
    "image": "base64_encoded_image_data",
    "save_to_db": true,
    "session_id": "optional-session-id"
}
```

**Response:**
```json
{
    "success": true,
    "result": {
        "frame_number": 1,
        "frame_size": [640, 480],
        "detection": {
            "object_id": 0,
            "prediction": "fish",
            "class_id": 0,
            "confidence": 0.92,
            "bounding_box": {
                "area_pixels": 26000,
                "area_percentage": 8.47,
                "coordinates_pixel": [100, 150, 300, 280],
                "yolo_format": [0, 0.3125, 0.4479, 0.3125, 0.2708],
                "size_pixels": [200, 130]
            }
        },
        "detection_id": 124,
        "session_id": "session-uuid"
    }
}
```

### 4. Statistics API

#### Get Detection Statistics
```http
GET /ai/api/objects/statistics/
```

**Response:**
```json
{
    "total_objects": 1250,
    "unique_predictions": 15,
    "average_confidence": 0.8234,
    "top_predictions": [
        {"prediction": "fish", "count": 800},
        {"prediction": "tuna", "count": 200},
        {"prediction": "salmon", "count": 150}
    ]
}
```

### 5. Model Information

#### Get Segmentation Model Info
```http
GET /ai/api/predict/model_info/
```

#### Get Detection Model Info
```http
GET /ai/api/detect/model_info/
```

## WebSocket Integration

### Connection

Connect to the unified WebSocket endpoint:

```javascript
const sessionId = 'your-session-id';
const socket = new WebSocket(`ws://your-domain.com/ws/ai/unified/${sessionId}/`);

socket.onopen = function(event) {
    console.log('Connected to AI Prediction Service');
};

socket.onmessage = function(event) {
    const data = JSON.parse(event.data);
    handleMessage(data);
};
```

### Message Types

#### 1. Connection Established
```json
{
    "type": "connection_established",
    "session_id": "uuid",
    "current_mode": "segmentation",
    "available_modes": ["segmentation", "detection"],
    "message": "Connected to Unified YOLO Prediction Service"
}
```

#### 2. Send Segmentation Request
```javascript
socket.send(JSON.stringify({
    type: 'predict_segmentation',
    image: base64ImageData,
    save_to_db: true
}));
```

#### 3. Send Detection Request
```javascript
socket.send(JSON.stringify({
    type: 'predict_object_detection',
    image: base64ImageData,
    save_to_db: true
}));
```

#### 4. Change Mode
```javascript
socket.send(JSON.stringify({
    type: 'change_mode',
    mode: 'detection' // or 'segmentation'
}));
```

#### 5. Ping/Pong
```javascript
socket.send(JSON.stringify({
    type: 'ping'
}));
```

### WebSocket Response Types

#### Prediction Result
```json
{
    "type": "prediction_result",
    "mode": "segmentation",
    "session_id": "uuid",
    "frame_number": 1,
    "processing_time": 0.234,
    "result": {
        // Same structure as REST API response
    }
}
```

#### Error Response
```json
{
    "type": "error",
    "message": "Error description"
}
```

## Data Models

### DetectionSession
```typescript
interface DetectionSession {
    id: number;
    session_id: string;
    created_at: string;
    updated_at: string;
    is_active: boolean;
    user: number | null;
}
```

### Detection
```typescript
interface Detection {
    id: number;
    session: number;
    frame_number: number;
    frame_width: number;
    frame_height: number;
    timestamp: string;
}
```

### DetectedObject
```typescript
interface DetectedObject {
    id: number;
    detection: number;
    object_id: number;
    prediction: string;
    class_id: number;
    confidence: number;
    detection_type: 'segmentation' | 'detection';
    
    // Segmentation data (optional)
    segmentation_area_pixels?: number;
    segmentation_area_percentage?: number;
    segmentation_polygons?: number[][];
    segmentation_bounding_coordinates?: number[];
    segmentation_yolo_format?: number[];
    segmentation_size?: number[];
    
    // Bounding box data
    bbox_area_pixels: number;
    bbox_area_percentage: number;
    bbox_coordinates: number[];
    bbox_yolo_format: number[];
    bbox_size: number[];
    
    // Ratio (optional)
    segmentation_bbox_ratio?: number;
    created_at: string;
}
```

## Code Examples

### React.js Integration

#### Basic Image Upload Component
```jsx
import React, { useState } from 'react';

const FishDetector = () => {
    const [image, setImage] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleImageUpload = (event) => {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onloadend = () => {
                setImage(reader.result);
            };
            reader.readAsDataURL(file);
        }
    };

    const detectFish = async () => {
        if (!image) return;

        setLoading(true);
        try {
            const base64Data = image.split(',')[1]; // Remove data:image/jpeg;base64,
            
            const response = await fetch('/ai/api/predict/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: base64Data,
                    save_to_db: true
                })
            });

            const data = await response.json();
            if (data.success) {
                setResult(data.result);
            }
        } catch (error) {
            console.error('Detection error:', error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <input type="file" accept="image/*" onChange={handleImageUpload} />
            <button onClick={detectFish} disabled={!image || loading}>
                {loading ? 'Detecting...' : 'Detect Fish'}
            </button>
            
            {result && (
                <div>
                    <h3>Detection Results:</h3>
                    <p>Detected {result.detections.length} fish</p>
                    {result.detections.map((detection, index) => (
                        <div key={index}>
                            <p>{detection.prediction} - {(detection.confidence * 100).toFixed(1)}%</p>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};
```

#### WebSocket Live Detection
```jsx
import React, { useRef, useEffect, useState } from 'react';

const LiveFishDetection = () => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const socketRef = useRef(null);
    const [isConnected, setIsConnected] = useState(false);
    const [detections, setDetections] = useState([]);

    useEffect(() => {
        // Initialize WebSocket
        const sessionId = generateSessionId();
        socketRef.current = new WebSocket(`ws://localhost:8000/ws/ai/unified/${sessionId}/`);

        socketRef.current.onopen = () => {
            setIsConnected(true);
            console.log('Connected to detection service');
        };

        socketRef.current.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'prediction_result') {
                setDetections(data.result.detections || []);
                drawDetections(data.result.detections || []);
            }
        };

        // Initialize camera
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                videoRef.current.srcObject = stream;
            });

        return () => {
            if (socketRef.current) {
                socketRef.current.close();
            }
        };
    }, []);

    const captureAndDetect = () => {
        if (!isConnected) return;

        const canvas = canvasRef.current;
        const video = videoRef.current;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const context = canvas.getContext('2d');
        context.drawImage(video, 0, 0);

        // Convert to base64
        const base64Data = canvas.toDataURL('image/jpeg').split(',')[1];

        // Send for detection
        socketRef.current.send(JSON.stringify({
            type: 'predict_segmentation',
            image: base64Data,
            save_to_db: false
        }));
    };

    const drawDetections = (detections) => {
        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');
        context.clearRect(0, 0, canvas.width, canvas.height);

        detections.forEach(detection => {
            const [x1, y1, x2, y2] = detection.bounding_box.coordinates_pixel;
            
            // Draw bounding box
            context.strokeStyle = '#00ff00';
            context.lineWidth = 2;
            context.strokeRect(x1, y1, x2 - x1, y2 - y1);
            
            // Draw label
            context.fillStyle = '#00ff00';
            context.font = '16px Arial';
            context.fillText(
                `${detection.prediction} ${(detection.confidence * 100).toFixed(1)}%`,
                x1, y1 - 5
            );
        });
    };

    return (
        <div>
            <video ref={videoRef} autoPlay muted style={{ width: '100%' }} />
            <canvas ref={canvasRef} style={{ position: 'absolute', top: 0, left: 0 }} />
            <button onClick={captureAndDetect} disabled={!isConnected}>
                Detect Fish
            </button>
            <p>Status: {isConnected ? 'Connected' : 'Disconnected'}</p>
        </div>
    );
};

const generateSessionId = () => {
    return 'session-' + Math.random().toString(36).substr(2, 9);
};
```

### Vue.js Integration

```vue
<template>
  <div class="fish-detector">
    <div class="upload-area">
      <input 
        type="file" 
        @change="handleFileUpload" 
        accept="image/*"
        ref="fileInput"
      />
      <button @click="detectFish" :disabled="!selectedImage || loading">
        {{ loading ? 'Detecting...' : 'Detect Fish' }}
      </button>
    </div>

    <div v-if="selectedImage" class="image-preview">
      <img :src="selectedImage" alt="Selected image" />
    </div>

    <div v-if="detectionResult" class="results">
      <h3>Detection Results</h3>
      <div v-for="(detection, index) in detectionResult.detections" :key="index">
        <p>{{ detection.prediction }} - {{ (detection.confidence * 100).toFixed(1) }}%</p>
        <p>Area: {{ detection.segmentation?.area_percentage?.toFixed(2) }}% of image</p>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'FishDetector',
  data() {
    return {
      selectedImage: null,
      detectionResult: null,
      loading: false,
      sessionId: null
    };
  },
  
  async mounted() {
    await this.createSession();
  },

  methods: {
    async createSession() {
      try {
        const response = await fetch('/ai/api/sessions/', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' }
        });
        const session = await response.json();
        this.sessionId = session.session_id;
      } catch (error) {
        console.error('Error creating session:', error);
      }
    },

    handleFileUpload(event) {
      const file = event.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          this.selectedImage = e.target.result;
        };
        reader.readAsDataURL(file);
      }
    },

    async detectFish() {
      if (!this.selectedImage) return;

      this.loading = true;
      try {
        const base64Data = this.selectedImage.split(',')[1];
        
        const response = await fetch('/ai/api/predict/', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image: base64Data,
            save_to_db: true,
            session_id: this.sessionId
          })
        });

        const data = await response.json();
        if (data.success) {
          this.detectionResult = data.result;
        }
      } catch (error) {
        console.error('Detection error:', error);
      } finally {
        this.loading = false;
      }
    }
  }
};
</script>
```

### JavaScript Utilities

#### Image Processing Utilities
```javascript
// Convert file to base64
function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            const base64 = reader.result.split(',')[1];
            resolve(base64);
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

// Resize image before sending
function resizeImage(file, maxWidth = 800, maxHeight = 600, quality = 0.8) {
    return new Promise((resolve) => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();
        
        img.onload = () => {
            const ratio = Math.min(maxWidth / img.width, maxHeight / img.height);
            canvas.width = img.width * ratio;
            canvas.height = img.height * ratio;
            
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            canvas.toBlob(resolve, 'image/jpeg', quality);
        };
        
        img.src = URL.createObjectURL(file);
    });
}

// API Client Class
class FishDetectionAPI {
    constructor(baseUrl = '/ai/api') {
        this.baseUrl = baseUrl;
        this.sessionId = null;
    }

    async createSession() {
        const response = await fetch(`${this.baseUrl}/sessions/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const session = await response.json();
        this.sessionId = session.session_id;
        return session;
    }

    async predictSegmentation(imageFile, saveToDb = true) {
        const base64 = await fileToBase64(imageFile);
        
        const response = await fetch(`${this.baseUrl}/predict/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: base64,
                save_to_db: saveToDb,
                session_id: this.sessionId
            })
        });

        return await response.json();
    }

    async detectObjects(imageFile, saveToDb = true) {
        const base64 = await fileToBase64(imageFile);
        
        const response = await fetch(`${this.baseUrl}/detect/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: base64,
                save_to_db: saveToDb,
                session_id: this.sessionId
            })
        });

        return await response.json();
    }

    async getSessionDetections() {
        if (!this.sessionId) return [];
        
        const response = await fetch(`${this.baseUrl}/sessions/${this.sessionId}/detections/`);
        return await response.json();
    }

    async getStatistics() {
        const response = await fetch(`${this.baseUrl}/objects/statistics/`);
        return await response.json();
    }
}
```

## Error Handling

### Common Error Responses

```javascript
// Handle API errors
function handleAPIError(response) {
    if (!response.success) {
        switch (response.error) {
            case 'No image data provided':
                alert('Please select an image first');
                break;
            case 'Failed to process image':
                alert('Image processing failed. Please try a different image.');
                break;
            default:
                alert(`Error: ${response.error}`);
        }
    }
}

// WebSocket error handling
socket.onerror = function(error) {
    console.error('WebSocket error:', error);
    setConnectionStatus('error');
};

socket.onclose = function(event) {
    if (event.wasClean) {
        console.log('Connection closed cleanly');
    } else {
        console.error('Connection died');
        // Attempt reconnection
        setTimeout(connectWebSocket, 3000);
    }
};
```

## Rate Limiting

Currently no rate limiting is implemented. For production:

```javascript
// Implement client-side rate limiting
class RateLimiter {
    constructor(maxRequests = 10, timeWindow = 60000) {
        this.maxRequests = maxRequests;
        this.timeWindow = timeWindow;
        this.requests = [];
    }

    canMakeRequest() {
        const now = Date.now();
        this.requests = this.requests.filter(time => now - time < this.timeWindow);
        return this.requests.length < this.maxRequests;
    }

    recordRequest() {
        this.requests.push(Date.now());
    }
}

const rateLimiter = new RateLimiter(10, 60000); // 10 requests per minute

async function makeAPICall() {
    if (!rateLimiter.canMakeRequest()) {
        throw new Error('Rate limit exceeded');
    }
    
    rateLimiter.recordRequest();
    // Make your API call here
}
```

## Best Practices

### 1. Image Optimization
```javascript
// Optimize images before sending
async function optimizeImage(file) {
    // Resize large images
    if (file.size > 2 * 1024 * 1024) { // 2MB
        file = await resizeImage(file, 1024, 768, 0.7);
    }
    
    return file;
}
```

### 2. Session Management
```javascript
// Reuse sessions for related detections
class SessionManager {
    constructor() {
        this.currentSession = null;
        this.sessionTimeout = 30 * 60 * 1000; // 30 minutes
    }

    async getOrCreateSession() {
        if (!this.currentSession || this.isSessionExpired()) {
            this.currentSession = await this.createNewSession();
        }
        return this.currentSession;
    }

    isSessionExpired() {
        if (!this.currentSession) return true;
        return Date.now() - this.currentSession.created > this.sessionTimeout;
    }
}
```

### 3. Performance Optimization
```javascript
// Debounce rapid predictions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

const debouncedDetect = debounce(detectFish, 500);
```

### 4. Loading States
```javascript
// Proper loading state management
class LoadingManager {
    constructor() {
        this.loadingStates = new Map();
    }

    setLoading(key, isLoading) {
        this.loadingStates.set(key, isLoading);
        this.updateUI();
    }

    isLoading(key) {
        return this.loadingStates.get(key) || false;
    }

    updateUI() {
        // Update your UI based on loading states
        const anyLoading = Array.from(this.loadingStates.values()).some(Boolean);
        document.getElementById('loader').style.display = anyLoading ? 'block' : 'none';
    }
}
```

## Testing

### Unit Tests Example (Jest)
```javascript
// api.test.js
import { FishDetectionAPI } from './fishDetectionAPI';

describe('Fish Detection API', () => {
    let api;

    beforeEach(() => {
        api = new FishDetectionAPI();
        global.fetch = jest.fn();
    });

    test('should create session successfully', async () => {
        const mockSession = { session_id: 'test-123', is_active: true };
        fetch.mockResolvedValueOnce({
            json: async () => mockSession
        });

        const session = await api.createSession();
        expect(session).toEqual(mockSession);
        expect(api.sessionId).toBe('test-123');
    });

    test('should handle API errors gracefully', async () => {
        fetch.mockResolvedValueOnce({
            json: async () => ({ success: false, error: 'No image data provided' })
        });

        const result = await api.predictSegmentation(new File([], 'test.jpg'));
        expect(result.success).toBe(false);
    });
});
```

## Troubleshooting

### Common Issues

1. **WebSocket Connection Failed**
   - Check if WebSocket URL is correct
   - Verify server is running and WebSocket support is enabled
   - Check for firewall/proxy issues

2. **Large Image Upload Fails**
   - Resize images before uploading
   - Check server upload limits
   - Use image compression

3. **Base64 Encoding Issues**
   - Ensure proper base64 format (remove data URL prefix)
   - Check for invalid characters
   - Verify image format is supported

4. **Session Management Issues**
   - Check session expiration
   - Verify session ID format
   - Handle session creation errors

### Debug Mode
```javascript
// Enable debug logging
const DEBUG = true;

function log(...args) {
    if (DEBUG) {
        console.log('[Fish Detection Debug]:', ...args);
    }
}

// Use in your code
log('Sending prediction request:', { imageSize: file.size, sessionId });
```

This comprehensive guide should provide all the necessary information for frontend developers to successfully integrate with your Fish Detection API system.