# Fish Detection API - Quick Reference

## 🚀 Quick Start

### 1. Basic Detection (REST API)
```javascript
// Upload and detect fish
const formData = new FormData();
formData.append('image', file);

fetch('/ai/api/predict/', {
    method: 'POST',
    body: JSON.stringify({
        image: base64ImageData,
        save_to_db: true
    }),
    headers: { 'Content-Type': 'application/json' }
})
.then(response => response.json())
.then(data => console.log(data.result));
```

### 2. Real-time Detection (WebSocket)
```javascript
const socket = new WebSocket('ws://localhost:8000/ws/ai/unified/session-123/');

socket.send(JSON.stringify({
    type: 'predict_segmentation',
    image: base64ImageData
}));

socket.onmessage = (event) => {
    const result = JSON.parse(event.data);
    if (result.type === 'prediction_result') {
        console.log('Detected fish:', result.result.detections);
    }
};
```

## 📋 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ai/api/predict/` | Fish segmentation |
| `POST` | `/ai/api/detect/` | Fish detection (bounding box) |
| `POST` | `/ai/api/sessions/` | Create session |
| `GET` | `/ai/api/sessions/{id}/detections/` | Get session detections |
| `GET` | `/ai/api/objects/statistics/` | Detection statistics |

## 🔌 WebSocket Messages

### Send Messages
```javascript
// Segmentation
{ type: 'predict_segmentation', image: 'base64...', save_to_db: true }

// Detection  
{ type: 'predict_object_detection', image: 'base64...', save_to_db: true }

// Change mode
{ type: 'change_mode', mode: 'detection' }

// Ping
{ type: 'ping' }
```

### Receive Messages
```javascript
// Connection established
{ type: 'connection_established', session_id: 'uuid', current_mode: 'segmentation' }

// Prediction result
{ type: 'prediction_result', result: {...}, processing_time: 0.234 }

// Error
{ type: 'error', message: 'Error description' }

// Pong
{ type: 'pong', timestamp: '2025-09-13T10:30:00Z' }
```

## 📊 Response Format

### Segmentation Response
```json
{
  "success": true,
  "result": {
    "frame_number": 1,
    "frame_size": [640, 480],
    "detections": [{
      "object_id": 0,
      "prediction": "fish",
      "confidence": 0.85,
      "segmentation": {
        "area_pixels": 15420,
        "area_percentage": 5.02,
        "polygons": [[x1, y1, x2, y2, ...]],
        "bounding_coordinates_pixel": [100, 150, 300, 280]
      },
      "bounding_box": {
        "coordinates_pixel": [100, 150, 300, 280],
        "area_percentage": 8.47
      }
    }]
  }
}
```

### Detection Response
```json
{
  "success": true,
  "result": {
    "detection": {
      "prediction": "fish",
      "confidence": 0.92,
      "bounding_box": {
        "coordinates_pixel": [100, 150, 300, 280],
        "area_percentage": 8.47
      }
    }
  }
}
```

## 🛠️ Utility Functions

### Convert File to Base64
```javascript
function fileToBase64(file) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result.split(',')[1]);
        reader.readAsDataURL(file);
    });
}
```

### Draw Detection Results
```javascript
function drawDetections(canvas, detections) {
    const ctx = canvas.getContext('2d');
    
    detections.forEach(detection => {
        const [x1, y1, x2, y2] = detection.bounding_box.coordinates_pixel;
        
        // Draw bounding box
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        
        // Draw label
        ctx.fillStyle = '#00ff00';
        ctx.font = '16px Arial';
        ctx.fillText(
            `${detection.prediction} ${(detection.confidence * 100).toFixed(1)}%`,
            x1, y1 - 5
        );
    });
}
```

### Session Management
```javascript
class SimpleSessionManager {
    constructor() {
        this.sessionId = null;
    }

    async createSession() {
        const response = await fetch('/ai/api/sessions/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const session = await response.json();
        this.sessionId = session.session_id;
        return session;
    }

    getSessionId() {
        return this.sessionId;
    }
}
```

## ⚡ React Hook Example

```jsx
import { useState, useCallback } from 'react';

function useFishDetection() {
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    const detectFish = useCallback(async (imageFile, mode = 'segmentation') => {
        setLoading(true);
        setError(null);
        
        try {
            const base64 = await fileToBase64(imageFile);
            const endpoint = mode === 'segmentation' ? '/ai/api/predict/' : '/ai/api/detect/';
            
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: base64, save_to_db: true })
            });

            const data = await response.json();
            if (data.success) {
                setResult(data.result);
            } else {
                setError(data.error);
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }, []);

    return { detectFish, loading, result, error };
}

// Usage
function FishDetectorComponent() {
    const { detectFish, loading, result, error } = useFishDetection();

    const handleFileUpload = (event) => {
        const file = event.target.files[0];
        if (file) {
            detectFish(file, 'segmentation');
        }
    };

    return (
        <div>
            <input type="file" onChange={handleFileUpload} accept="image/*" />
            {loading && <p>Detecting...</p>}
            {error && <p>Error: {error}</p>}
            {result && (
                <div>
                    <h3>Found {result.detections?.length || 0} fish</h3>
                    {result.detections?.map((fish, i) => (
                        <p key={i}>{fish.prediction} - {(fish.confidence * 100).toFixed(1)}%</p>
                    ))}
                </div>
            )}
        </div>
    );
}
```

## 🚨 Error Handling

```javascript
function handleAPIResponse(response) {
    if (!response.success) {
        switch (response.error) {
            case 'No image data provided':
                throw new Error('Please select an image');
            case 'Failed to process image':
                throw new Error('Image processing failed. Try a different image.');
            default:
                throw new Error(response.error);
        }
    }
    return response.result;
}

// Usage
try {
    const result = handleAPIResponse(apiResponse);
    console.log('Success:', result);
} catch (error) {
    console.error('Detection failed:', error.message);
}
```

## 📱 Mobile Considerations

### Camera Capture
```javascript
// Mobile camera access
function captureFromCamera() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.capture = 'environment'; // Use back camera
    input.onchange = (event) => {
        const file = event.target.files[0];
        detectFish(file);
    };
    input.click();
}
```

### Touch Events
```javascript
// Handle touch events for mobile
canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    const x = touch.clientX - rect.left;
    const y = touch.clientY - rect.top;
    
    // Handle touch at coordinates (x, y)
});
```

---

📖 **For detailed documentation, see [FRONTEND_INTEGRATION_GUIDE.md](./FRONTEND_INTEGRATION_GUIDE.md)**