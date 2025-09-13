# 🐟 Fish Detection API - Frontend Integration Documentation

A comprehensive guide and resources for integrating with the Fish Detection API system powered by YOLO deep learning models.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Documentation Files](#documentation-files)
- [API Overview](#api-overview)
- [Integration Examples](#integration-examples)
- [Testing Tools](#testing-tools)
- [SDK and Libraries](#sdk-and-libraries)
- [Deployment Guide](#deployment-guide)

## 🚀 Quick Start

### 1. Test the API Immediately

Visit the interactive testing page:
```
http://your-server.com/ai/test-unified/
```

### 2. Basic REST API Call

```javascript
// Detect fish in an image
const response = await fetch('/ai/api/predict/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        image: base64ImageData,
        save_to_db: true
    })
});

const result = await response.json();
if (result.success) {
    console.log('Fish detected:', result.result.detections);
}
```

### 3. WebSocket Real-time Detection

```javascript
const socket = new WebSocket('ws://localhost:8000/ws/ai/unified/session-123/');

socket.send(JSON.stringify({
    type: 'predict_segmentation',
    image: base64ImageData
}));

socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'prediction_result') {
        console.log('Real-time fish detection:', data.result);
    }
};
```

## 📚 Documentation Files

This package includes comprehensive documentation for different use cases:

### Core Documentation

| File | Purpose | Target Audience |
|------|---------|----------------|
| **[FRONTEND_INTEGRATION_GUIDE.md](./FRONTEND_INTEGRATION_GUIDE.md)** | Complete integration guide with examples | Frontend developers |
| **[API_QUICK_REFERENCE.md](./API_QUICK_REFERENCE.md)** | Fast lookup reference | All developers |
| **[API_DOCUMENTATION.md](./API_DOCUMENTATION.md)** | API overview and architecture | Technical leads |

### Code Examples and Tools

| File | Purpose | Usage |
|------|---------|-------|
| **[fish_detection_sdk.py](./fish_detection_sdk.py)** | Python SDK for easy integration | Python developers |
| **[frontend_integration_example.html](./templates/ai/frontend_integration_example.html)** | Interactive testing interface | Testing and demos |

## 🔧 API Overview

### Supported Detection Types

1. **Fish Segmentation** (`/ai/api/predict/`)
   - Detailed polygon-based fish detection
   - Precise area measurements
   - Multiple fish detection
   - Comprehensive analysis

2. **Fish Detection** (`/ai/api/detect/`)
   - Fast bounding box detection
   - Single fish focus
   - Real-time optimized
   - Quick identification

### Core Features

- **Session Management**: Track detection history
- **Real-time WebSocket**: Live video processing
- **Statistics API**: Detection analytics
- **Dual Models**: Segmentation and detection optimized models
- **Flexible Formats**: JSON API with base64 images

## 💻 Integration Examples

### React.js Component

```jsx
import React, { useState } from 'react';

function FishDetector() {
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    const detectFish = async (imageFile) => {
        setLoading(true);
        try {
            const base64 = await fileToBase64(imageFile);
            const response = await fetch('/ai/api/predict/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: base64 })
            });
            
            const data = await response.json();
            if (data.success) {
                setResult(data.result);
            }
        } catch (error) {
            console.error('Detection failed:', error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <input 
                type="file" 
                accept="image/*"
                onChange={(e) => detectFish(e.target.files[0])}
            />
            {loading && <p>Detecting fish...</p>}
            {result && (
                <div>
                    <h3>Found {result.detections.length} fish:</h3>
                    {result.detections.map((fish, i) => (
                        <p key={i}>
                            {fish.prediction} - {(fish.confidence * 100).toFixed(1)}%
                        </p>
                    ))}
                </div>
            )}
        </div>
    );
}
```

### Python SDK Usage

```python
from fish_detection_sdk import FishDetectionClient

# Initialize client
client = FishDetectionClient("http://localhost:8000")

# Create session and detect fish
session = client.create_session()
result = client.detect_fish_segmentation("fish_image.jpg")

print(f"Detected {len(result['detections'])} fish")
for fish in result['detections']:
    print(f"- {fish['prediction']}: {fish['confidence']:.2f}")
```

### Vue.js Integration

```vue
<template>
  <div class="fish-detector">
    <input @change="handleUpload" type="file" accept="image/*">
    <div v-if="detections.length">
      <h3>Detected Fish:</h3>
      <div v-for="fish in detections" :key="fish.object_id">
        {{ fish.prediction }} - {{ (fish.confidence * 100).toFixed(1) }}%
      </div>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return { detections: [] };
  },
  methods: {
    async handleUpload(event) {
      const file = event.target.files[0];
      if (file) {
        const base64 = await this.fileToBase64(file);
        const response = await fetch('/ai/api/predict/', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: base64 })
        });
        
        const data = await response.json();
        if (data.success) {
          this.detections = data.result.detections;
        }
      }
    },
    fileToBase64(file) {
      return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result.split(',')[1]);
        reader.readAsDataURL(file);
      });
    }
  }
};
</script>
```

## 🧪 Testing Tools

### Interactive Web Interface

Access the comprehensive testing interface at:
```
http://your-server.com/ai/templates/ai/frontend_integration_example.html
```

Features:
- ✅ REST API testing with both detection modes
- ✅ WebSocket real-time connection testing
- ✅ Session management
- ✅ Statistics dashboard
- ✅ Code examples and debugging tools
- ✅ Drag & drop image upload
- ✅ Visual results display

### API Endpoints Testing

```bash
# Test segmentation endpoint
curl -X POST http://localhost:8000/ai/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image_data"}'

# Test detection endpoint  
curl -X POST http://localhost:8000/ai/api/detect/ \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image_data"}'

# Get statistics
curl http://localhost:8000/ai/api/objects/statistics/
```

## 📦 SDK and Libraries

### Python SDK

The included Python SDK (`fish_detection_sdk.py`) provides:

- **FishDetectionClient**: REST API client
- **FishDetectionWebSocket**: Real-time WebSocket client  
- **FishDetectionUtils**: Result processing utilities

```python
# Install dependencies
pip install requests websocket-client

# Basic usage
from fish_detection_sdk import FishDetectionClient
client = FishDetectionClient("http://localhost:8000")
result = client.detect_fish_segmentation("image.jpg")
```

### JavaScript Utilities

```javascript
// Utility functions for frontend integration
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

    async detectFish(imageFile, mode = 'segmentation') {
        const base64 = await this.fileToBase64(imageFile);
        const endpoint = mode === 'segmentation' ? 'predict' : 'detect';
        
        const response = await fetch(`${this.baseUrl}/${endpoint}/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: base64,
                session_id: this.sessionId,
                save_to_db: true
            })
        });
        
        return await response.json();
    }

    fileToBase64(file) {
        return new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result.split(',')[1]);
            reader.readAsDataURL(file);
        });
    }
}
```

## 🚀 Deployment Guide

### Prerequisites

- Python 3.8+
- Django 4.0+
- Django Channels for WebSocket support
- PyTorch + Ultralytics YOLO
- Sufficient GPU/CPU resources

### Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure database
python manage.py migrate

# Start development server
python manage.py runserver

# For production with WebSocket support
daphne -p 8000 core.asgi:application
```

### Production Considerations

1. **Authentication**: Implement proper API authentication
2. **Rate Limiting**: Add request rate limiting
3. **Monitoring**: Set up logging and metrics
4. **Scaling**: Consider load balancing for high traffic
5. **Security**: Use HTTPS and validate inputs
6. **Caching**: Implement result caching for performance

### Docker Deployment

```dockerfile
FROM python:3.9

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["daphne", "-p", "8000", "core.asgi:application"]
```

## 📊 Performance Guidelines

### Optimization Tips

1. **Image Size**: Resize images to 640x640 for optimal performance
2. **Batch Processing**: Use WebSocket for multiple images
3. **Caching**: Cache model results when possible
4. **Error Handling**: Implement proper retry logic
5. **Timeouts**: Set appropriate request timeouts

### Expected Performance

- **REST API**: 200-500ms per image
- **WebSocket**: 100-300ms per frame  
- **Throughput**: 10-50 images per second (depending on hardware)
- **Memory**: ~1-2GB for model loading

## 🔧 Troubleshooting

### Common Issues

1. **Image Upload Fails**
   - Check image format (JPEG/PNG supported)
   - Verify base64 encoding is correct
   - Ensure image size is reasonable (<10MB)

2. **WebSocket Connection Fails**
   - Verify WebSocket URL format
   - Check if server supports WebSocket
   - Ensure proper session ID format

3. **Detection Takes Too Long**
   - Resize images before upload
   - Check server resources
   - Consider using detection mode instead of segmentation

4. **Poor Detection Results**
   - Ensure good image quality
   - Check if fish is clearly visible
   - Verify model compatibility

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📞 Support

### Resources

- **Documentation**: See files in this directory
- **Examples**: Check the HTML testing interface
- **SDK**: Use the Python SDK for rapid development
- **API Reference**: Review the quick reference guide

### Contributing

To contribute improvements:

1. Test your changes with the provided examples
2. Update documentation as needed
3. Ensure backward compatibility
4. Add tests for new features

---

**Last Updated**: September 2025  
**Version**: 1.0  
**License**: MIT  

For technical support, refer to the project repository or development team.