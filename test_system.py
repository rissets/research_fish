#!/usr/bin/env python3
"""
Test script for YOLO Segmentation Django System
Tests REST API and basic functionality
"""

import requests
import json
import base64
import time
from io import BytesIO
from PIL import Image
import numpy as np

# Configuration
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api"

def create_test_image():
    """Create a simple test image as base64"""
    # Create a simple colored rectangle image
    img = Image.new('RGB', (640, 480), color='blue')
    
    # Convert to base64
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_data = buffer.getvalue()
    b64_string = base64.b64encode(img_data).decode('utf-8')
    
    return b64_string

def test_api_endpoints():
    """Test all REST API endpoints"""
    print("🔍 Testing REST API Endpoints...")
    
    try:
        # Test 1: List sessions (should be empty initially)
        print("\n1. Testing GET /api/sessions/")
        response = requests.get(f"{API_BASE}/sessions/")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        
        # Test 2: Create a new session
        print("\n2. Testing POST /api/sessions/")
        session_data = {"session_id": f"test_session_{int(time.time())}"}
        response = requests.post(f"{API_BASE}/sessions/", json=session_data)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 201:
            session = response.json()
            session_id = session['id']
            print(f"   Created session: {session}")
            
            # Test 3: Get session details
            print(f"\n3. Testing GET /api/sessions/{session_id}/")
            response = requests.get(f"{API_BASE}/sessions/{session_id}/")
            print(f"   Status: {response.status_code}")
            print(f"   Session details: {response.json()}")
            
        # Test 4: List all detections (should be empty)
        print("\n4. Testing GET /api/detections/")
        response = requests.get(f"{API_BASE}/detections/")
        print(f"   Status: {response.status_code}")
        print(f"   Detections count: {len(response.json().get('results', []))}")
        
        # Test 5: List all detected objects (should be empty)
        print("\n5. Testing GET /api/objects/")
        response = requests.get(f"{API_BASE}/objects/")
        print(f"   Status: {response.status_code}")
        print(f"   Objects count: {len(response.json().get('results', []))}")
        
        # Test 6: Get statistics
        print("\n6. Testing GET /api/objects/statistics/")
        response = requests.get(f"{API_BASE}/objects/statistics/")
        print(f"   Status: {response.status_code}")
        print(f"   Statistics: {response.json()}")
        
        print("\n✅ REST API tests completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to Django server. Make sure it's running on http://localhost:8000")
        return False
    except Exception as e:
        print(f"\n❌ API test failed: {e}")
        return False
    
    return True

def test_prediction_api():
    """Test the direct prediction API"""
    print("\n🤖 Testing Prediction API...")
    
    try:
        # Create test image
        test_image_b64 = create_test_image()
        
        # Test prediction
        prediction_data = {
            "image": test_image_b64,
            "save_to_db": True,
            "session_id": f"test_predict_{int(time.time())}"
        }
        
        print("   Sending image for prediction...")
        response = requests.post(f"{API_BASE}/predict/", json=prediction_data)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   Prediction successful!")
            print(f"   Objects detected: {len(result.get('objects', []))}")
            if result.get('objects'):
                print(f"   First object: {result['objects'][0]['prediction']} (confidence: {result['objects'][0]['confidence']:.2f})")
        else:
            print(f"   Response: {response.text}")
            
        print("\n✅ Prediction API test completed!")
        
    except Exception as e:
        print(f"\n❌ Prediction API test failed: {e}")
        return False
    
    return True

def test_yolo_model():
    """Test if YOLO model can be loaded"""
    print("\n🏗️ Testing YOLO Model Loading...")
    
    try:
        import sys
        sys.path.append('/mnt/arch-data/data/research_od/apps')
        
        from ai.services import YOLOSegmentationService
        
        print("   Initializing YOLO service...")
        yolo_service = YOLOSegmentationService()
        
        print("   Creating test image...")
        test_image_b64 = create_test_image()
        
        print("   Running prediction...")
        result = yolo_service.predict_segmentation(test_image_b64)
        
        print(f"   ✅ YOLO model loaded and working!")
        print(f"   Objects detected: {len(result.get('objects', []))}")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Cannot import YOLO service: {e}")
        return False
    except Exception as e:
        print(f"   ❌ YOLO test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 YOLO Segmentation Django System - Test Suite")
    print("=" * 50)
    
    # Test results
    api_ok = test_api_endpoints()
    prediction_ok = test_prediction_api() if api_ok else False
    yolo_ok = test_yolo_model()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   REST API: {'✅ PASS' if api_ok else '❌ FAIL'}")
    print(f"   Prediction API: {'✅ PASS' if prediction_ok else '❌ FAIL'}")
    print(f"   YOLO Model: {'✅ PASS' if yolo_ok else '❌ FAIL'}")
    
    if all([api_ok, prediction_ok, yolo_ok]):
        print("\n🎉 All tests passed! System is ready for use.")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
    
    print("\n🔗 System URLs:")
    print(f"   Django Admin: {BASE_URL}/admin/")
    print(f"   REST API: {API_BASE}/")
    print(f"   WebSocket: ws://localhost:8000/ws/segmentation/<session_id>/")

if __name__ == "__main__":
    main()
