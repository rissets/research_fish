#!/usr/bin/env python3
"""
Simple WebSocket client test for YOLO Segmentation System
"""

import asyncio
import websockets
import json
import base64
from PIL import Image
from io import BytesIO

def create_test_image():
    """Create a simple test image as base64"""
    img = Image.new('RGB', (640, 480), color='red')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_data = buffer.getvalue()
    return base64.b64encode(img_data).decode('utf-8')

async def test_websocket():
    """Test WebSocket connection and image sending"""
    uri = "ws://localhost:8000/ws/segmentation/test_ws_session/"
    
    try:
        print("🔌 Connecting to WebSocket...")
        async with websockets.connect(uri) as websocket:
            print("✅ Connected successfully!")
            
            # Create and send test image
            test_image = create_test_image()
            message = {
                "image": test_image
            }
            
            print("📤 Sending test image...")
            await websocket.send(json.dumps(message))
            
            # Wait for response
            print("⏳ Waiting for response...")
            response = await websocket.recv()
            data = json.loads(response)
            
            print("📥 Received response:")
            print(json.dumps(data, indent=2))
            
            if data.get('success'):
                print("✅ WebSocket test successful!")
            else:
                print("❌ WebSocket test failed")
                
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")

if __name__ == "__main__":
    print("🚀 WebSocket Test for YOLO Segmentation System")
    print("=" * 50)
    asyncio.run(test_websocket())
