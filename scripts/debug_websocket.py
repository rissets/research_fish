#!/usr/bin/env python3
"""
Simple WebSocket test script untuk debug connection
"""

import asyncio
import websockets
import json
import ssl

async def test_websocket_connection():
    """Test basic WebSocket connection"""
    
    # Test URLs
    urls = [
        "ws://localhost:8001/ws/segmentation/test_session/",
        "ws://127.0.0.1:8001/ws/segmentation/test_session/"
    ]
    
    for url in urls:
        print(f"\n🔍 Testing connection to: {url}")
        
        try:
            # Test connection with timeout
            async with websockets.connect(url, ping_timeout=10, close_timeout=10) as websocket:
                print(f"✅ Connected successfully to {url}")
                
                # Send a test message
                test_message = {
                    "test": "Hello WebSocket!",
                    "timestamp": "2025-09-12T08:00:00Z"
                }
                
                print("📤 Sending test message...")
                await websocket.send(json.dumps(test_message))
                
                # Wait for response with timeout
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    print(f"📥 Received response: {response}")
                except asyncio.TimeoutError:
                    print("⏰ No response received within 5 seconds")
                
                print(f"✅ WebSocket test completed for {url}")
                
        except websockets.exceptions.InvalidURI as e:
            print(f"❌ Invalid URI: {e}")
        except websockets.exceptions.ConnectionClosed as e:
            print(f"❌ Connection closed: {e}")
        except ConnectionRefusedError as e:
            print(f"❌ Connection refused: {e}")
        except OSError as e:
            print(f"❌ OS Error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    print("🚀 WebSocket Connection Test")
    print("=" * 40)
    asyncio.run(test_websocket_connection())
