"""
Fish Detection API - Python SDK

A Python client library for easy integration with the Fish Detection API.
Supports both REST API and WebSocket connections.

Installation:
    pip install requests websocket-client

Usage:
    from fish_detection_sdk import FishDetectionClient
    
    client = FishDetectionClient("http://localhost:8000")
    result = client.detect_fish("path/to/image.jpg")
"""

import requests
import base64
import json
import uuid
from typing import Optional, Dict, Any, List
from pathlib import Path
import websocket
import threading
import time

class FishDetectionClient:
    """
    Python SDK for Fish Detection API
    
    Provides easy-to-use methods for fish detection and segmentation.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        """
        Initialize the Fish Detection client.
        
        Args:
            base_url: Base URL of the API server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/ai/api"
        self.ws_url = self.base_url.replace('http', 'ws') + "/ws/ai"
        self.timeout = timeout
        self.session_id = None
        
    def _image_to_base64(self, image_path: str) -> str:
        """Convert image file to base64 string."""
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to API endpoint."""
        url = f"{self.api_url}/{endpoint.lstrip('/')}"
        
        headers = {'Content-Type': 'application/json'}
        
        try:
            response = requests.post(url, json=data, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
    
    def create_session(self, custom_session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new detection session.
        
        Args:
            custom_session_id: Optional custom session ID
            
        Returns:
            Session information
        """
        data = {}
        if custom_session_id:
            data['session_id'] = custom_session_id
            
        try:
            response = requests.post(
                f"{self.api_url}/sessions/",
                json=data,
                headers={'Content-Type': 'application/json'},
                timeout=self.timeout
            )
            response.raise_for_status()
            session = response.json()
            self.session_id = session['session_id']
            return session
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to create session: {str(e)}")
    
    def detect_fish_segmentation(self, image_path: str, save_to_db: bool = True) -> Dict[str, Any]:
        """
        Detect fish with detailed segmentation.
        
        Args:
            image_path: Path to image file
            save_to_db: Whether to save results to database
            
        Returns:
            Detection results with segmentation data
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        base64_image = self._image_to_base64(image_path)
        
        data = {
            'image': base64_image,
            'save_to_db': save_to_db
        }
        
        if self.session_id:
            data['session_id'] = self.session_id
        
        result = self._make_request('predict/', data)
        
        if not result.get('success'):
            raise Exception(f"Detection failed: {result.get('error', 'Unknown error')}")
        
        return result['result']
    
    def detect_fish_objects(self, image_path: str, save_to_db: bool = True) -> Dict[str, Any]:
        """
        Detect fish objects with bounding boxes.
        
        Args:
            image_path: Path to image file
            save_to_db: Whether to save results to database
            
        Returns:
            Detection results with bounding box data
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        base64_image = self._image_to_base64(image_path)
        
        data = {
            'image': base64_image,
            'save_to_db': save_to_db
        }
        
        if self.session_id:
            data['session_id'] = self.session_id
        
        result = self._make_request('detect/', data)
        
        if not result.get('success'):
            raise Exception(f"Detection failed: {result.get('error', 'Unknown error')}")
        
        return result['result']
    
    def get_session_detections(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all detections for a session.
        
        Args:
            session_id: Session ID (uses current session if not provided)
            
        Returns:
            List of detection records
        """
        sid = session_id or self.session_id
        if not sid:
            raise ValueError("No session ID available")
        
        try:
            response = requests.get(
                f"{self.api_url}/sessions/{sid}/detections/",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to get session detections: {str(e)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get detection statistics.
        
        Returns:
            Statistics data
        """
        try:
            response = requests.get(
                f"{self.api_url}/objects/statistics/",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to get statistics: {str(e)}")
    
    def close_session(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Close a detection session.
        
        Args:
            session_id: Session ID (uses current session if not provided)
            
        Returns:
            Close confirmation
        """
        sid = session_id or self.session_id
        if not sid:
            raise ValueError("No session ID available")
        
        try:
            response = requests.post(
                f"{self.api_url}/sessions/{sid}/close/",
                headers={'Content-Type': 'application/json'},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to close session: {str(e)}")

class FishDetectionWebSocket:
    """
    WebSocket client for real-time fish detection.
    """
    
    def __init__(self, base_url: str = "ws://localhost:8000", session_id: Optional[str] = None):
        """
        Initialize WebSocket client.
        
        Args:
            base_url: Base WebSocket URL
            session_id: Session ID (generates one if not provided)
        """
        self.base_url = base_url.rstrip('/')
        self.session_id = session_id or str(uuid.uuid4())
        self.ws = None
        self.connected = False
        self.message_handlers = {}
        
    def on_message(self, message_type: str, handler):
        """
        Register message handler for specific message type.
        
        Args:
            message_type: Type of message to handle
            handler: Callback function
        """
        self.message_handlers[message_type] = handler
    
    def connect(self):
        """Connect to WebSocket."""
        ws_url = f"{self.base_url}/ws/ai/unified/{self.session_id}/"
        
        def on_open(ws):
            self.connected = True
            print(f"Connected to WebSocket: {ws_url}")
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                message_type = data.get('type')
                
                if message_type in self.message_handlers:
                    self.message_handlers[message_type](data)
                else:
                    print(f"Unhandled message type: {message_type}")
                    
            except json.JSONDecodeError:
                print(f"Invalid JSON message: {message}")
        
        def on_error(ws, error):
            print(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            self.connected = False
            print("WebSocket connection closed")
        
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        # Start WebSocket in a separate thread
        wst = threading.Thread(target=self.ws.run_forever)
        wst.daemon = True
        wst.start()
        
        # Wait for connection
        timeout = 10
        start_time = time.time()
        while not self.connected and time.time() - start_time < timeout:
            time.sleep(0.1)
        
        if not self.connected:
            raise Exception("Failed to connect to WebSocket")
    
    def disconnect(self):
        """Disconnect from WebSocket."""
        if self.ws:
            self.ws.close()
            self.connected = False
    
    def send_message(self, message: Dict[str, Any]):
        """
        Send message to WebSocket.
        
        Args:
            message: Message data
        """
        if not self.connected:
            raise Exception("WebSocket not connected")
        
        self.ws.send(json.dumps(message))
    
    def predict_segmentation(self, image_path: str, save_to_db: bool = True):
        """
        Send image for segmentation prediction.
        
        Args:
            image_path: Path to image file
            save_to_db: Whether to save to database
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        with open(image_path, 'rb') as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')
        
        message = {
            'type': 'predict_segmentation',
            'image': base64_image,
            'save_to_db': save_to_db
        }
        
        self.send_message(message)
    
    def predict_detection(self, image_path: str, save_to_db: bool = True):
        """
        Send image for object detection.
        
        Args:
            image_path: Path to image file
            save_to_db: Whether to save to database
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        with open(image_path, 'rb') as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')
        
        message = {
            'type': 'predict_object_detection',
            'image': base64_image,
            'save_to_db': save_to_db
        }
        
        self.send_message(message)
    
    def change_mode(self, mode: str):
        """
        Change detection mode.
        
        Args:
            mode: 'segmentation' or 'detection'
        """
        if mode not in ['segmentation', 'detection']:
            raise ValueError("Mode must be 'segmentation' or 'detection'")
        
        message = {
            'type': 'change_mode',
            'mode': mode
        }
        
        self.send_message(message)
    
    def ping(self):
        """Send ping message."""
        self.send_message({'type': 'ping'})

# Example usage and utilities
class FishDetectionUtils:
    """Utility functions for fish detection results."""
    
    @staticmethod
    def extract_fish_info(detection_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract simplified fish information from detection result.
        
        Args:
            detection_result: Detection result from API
            
        Returns:
            List of simplified fish data
        """
        fish_list = []
        
        # Handle segmentation results
        if 'detections' in detection_result:
            for detection in detection_result['detections']:
                fish_info = {
                    'species': detection['prediction'],
                    'confidence': detection['confidence'],
                    'bounding_box': detection['bounding_box']['coordinates_pixel'],
                    'area_percentage': detection['bounding_box']['area_percentage']
                }
                
                if 'segmentation' in detection:
                    fish_info.update({
                        'segmentation_area': detection['segmentation']['area_percentage'],
                        'polygon_count': len(detection['segmentation']['polygons']) if detection['segmentation']['polygons'] else 0
                    })
                
                fish_list.append(fish_info)
        
        # Handle single detection results
        elif 'detection' in detection_result:
            detection = detection_result['detection']
            fish_info = {
                'species': detection['prediction'],
                'confidence': detection['confidence'],
                'bounding_box': detection['bounding_box']['coordinates_pixel'],
                'area_percentage': detection['bounding_box']['area_percentage']
            }
            fish_list.append(fish_info)
        
        return fish_list
    
    @staticmethod
    def filter_by_confidence(fish_list: List[Dict[str, Any]], min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """
        Filter fish detections by minimum confidence.
        
        Args:
            fish_list: List of fish detection data
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered fish list
        """
        return [fish for fish in fish_list if fish['confidence'] >= min_confidence]
    
    @staticmethod
    def group_by_species(fish_list: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group fish detections by species.
        
        Args:
            fish_list: List of fish detection data
            
        Returns:
            Dictionary grouped by species
        """
        grouped = {}
        for fish in fish_list:
            species = fish['species']
            if species not in grouped:
                grouped[species] = []
            grouped[species].append(fish)
        return grouped

# Example usage functions
def example_basic_detection():
    """Example: Basic fish detection using REST API."""
    client = FishDetectionClient("http://localhost:8000")
    
    # Create session
    session = client.create_session()
    print(f"Created session: {session['session_id']}")
    
    # Detect fish in image
    result = client.detect_fish_segmentation("path/to/fish_image.jpg")
    
    # Extract fish information
    fish_list = FishDetectionUtils.extract_fish_info(result)
    high_confidence_fish = FishDetectionUtils.filter_by_confidence(fish_list, 0.8)
    
    print(f"Found {len(fish_list)} fish total")
    print(f"High confidence detections: {len(high_confidence_fish)}")
    
    for fish in high_confidence_fish:
        print(f"- {fish['species']}: {fish['confidence']:.2f}")

def example_realtime_detection():
    """Example: Real-time fish detection using WebSocket."""
    ws_client = FishDetectionWebSocket("ws://localhost:8000")
    
    # Set up message handlers
    def handle_prediction_result(data):
        result = data['result']
        fish_list = FishDetectionUtils.extract_fish_info(result)
        print(f"Real-time detection: {len(fish_list)} fish found")
        for fish in fish_list:
            print(f"  - {fish['species']}: {fish['confidence']:.2f}")
    
    def handle_connection(data):
        print(f"Connected to session: {data['session_id']}")
    
    ws_client.on_message('prediction_result', handle_prediction_result)
    ws_client.on_message('connection_established', handle_connection)
    
    # Connect and send image
    ws_client.connect()
    ws_client.predict_segmentation("path/to/fish_image.jpg")
    
    # Keep connection alive for real-time processing
    time.sleep(5)
    
    ws_client.disconnect()

if __name__ == "__main__":
    # Run examples
    print("Fish Detection SDK Examples")
    print("=" * 40)
    
    try:
        print("\n1. Basic Detection Example:")
        example_basic_detection()
        
        print("\n2. Real-time Detection Example:")
        example_realtime_detection()
        
    except Exception as e:
        print(f"Example failed: {e}")
        print("Make sure the Fish Detection API server is running on localhost:8000")