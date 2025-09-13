import json
import asyncio
import numpy as np
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from .services import YOLOSegmentationService, YOLOObjectDetectionService
from .models import DetectionSession, Detection, DetectedObject
from django.utils import timezone
import uuid

class UnifiedAIPredictionConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer untuk YOLO segmentation dan object detection"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.segmentation_service = YOLOSegmentationService()
        self.detection_service = YOLOObjectDetectionService()
        self.session_id = None
        self.frame_count = 0
        self.current_mode = 'segmentation'  # default mode
    
    def convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_numpy_types(item) for item in obj)
        else:
            return obj
    
    async def connect(self):
        """Handle WebSocket connection"""
        # Get session ID from URL
        self.session_id = self.scope['url_route']['kwargs']['session_id']
        if not self.session_id:
            self.session_id = str(uuid.uuid4())
            
        await self.accept()
        
        # Create detection session
        await self.create_session()
        
        # Send connection confirmation
        await self.send(text_data=json.dumps({
            'type': 'connection_established',
            'session_id': self.session_id,
            'current_mode': self.current_mode,
            'available_modes': ['segmentation', 'detection'],
            'message': 'Connected to Unified YOLO Prediction Service'
        }))
    
    async def disconnect(self, close_code):
        """Handle WebSocket disconnection"""
        if self.session_id:
            await self.close_session()
    
    async def receive(self, text_data):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(text_data)
            message_type = data.get('type')
            
            if message_type == 'predict_segmentation':
                await self.handle_segmentation_prediction(data)
            elif message_type == 'predict_object_detection':
                await self.handle_object_detection_prediction(data)
            elif message_type == 'change_mode':
                await self.handle_mode_change(data)
            elif message_type == 'ping':
                await self.send(text_data=json.dumps({
                    'type': 'pong',
                    'timestamp': timezone.now().isoformat()
                }))
            else:
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'message': f'Unknown message type: {message_type}'
                }))
                
        except json.JSONDecodeError:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': 'Invalid JSON data'
            }))
        except Exception as e:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Error processing message: {str(e)}'
            }))
    
    async def handle_mode_change(self, data):
        """Handle mode change request"""
        try:
            new_mode = data.get('mode')
            if new_mode in ['segmentation', 'detection']:
                self.current_mode = new_mode
                await self.send(text_data=json.dumps({
                    'type': 'mode_changed',
                    'current_mode': self.current_mode,
                    'message': f'Mode changed to {self.current_mode}'
                }))
            else:
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'message': f'Invalid mode: {new_mode}. Available modes: segmentation, detection'
                }))
        except Exception as e:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Error changing mode: {str(e)}'
            }))

    async def handle_segmentation_prediction(self, data):
        """Handle segmentation prediction request"""
        try:
            image_data = data.get('image')
            if not image_data:
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'message': 'No image data provided'
                }))
                return
            
            self.frame_count += 1
            
            # Perform segmentation prediction
            result = await asyncio.to_thread(
                self.segmentation_service.predict_segmentation, 
                image_data, 
                self.frame_count
            )
            
            if result is None:
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'message': 'Failed to process image for segmentation'
                }))
                return
            
            # Convert numpy types to Python native types BEFORE saving to database
            result_converted = self.convert_numpy_types(result)
            
            # Save to database using converted data
            detection_id = await self.save_segmentation_detection(result_converted)
            
            # Send result
            await self.send(text_data=json.dumps({
                'type': 'prediction_result',
                'mode': 'segmentation',
                'result': result_converted,
                'detection_id': detection_id,
                'session_id': self.session_id
            }))
            
        except Exception as e:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Segmentation prediction error: {str(e)}'
            }))

    async def handle_object_detection_prediction(self, data):
        """Handle object detection prediction request"""
        try:
            image_data = data.get('image')
            if not image_data:
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'message': 'No image data provided'
                }))
                return
            
            self.frame_count += 1
            
            # Perform object detection prediction
            result = await asyncio.to_thread(
                self.detection_service.predict_single_object, 
                image_data, 
                self.frame_count
            )
            
            if result is None:
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'message': 'Failed to process image for object detection'
                }))
                return
            
            # Convert numpy types to Python native types BEFORE saving to database
            result_converted = self.convert_numpy_types(result)
            
            # Save to database if detection exists using converted data
            detection_id = None
            if result_converted.get('detection'):
                detection_id = await self.save_object_detection(result_converted)
            
            # Send result
            await self.send(text_data=json.dumps({
                'type': 'prediction_result',
                'mode': 'detection',
                'result': result_converted,
                'detection_id': detection_id,
                'session_id': self.session_id
            }))
            
        except Exception as e:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Object detection prediction error: {str(e)}'
            }))

    @database_sync_to_async
    def create_session(self):
        """Create detection session in database"""
        try:
            session, created = DetectionSession.objects.get_or_create(
                session_id=self.session_id,
                defaults={'is_active': True}
            )
            return session.id
        except Exception as e:
            print(f"Error creating session: {e}")
            return None

    @database_sync_to_async
    def close_session(self):
        """Close detection session"""
        try:
            session = DetectionSession.objects.get(session_id=self.session_id)
            session.is_active = False
            session.save()
        except DetectionSession.DoesNotExist:
            pass
        except Exception as e:
            print(f"Error closing session: {e}")

    @database_sync_to_async
    def save_segmentation_detection(self, result):
        """Save segmentation detection result to database"""
        try:
            session = DetectionSession.objects.get(session_id=self.session_id)
            
            # Create detection record
            detection = Detection.objects.create(
                session=session,
                frame_number=result['frame_number'],
                frame_width=result['frame_size'][0],
                frame_height=result['frame_size'][1]
            )
            
            # Save detected objects
            for obj_data in result['detections']:
                DetectedObject.objects.create(
                    detection=detection,
                    object_id=obj_data['object_id'],
                    prediction=obj_data['prediction'],
                    class_id=obj_data['class_id'],
                    confidence=obj_data['confidence'],
                    detection_type='segmentation',
                    
                    # Segmentation data
                    segmentation_area_pixels=obj_data['segmentation']['area_pixels'],
                    segmentation_area_percentage=obj_data['segmentation']['area_percentage'],
                    segmentation_polygons=obj_data['segmentation']['polygons'],
                    segmentation_bounding_coordinates=obj_data['segmentation']['bounding_coordinates_pixel'],
                    segmentation_yolo_format=obj_data['segmentation']['yolo_format'],
                    segmentation_size=obj_data['segmentation']['size_pixels'],
                    
                    # Bounding box data
                    bbox_area_pixels=obj_data['bounding_box']['area_pixels'],
                    bbox_area_percentage=obj_data['bounding_box']['area_percentage'],
                    bbox_coordinates=obj_data['bounding_box']['coordinates_pixel'],
                    bbox_yolo_format=obj_data['bounding_box']['yolo_format'],
                    bbox_size=obj_data['bounding_box']['size_pixels'],
                    
                    # Ratio
                    segmentation_bbox_ratio=obj_data['segmentation_bbox_ratio']
                )
            
            return detection.id
            
        except Exception as e:
            print(f"Error saving segmentation detection: {e}")
            return None

    @database_sync_to_async
    def save_object_detection(self, result):
        """Save object detection result to database"""
        try:
            session = DetectionSession.objects.get(session_id=self.session_id)
            
            # Create detection record
            detection = Detection.objects.create(
                session=session,
                frame_number=result['frame_number'],
                frame_width=result['frame_size'][0],
                frame_height=result['frame_size'][1]
            )
            
            # Save detected object
            obj_data = result['detection']
            DetectedObject.objects.create(
                detection=detection,
                object_id=obj_data['object_id'],
                prediction=obj_data['prediction'],
                class_id=obj_data['class_id'],
                confidence=obj_data['confidence'],
                detection_type='detection',
                
                # Bounding box data
                bbox_area_pixels=obj_data['bounding_box']['area_pixels'],
                bbox_area_percentage=obj_data['bounding_box']['area_percentage'],
                bbox_coordinates=obj_data['bounding_box']['coordinates_pixel'],
                bbox_yolo_format=obj_data['bounding_box']['yolo_format'],
                bbox_size=obj_data['bounding_box']['size_pixels']
            )
            
            return detection.id
            
        except Exception as e:
            print(f"Error saving object detection: {e}")
            return None