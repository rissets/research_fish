import json
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from .services import YOLOSegmentationService
from .models import DetectionSession, Detection, DetectedObject
from django.utils import timezone
import uuid

class SegmentationConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer untuk YOLO segmentation"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.yolo_service = YOLOSegmentationService()
        self.session_id = None
        self.frame_count = 0
    
    async def connect(self):
        """Handle WebSocket connection"""
        await self.accept()
        
        # Generate session ID
        self.session_id = str(uuid.uuid4())
        
        # Create detection session
        await self.create_session()
        
        # Send connection confirmation
        await self.send(text_data=json.dumps({
            'type': 'connection_established',
            'session_id': self.session_id,
            'message': 'Connected to YOLO Segmentation Service'
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
            
            if message_type == 'predict_frame':
                await self.handle_frame_prediction(data)
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
                'message': 'Invalid JSON format'
            }))
        except Exception as e:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Error processing message: {str(e)}'
            }))
    
    async def handle_frame_prediction(self, data):
        """Handle frame prediction request"""
        try:
            base64_image = data.get('image')
            if not base64_image:
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'message': 'No image data provided'
                }))
                return
            
            self.frame_count += 1
            
            # Perform prediction
            prediction_result = await asyncio.get_event_loop().run_in_executor(
                None, self.yolo_service.predict_segmentation, base64_image, self.frame_count
            )
            
            if prediction_result is None:
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'message': 'Failed to process image'
                }))
                return
            
            # Save to database if detections found
            if prediction_result['detections']:
                detection_id = await self.save_detection(prediction_result)
                prediction_result['detection_id'] = detection_id
            
            # Send result back to client
            await self.send(text_data=json.dumps({
                'type': 'prediction_result',
                'session_id': self.session_id,
                'timestamp': timezone.now().isoformat(),
                'result': prediction_result
            }))
            
        except Exception as e:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Prediction error: {str(e)}'
            }))
    
    @database_sync_to_async
    def create_session(self):
        """Create detection session in database"""
        session = DetectionSession.objects.create(
            session_id=self.session_id,
            is_active=True
        )
        return session
    
    @database_sync_to_async
    def close_session(self):
        """Close detection session"""
        try:
            session = DetectionSession.objects.get(session_id=self.session_id)
            session.is_active = False
            session.save()
        except DetectionSession.DoesNotExist:
            pass
    
    @database_sync_to_async
    def save_detection(self, prediction_result):
        """Save detection result to database"""
        try:
            # Get session
            session = DetectionSession.objects.get(session_id=self.session_id)
            
            # Create detection record
            detection = Detection.objects.create(
                session=session,
                frame_number=prediction_result['frame_number'],
                frame_width=prediction_result['frame_size'][0],
                frame_height=prediction_result['frame_size'][1]
            )
            
            # Save detected objects
            for obj_data in prediction_result['detections']:
                DetectedObject.objects.create(
                    detection=detection,
                    object_id=obj_data['object_id'],
                    prediction=obj_data['prediction'],
                    class_id=obj_data['class_id'],
                    confidence=obj_data['confidence'],
                    
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
            print(f"Error saving detection: {e}")
            return None
