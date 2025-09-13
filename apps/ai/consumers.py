import json
import asyncio
import logging
from datetime import datetime
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from .services import YOLOSegmentationService
from .models import DetectionSession, Detection, DetectedObject
from django.utils import timezone
import uuid

# Set up logging
logger = logging.getLogger(__name__)

class SegmentationConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer untuk YOLO segmentation"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.yolo_service = YOLOSegmentationService()
            logger.info("YOLO service initialized successfully in consumer")
        except Exception as e:
            logger.error(f"Failed to initialize YOLO service: {str(e)}")
            self.yolo_service = None
        self.session_id = None
        self.frame_count = 0
    
    async def connect(self):
        """Handle WebSocket connection"""
        try:
            logger.info("WebSocket connection attempt started")
            
            # Extract session ID from URL
            self.session_id = self.scope['url_route']['kwargs']['session_id']
            logger.info(f"Session ID from URL: {self.session_id}")
            
            # Accept the connection first
            await self.accept()
            logger.info("WebSocket connection accepted")
            
            # Initialize YOLO service
            try:
                from .services import YOLOSegmentationService
                self.yolo_service = YOLOSegmentationService()
                logger.info("YOLO service initialized successfully in consumer")
            except Exception as e:
                logger.error(f"Failed to initialize YOLO service: {str(e)}")
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'message': f'Failed to initialize YOLO service: {str(e)}'
                }))
                await self.close()
                return
            
            # Create or get detection session
            success = await self.create_session()
            if not success:
                await self.close()
                return
            
            # Add longer delay before sending confirmation to prevent race condition
            await asyncio.sleep(0.5)
            
            # Send connection confirmation
            try:
                await self.send(text_data=json.dumps({
                    'type': 'connection_established',
                    'session_id': self.session_id,
                    'message': 'WebSocket connection established and ready for processing',
                    'timestamp': datetime.now().isoformat()
                }))
                logger.info("Connection confirmation sent successfully")
                
                # Add another small delay after sending confirmation
                await asyncio.sleep(0.2)
                logger.info("Post-confirmation delay completed")
                
            except Exception as e:
                logger.error(f"Error sending connection confirmation: {str(e)}")
                await self.close()
                return
            
        except Exception as e:
            logger.error(f"Error in connect method: {str(e)}")
            await self.close()
    
    async def disconnect(self, close_code):
        """Handle WebSocket disconnection"""
        logger.warning(f"WebSocket disconnected with code: {close_code}")
        
        # Log different close codes
        if close_code == 1000:
            logger.info("Normal closure - client closed connection")
        elif close_code == 1001:
            logger.warning("Going away - client is leaving (page refresh/close)")
        elif close_code == 1006:
            logger.error("Abnormal closure - connection lost unexpectedly")
        else:
            logger.warning(f"Unexpected close code: {close_code}")
            
        if self.session_id:
            try:
                await self.close_session()
                logger.info(f"Session {self.session_id} closed successfully")
            except Exception as e:
                logger.error(f"Error closing session: {str(e)}")
    
    async def receive(self, text_data):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(text_data)
            message_type = data.get('type')
            logger.info(f"Received message type: {message_type}")
            
            if message_type == 'ping':
                # Respond to ping with pong
                await self.send(text_data=json.dumps({
                    'type': 'pong',
                    'timestamp': datetime.now().isoformat()
                }))
                logger.info("Pong sent in response to ping")
                
            elif message_type == 'process_frame':
                await self.handle_frame_prediction(data)
                
            elif message_type == 'predict_frame':
                # Handle the message type from the HTML client
                await self.handle_frame_prediction({
                    'type': 'predict_frame',
                    'image': data.get('image')
                })
                
            else:
                logger.warning(f"Unknown message type: {message_type}")
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'message': f'Unknown message type: {message_type}'
                }))
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON received: {str(e)}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': 'Invalid JSON format'
            }))
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
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
            
            # Convert numpy types in the response to avoid JSON serialization issues
            def convert_numpy_types(data):
                """Recursively convert numpy types to Python native types"""
                if isinstance(data, dict):
                    return {k: convert_numpy_types(v) for k, v in data.items()}
                elif isinstance(data, list):
                    return [convert_numpy_types(item) for item in data]
                elif hasattr(data, 'item'):  # numpy scalar
                    return data.item()
                elif hasattr(data, 'tolist'):  # numpy array
                    return data.tolist()
                else:
                    return data
            
            clean_result = convert_numpy_types(prediction_result)
            
            # Send result back to client
            await self.send(text_data=json.dumps({
                'type': 'prediction_result',
                'session_id': self.session_id,
                'timestamp': timezone.now().isoformat(),
                'result': clean_result
            }))
            
        except Exception as e:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Prediction error: {str(e)}'
            }))
    
    @database_sync_to_async
    def create_session(self):
        """Create detection session in database"""
        try:
            # Try to get existing session first
            session = DetectionSession.objects.get(session_id=self.session_id)
            session.is_active = True
            session.save()
            logger.info(f"Reactivated existing session: {self.session_id}")
            return True
        except DetectionSession.DoesNotExist:
            try:
                # Create new session if doesn't exist
                session = DetectionSession.objects.create(
                    session_id=self.session_id,
                    is_active=True
                )
                logger.info(f"Created new session: {self.session_id}")
                return True
            except Exception as e:
                logger.error(f"Error creating new session: {str(e)}")
                return False
        except Exception as e:
            logger.error(f"Error in create_session: {str(e)}")
            return False
    
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
                # Convert numpy types to Python native types to avoid JSON serialization issues
                def convert_numpy_types(data):
                    """Recursively convert numpy types to Python native types"""
                    if isinstance(data, dict):
                        return {k: convert_numpy_types(v) for k, v in data.items()}
                    elif isinstance(data, list):
                        return [convert_numpy_types(item) for item in data]
                    elif hasattr(data, 'item'):  # numpy scalar
                        return data.item()
                    elif hasattr(data, 'tolist'):  # numpy array
                        return data.tolist()
                    else:
                        return data
                
                # Convert the object data
                clean_obj_data = convert_numpy_types(obj_data)
                
                DetectedObject.objects.create(
                    detection=detection,
                    object_id=clean_obj_data['object_id'],
                    prediction=clean_obj_data['prediction'],
                    class_id=clean_obj_data['class_id'],
                    confidence=clean_obj_data['confidence'],
                    
                    # Segmentation data
                    segmentation_area_pixels=clean_obj_data['segmentation']['area_pixels'],
                    segmentation_area_percentage=clean_obj_data['segmentation']['area_percentage'],
                    segmentation_polygons=clean_obj_data['segmentation']['polygons'],
                    segmentation_bounding_coordinates=clean_obj_data['segmentation']['bounding_coordinates_pixel'],
                    segmentation_yolo_format=clean_obj_data['segmentation']['yolo_format'],
                    segmentation_size=clean_obj_data['segmentation']['size_pixels'],
                    
                    # Bounding box data
                    bbox_area_pixels=clean_obj_data['bounding_box']['area_pixels'],
                    bbox_area_percentage=clean_obj_data['bounding_box']['area_percentage'],
                    bbox_coordinates=clean_obj_data['bounding_box']['coordinates_pixel'],
                    bbox_yolo_format=clean_obj_data['bounding_box']['yolo_format'],
                    bbox_size=clean_obj_data['bounding_box']['size_pixels'],
                    
                    # Ratio
                    segmentation_bbox_ratio=clean_obj_data['segmentation_bbox_ratio']
                )
            
            return detection.id
            
        except Exception as e:
            print(f"Error saving detection: {e}")
            return None
