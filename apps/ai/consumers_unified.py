import json
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from .services import YOLOSegmentationService, YOLOObjectDetectionService
from .models import DetectionSession, Detection, DetectedObject
from django.utils import timezone
import uuid

class UnifiedPredictionConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer untuk YOLO segmentation dan object detection"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.segmentation_service = YOLOSegmentationService()
        self.detection_service = YOLOObjectDetectionService()
        self.session_id = None
        self.frame_count = 0
        self.current_mode = 'segmentation'  # default mode
    
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
            
            if message_type == 'predict_frame':
                await self.handle_frame_prediction(data)
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
                'message': 'Invalid JSON format'
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
            
            # Perform prediction based on current mode
            if self.current_mode == 'segmentation':
                prediction_result = await asyncio.get_event_loop().run_in_executor(
                    None, self.segmentation_service.predict_segmentation, base64_image, self.frame_count
                )
            else:  # detection mode
                prediction_result = await asyncio.get_event_loop().run_in_executor(
                    None, self.detection_service.predict_single_object, base64_image, self.frame_count
                )
            
            if prediction_result is None:
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'message': 'Failed to process image'
                }))
                return
            
            # Save to database based on mode
            if self.current_mode == 'segmentation' and prediction_result.get('detections'):
                detection_id = await self.save_segmentation_detection(prediction_result)
                prediction_result['detection_id'] = detection_id
            elif self.current_mode == 'detection' and prediction_result.get('detection'):
                detection_id = await self.save_object_detection(prediction_result)
                prediction_result['detection_id'] = detection_id
            
            # Convert numpy types to avoid JSON serialization issues
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
                'mode': self.current_mode,
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
    def save_segmentation_detection(self, prediction_result):
        """Save segmentation detection result to database"""
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
                # Convert numpy types to Python native types
                def convert_numpy_types(data):
                    if isinstance(data, dict):
                        return {k: convert_numpy_types(v) for k, v in data.items()}
                    elif isinstance(data, list):
                        return [convert_numpy_types(item) for item in data]
                    elif hasattr(data, 'item'):
                        return data.item()
                    elif hasattr(data, 'tolist'):
                        return data.tolist()
                    else:
                        return data
                
                clean_obj_data = convert_numpy_types(obj_data)
                
                DetectedObject.objects.create(
                    detection=detection,
                    object_id=clean_obj_data['object_id'],
                    prediction=clean_obj_data['prediction'],
                    class_id=clean_obj_data['class_id'],
                    confidence=clean_obj_data['confidence'],
                    detection_type='segmentation',
                    
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
            print(f"Error saving segmentation detection: {e}")
            return None
    
    @database_sync_to_async
    def save_object_detection(self, prediction_result):
        """Save object detection result to database"""
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
            
            # Save detected object (single object)
            obj_data = prediction_result['detection']
            
            # Convert numpy types to Python native types
            def convert_numpy_types(data):
                if isinstance(data, dict):
                    return {k: convert_numpy_types(v) for k, v in data.items()}
                elif isinstance(data, list):
                    return [convert_numpy_types(item) for item in data]
                elif hasattr(data, 'item'):
                    return data.item()
                elif hasattr(data, 'tolist'):
                    return data.tolist()
                else:
                    return data
            
            clean_obj_data = convert_numpy_types(obj_data)
            
            DetectedObject.objects.create(
                detection=detection,
                object_id=clean_obj_data['object_id'],
                prediction=clean_obj_data['prediction'],
                class_id=clean_obj_data['class_id'],
                confidence=clean_obj_data['confidence'],
                detection_type='detection',
                
                # Bounding box data only
                bbox_area_pixels=clean_obj_data['bounding_box']['area_pixels'],
                bbox_area_percentage=clean_obj_data['bounding_box']['area_percentage'],
                bbox_coordinates=clean_obj_data['bounding_box']['coordinates_pixel'],
                bbox_yolo_format=clean_obj_data['bounding_box']['yolo_format'],
                bbox_size=clean_obj_data['bounding_box']['size_pixels']
            )
            
            return detection.id
            
        except Exception as e:
            print(f"Error saving object detection: {e}")
            return None


# Keep the old SegmentationConsumer for backward compatibility
class SegmentationConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer untuk YOLO segmentation (backward compatibility)"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unified_consumer = UnifiedPredictionConsumer(*args, **kwargs)
    
    async def connect(self):
        """Handle WebSocket connection"""
        await self.unified_consumer.connect()
    
    async def disconnect(self, close_code):
        """Handle WebSocket disconnection"""
        await self.unified_consumer.disconnect(close_code)
    
    async def receive(self, text_data):
        """Handle incoming WebSocket messages"""
        await self.unified_consumer.receive(text_data)