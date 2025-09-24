"""Damage Detection Agent

A deterministic agent that sends enhanced vehicle images to a fine-tuned YOLOv8 model
via HTTP requests and processes the damage detection results.
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

import requests
import base64
from PIL import Image


class DamageDetectionAgent:
    """A deterministic agent for vehicle damage detection using YOLOv8 model.
    
    This agent sends enhanced vehicle images to a fine-tuned YOLOv8 model
    hosted via ngrok and processes the detection results into a structured format.
    """
    
    def __init__(self, 
                 model_url: str,
                 confidence_threshold: float = 0.5,
                 timeout: int = 30,
                 enable_logging: bool = True):
        """Initialize the Damage Detection Agent.
        
        Args:
            model_url: The ngrok URL where YOLOv8 model is hosted
            confidence_threshold: Minimum confidence score for detections (0.0-1.0)
            timeout: HTTP request timeout in seconds
            enable_logging: Whether to enable detailed logging
        """
        self.model_url = model_url.rstrip('/')
        self.detect_endpoint = f"{self.model_url}/detect"
        self.confidence_threshold = confidence_threshold
        self.timeout = timeout
        self.enable_logging = enable_logging
        
        # Setup logging
        if self.enable_logging:
            self.logger = self._setup_logging()
        
        # Track processing statistics
        self.stats = {
            'total_requests': 0,
            'successful_detections': 0,
            'failed_requests': 0,
            'total_damages_detected': 0
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('DamageDetectionAgent')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def detect_damage(self, 
                     image_path: str, 
                     confidence_threshold: Optional[float] = None,
                     enhancement_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Detect damage in a vehicle image using YOLOv8 model.
        
        Args:
            image_path: Path to the enhanced vehicle image
            confidence_threshold: Override default confidence threshold
            enhancement_metadata: Metadata from image enhancement step
            
        Returns:
            Dictionary containing detection results and metadata
        """
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        if self.enable_logging:
            self.logger.info(f"Starting damage detection for: {image_path}")
        
        try:
            # Validate input image
            if not self._validate_image(image_path):
                return self._create_error_response(
                    image_path, 
                    "Invalid image file", 
                    enhancement_metadata
                )
            
            # Use provided threshold or default
            threshold = confidence_threshold or self.confidence_threshold
            
            # Send image to YOLOv8 model
            yolo_response = self._send_to_yolo(image_path, threshold)
            
            # Process and structure results
            result = self._process_detection_results(
                yolo_response, 
                image_path, 
                threshold,
                enhancement_metadata
            )
            
            # Update statistics
            self.stats['successful_detections'] += 1
            self.stats['total_damages_detected'] += result['total_damages']
            
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            
            if self.enable_logging:
                self.logger.info(
                    f"Detection completed: {result['total_damages']} damages found "
                    f"in {processing_time:.2f}s"
                )
            
            return result
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            
            if self.enable_logging:
                self.logger.error(f"Detection failed for {image_path}: {str(e)}")
            
            return self._create_error_response(
                image_path, 
                str(e), 
                enhancement_metadata
            )
    
    def _validate_image(self, image_path: str) -> bool:
        """Validate that the image file exists and is readable."""
        try:
            if not os.path.exists(image_path):
                return False
            
            # Try to open with PIL to verify it's a valid image
            with Image.open(image_path) as img:
                img.verify()
            
            return True
            
        except Exception:
            return False
    
    def _send_to_yolo(self, image_path: str, confidence_threshold: float) -> Dict[str, Any]:
        """Send image to YOLOv8 model via HTTP request.
        
        Args:
            image_path: Path to the image file
            confidence_threshold: Confidence threshold for detections
            
        Returns:
            Raw response from YOLOv8 model
        """
        # Convert image to base64
        with open(image_path, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Prepare payload
        payload = {
            'image': img_data,
            'confidence': confidence_threshold,
            'format': 'base64'
        }
        
        # Send HTTP request
        response = requests.post(
            self.detect_endpoint,
            json=payload,
            timeout=self.timeout,
            headers={'Content-Type': 'application/json'}
        )
        
        response.raise_for_status()
        return response.json()
    
    def _process_detection_results(self, 
                                 yolo_response: Dict[str, Any],
                                 image_path: str,
                                 confidence_threshold: float,
                                 enhancement_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Process YOLOv8 detection results into structured format.
        
        Args:
            yolo_response: Raw response from YOLOv8 model
            image_path: Path to the processed image
            confidence_threshold: Used confidence threshold
            enhancement_metadata: Metadata from image enhancement
            
        Returns:
            Structured detection results
        """
        detections = []
        raw_detections = yolo_response.get('detections', [])
        
        for i, detection in enumerate(raw_detections):
            # Calculate bounding box area
            bbox = detection.get('bbox', [0, 0, 0, 0])
            if len(bbox) >= 4:
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            else:
                area = detection.get('area', 0)
            
            processed_detection = {
                'damage_id': f"dmg_{i+1:03d}",
                'type': detection.get('class_name', 'unknown'),
                'class_id': detection.get('class_id', -1),
                'bbox': bbox,
                'confidence': detection.get('confidence', 0.0),
                'area_pixels': area
            }
            
            detections.append(processed_detection)
        
        # Create structured result
        result = {
            'image_path': image_path,
            'damage_detections': detections,
            'total_damages': len(detections),
            'detection_metadata': {
                'model_endpoint': self.detect_endpoint,
                'confidence_threshold': confidence_threshold,
                'model_processing_time': yolo_response.get('processing_time', 0),
                'image_dimensions': yolo_response.get('image_dimensions', []),
                'timestamp': datetime.now().isoformat()
            },
            'success': True
        }
        
        # Include enhancement metadata if provided
        if enhancement_metadata:
            result['enhancement_metadata'] = enhancement_metadata
        
        return result
    
    def _create_error_response(self, 
                             image_path: str, 
                             error_message: str,
                             enhancement_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Create standardized error response."""
        result = {
            'image_path': image_path,
            'damage_detections': [],
            'total_damages': 0,
            'error': error_message,
            'detection_metadata': {
                'model_endpoint': self.detect_endpoint,
                'timestamp': datetime.now().isoformat()
            },
            'success': False
        }
        
        if enhancement_metadata:
            result['enhancement_metadata'] = enhancement_metadata
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            **self.stats,
            'success_rate': (
                self.stats['successful_detections'] / max(self.stats['total_requests'], 1)
            ) * 100,
            'average_damages_per_image': (
                self.stats['total_damages_detected'] / max(self.stats['successful_detections'], 1)
            )
        }
    
    def reset_statistics(self):
        """Reset processing statistics."""
        self.stats = {
            'total_requests': 0,
            'successful_detections': 0,
            'failed_requests': 0,
            'total_damages_detected': 0
        }


def create_damage_detection_agent(model_url: str, **kwargs) -> DamageDetectionAgent:
    """Factory function to create a DamageDetectionAgent instance.
    
    Args:
        model_url: The ngrok URL where YOLOv8 model is hosted
        **kwargs: Additional arguments for agent initialization
        
    Returns:
        Configured DamageDetectionAgent instance
    """
    return DamageDetectionAgent(model_url, **kwargs)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python DamageDetectionAgent.py <ngrok_url> <image_path>")
        sys.exit(1)
    
    ngrok_url = sys.argv[1]
    image_path = sys.argv[2]
    
    # Create agent
    agent = create_damage_detection_agent(ngrok_url)
    
    # Detect damage
    result = agent.detect_damage(image_path)
    
    # Print results
    print(json.dumps(result, indent=2))
    
    # Print statistics
    print("\nStatistics:")
    print(json.dumps(agent.get_statistics(), indent=2))