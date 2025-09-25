"""Part Identification Agent

This agent processes segmented vehicle images and uses GPT-4o mini to identify
damaged parts with their corresponding damage information.
"""

import json
import os
import tempfile
import time
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import base64
from dotenv import load_dotenv

from .prompts.part_identification_prompt import PART_IDENTIFICATION_PROMPT

# Load environment variables
load_dotenv(override=True)


class PartIdentificationAgent:
    """Agent for identifying damaged vehicle parts using GPT-4o mini vision analysis."""
    
    # Damage type mappings with colors and abbreviations
    DAMAGE_TYPES = {
        'dent': {'color': '#FF6B6B', 'abbrev': 'DE'},
        'scratch': {'color': '#4ECDC4', 'abbrev': 'SC'},
        'crack': {'color': '#45B7D1', 'abbrev': 'CR'},
        'glass shatter': {'color': '#96CEB4', 'abbrev': 'GS'},
        'tire flat': {'color': '#FFEAA7', 'abbrev': 'TF'},
        'lamp broken': {'color': '#DDA0DD', 'abbrev': 'LB'}
    }
    
    # Manual part ID mapping (to be assigned later)
    PART_ID_MAPPING = {
        'front bumper': 'P001',
        'rear bumper': 'P002',
        'hood': 'P003',
        'trunk': 'P004',
        'tailgate': 'P004',
        'roof': 'P005',
        'front door (left)': 'P006',
        'front door (right)': 'P007',
        'rear door (left)': 'P008',
        'rear door (right)': 'P009',
        'front fender (left)': 'P010',
        'front fender (right)': 'P011',
        'rear fender (left)': 'P012',
        'rear fender (right)': 'P013',
        'side mirror (left)': 'P014',
        'side mirror (right)': 'P015',
        'front grille': 'P016',
        'rear grille': 'P017',
        'headlight (left)': 'P018',
        'headlight (right)': 'P019',
        'taillight (left)': 'P020',
        'taillight (right)': 'P021',
        'fog light (left)': 'P022',
        'fog light (right)': 'P023',
        'turn signal (left)': 'P024',
        'turn signal (right)': 'P025',
        'brake light (left)': 'P026',
        'brake light (right)': 'P027',
        'windshield': 'P028',
        'rear window': 'P029',
        'front door window (left)': 'P030',
        'front door window (right)': 'P031',
        'rear door window (left)': 'P032',
        'rear door window (right)': 'P033',
        'sunroof': 'P034',
        'front wheel (left)': 'P035',
        'front wheel (right)': 'P036',
        'rear wheel (left)': 'P037',
        'rear wheel (right)': 'P038',
        'front tire (left)': 'P039',
        'front tire (right)': 'P040',
        'rear tire (left)': 'P041',
        'rear tire (right)': 'P042',
        'license plate': 'P043',
        'antenna': 'P044',
        'running board': 'P045',
        'step bar': 'P046',
        'spare tire': 'P047',
        'unknown': 'P999'
    }
    
    def __init__(self, openai_api_key: Optional[str] = None, model: str = "gpt-4o-mini", 
                 timeout: int = 30, enable_logging: bool = True):
        """Initialize the Part Identification Agent.
        
        Args:
            openai_api_key: OpenAI API key for GPT-4o mini (if None, loads from OPENAI_API_KEY env var)
            model: OpenAI model to use (default: gpt-4o-mini)
            timeout: Request timeout in seconds
            enable_logging: Enable detailed logging
        """
        # Load API key from environment if not provided
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key must be provided either as parameter or OPENAI_API_KEY environment variable")
        self.model = model
        self.timeout = timeout
        self.enable_logging = enable_logging
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_processing_time': 0.0,
            'parts_identified': 0
        }
        
        if self.enable_logging:
            print(f"PartIdentificationAgent initialized with model: {self.model}")
    
    def identify_damaged_parts(self, image_path: str, yolo_detections: List[Dict]) -> Dict[str, Any]:
        """Main method to identify damaged parts from image and YOLO detections.
        
        Args:
            image_path: Path to the vehicle image
            yolo_detections: List of YOLO detection results
            
        Returns:
            Dictionary containing identified parts and metadata
        """
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        try:
            # Create annotated image
            annotated_image_path = self.create_annotated_image(image_path, yolo_detections)
            
            # Analyze with GPT-4o mini
            gpt_results = self.analyze_with_gpt4o_mini(annotated_image_path)
            
            # Combine results
            final_results = self.combine_results(yolo_detections, gpt_results)
            
            # Clean up temporary file
            if os.path.exists(annotated_image_path):
                os.remove(annotated_image_path)
            
            processing_time = time.time() - start_time
            self.stats['total_processing_time'] += processing_time
            self.stats['successful_requests'] += 1
            self.stats['parts_identified'] += len(final_results.get('damaged_parts', []))
            
            if self.enable_logging:
                print(f"Part identification completed in {processing_time:.2f}s")
                print(f"Identified {len(final_results.get('damaged_parts', []))} damaged parts")
            
            return final_results
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            error_msg = f"Part identification failed: {str(e)}"
            if self.enable_logging:
                print(error_msg)
            
            return {
                'success': False,
                'error': error_msg,
                'damaged_parts': [],
                'processing_time': time.time() - start_time
            }
    
    def create_annotated_image(self, image_path: str, yolo_detections: List[Dict]) -> str:
        """Create an annotated image with damage IDs and types.
        
        Args:
            image_path: Path to the original image
            yolo_detections: List of YOLO detection results
            
        Returns:
            Path to the annotated image file
        """
        # Load image
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        # Try to load a font, fallback to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 20)
            small_font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Annotate each detection
        for i, detection in enumerate(yolo_detections):
            damage_id = f"D{i+1:02d}"  # D01, D02, etc.
            damage_type = detection.get('class', 'unknown')
            bbox = detection.get('bbox', [])
            confidence = detection.get('confidence', 0.0)
            
            if len(bbox) != 4:
                continue
                
            x1, y1, x2, y2 = bbox
            
            # Get damage type info
            damage_info = self.DAMAGE_TYPES.get(damage_type, {
                'color': '#FF0000', 'abbrev': 'UK'
            })
            
            color = damage_info['color']
            abbrev = damage_info['abbrev']
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Create label text
            label_text = f"{damage_id}: {abbrev}"
            confidence_text = f"({confidence:.2f})"
            
            # Draw label background
            label_bbox = draw.textbbox((x1, y1-30), label_text, font=font)
            draw.rectangle(label_bbox, fill=color)
            
            # Draw label text
            draw.text((x1, y1-30), label_text, fill='white', font=font)
            draw.text((x1, y1-10), confidence_text, fill=color, font=small_font)
        
        # Save annotated image to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        image.save(temp_file.name, 'JPEG')
        
        if self.enable_logging:
            print(f"Created annotated image with {len(yolo_detections)} damage annotations")
        
        return temp_file.name
    
    def analyze_with_gpt4o_mini(self, image_path: str) -> List[Dict]:
        """Send annotated image to GPT-4o mini for part identification.
        
        Args:
            image_path: Path to the annotated image
            
        Returns:
            List of part identification results from GPT
        """
        # Encode image to base64
        with open(image_path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Prepare API request
        headers = {
            'Authorization': f'Bearer {self.openai_api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': self.model,
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': PART_IDENTIFICATION_PROMPT
                        },
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': f'data:image/jpeg;base64,{image_data}'
                            }
                        }
                    ]
                }
            ],
            'max_tokens': 1000,
            'temperature': 0.1
        }
        
        # Make API request
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
        
        # Parse response
        response_data = response.json()
        content = response_data['choices'][0]['message']['content']
        
        # Extract JSON from response
        try:
            # Find JSON content between ```json and ```
            if '```json' in content:
                json_start = content.find('```json') + 7
                json_end = content.find('```', json_start)
                json_content = content[json_start:json_end].strip()
            else:
                # Assume entire content is JSON
                json_content = content.strip()
            
            gpt_results = json.loads(json_content)
            
            if self.enable_logging:
                print(f"GPT-4o mini identified {len(gpt_results)} parts")
            
            return gpt_results
            
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse GPT response as JSON: {str(e)}")
    
    def combine_results(self, yolo_detections: List[Dict], gpt_results: List[Dict]) -> Dict[str, Any]:
        """Combine YOLO detections with GPT part identification results.
        
        Args:
            yolo_detections: Original YOLO detection results
            gpt_results: GPT part identification results
            
        Returns:
            Combined results with complete damage part information
        """
        # Create mapping from damage_id to GPT results
        gpt_mapping = {result['damage_id']: result for result in gpt_results}
        
        damaged_parts = []
        
        for i, detection in enumerate(yolo_detections):
            damage_id = f"D{i+1:02d}"
            
            # Get GPT analysis for this damage
            gpt_analysis = gpt_mapping.get(damage_id, {})
            
            # Get part ID from mapping
            part_name = gpt_analysis.get('part_name', 'unknown')
            part_id = self.PART_ID_MAPPING.get(part_name, 'P999')
            
            # Combine all information
            part_info = {
                'damage_id': damage_id,
                'part_name': part_name,
                'part_id': part_id,
                'damage_type': detection.get('class', 'unknown'),
                'damage_percentage': gpt_analysis.get('damage_percentage', 0),
                'bbox': detection.get('bbox', []),
                'confidence': detection.get('confidence', 0.0),
                'location_description': gpt_analysis.get('location_description', 'Location not specified')
            }
            
            damaged_parts.append(part_info)
        
        return {
            'success': True,
            'damaged_parts': damaged_parts,
            'total_parts_identified': len(damaged_parts),
            'processing_time': self.stats['total_processing_time'] / max(1, self.stats['total_requests'])
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent performance statistics.
        
        Returns:
            Dictionary containing performance metrics
        """
        success_rate = (self.stats['successful_requests'] / 
                       max(1, self.stats['total_requests'])) * 100
        
        avg_processing_time = (self.stats['total_processing_time'] / 
                              max(1, self.stats['successful_requests']))
        
        return {
            'total_requests': self.stats['total_requests'],
            'successful_requests': self.stats['successful_requests'],
            'failed_requests': self.stats['failed_requests'],
            'success_rate_percent': round(success_rate, 2),
            'total_processing_time': round(self.stats['total_processing_time'], 2),
            'average_processing_time': round(avg_processing_time, 2),
            'parts_identified': self.stats['parts_identified']
        }


def create_part_identification_agent(openai_api_key: Optional[str] = None, **kwargs) -> PartIdentificationAgent:
    """Factory function to create a PartIdentificationAgent instance.
    
    Args:
        openai_api_key: OpenAI API key for GPT-4o mini (if None, loads from OPENAI_API_KEY env var)
        **kwargs: Additional configuration options
        
    Returns:
        Configured PartIdentificationAgent instance
    """
    return PartIdentificationAgent(openai_api_key=openai_api_key, **kwargs)


# Example usage
if __name__ == "__main__":
    # Example YOLO detections
    sample_detections = [
        {
            'class': 'dent',
            'confidence': 0.85,
            'bbox': [100, 150, 200, 250]
        },
        {
            'class': 'scratch',
            'confidence': 0.72,
            'bbox': [300, 100, 450, 180]
        }
    ]
    
    # Create agent (requires OpenAI API key)
    # agent = create_part_identification_agent(
    #     openai_api_key="your-openai-api-key",
    #     enable_logging=True
    # )
    
    # Identify damaged parts
    # results = agent.identify_damaged_parts(
    #     image_path="path/to/vehicle/image.jpg",
    #     yolo_detections=sample_detections
    # )
    
    # print(json.dumps(results, indent=2))
    # print("\nAgent Statistics:")
    # print(json.dumps(agent.get_statistics(), indent=2))
    
    print("PartIdentificationAgent implementation complete!")