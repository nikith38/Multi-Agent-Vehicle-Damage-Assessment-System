#!/usr/bin/env python3
"""
Complete Pipeline Test: Image Enhancement -> Damage Detection -> Part Identification
This script demonstrates the full workflow from enhanced image to final damage assessment.
"""

import json
from pathlib import Path
from agents import create_damage_detection_agent, create_part_identification_agent
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv(override=True)

def run_complete_pipeline():
    """
    Run the complete pipeline from image enhancement output to part identification.
    """
    print("🚀 Starting Complete Vehicle Damage Assessment Pipeline")
    print("=" * 60)
    
    # Step 1: Load image enhancement output
    enhancement_output_path = "tests/front_left-18_sharpened_output.json"
    
    if not os.path.exists(enhancement_output_path):
        print(f"❌ Enhancement output file not found: {enhancement_output_path}")
        return
    
    with open(enhancement_output_path, 'r') as f:
        enhancement_data = json.load(f)
    
    print(f"✅ Loaded image enhancement output:")
    print(f"   - Quality Score: {enhancement_data['quality_score']:.3f}")
    print(f"   - Issues Found: {', '.join(enhancement_data['issues'])}")
    print(f"   - Enhanced Image: {enhancement_data['enhanced_image']}")
    print(f"   - Processable: {enhancement_data['processable']}")
    print()
    
    if not enhancement_data['processable']:
        print("❌ Image is not processable according to enhancement output")
        return
    
    enhanced_image_path = enhancement_data['enhanced_image']
    
    # Check if enhanced image exists
    if not os.path.exists(enhanced_image_path):
        print(f"❌ Enhanced image not found: {enhanced_image_path}")
        print("   Using original image instead...")
        enhanced_image_path = "tests/front_left-18.jpeg"
    
    # Step 2: Initialize Damage Detection Agent
    print("🔍 Step 2: Initializing Damage Detection Agent")
    damage_agent = create_damage_detection_agent(
        model_url="https://8a528dc1103f.ngrok-free.app/"
    )
    print("✅ Damage Detection Agent initialized")
    print()
    
    # Step 3: Run Damage Detection
    print("🎯 Step 3: Running Damage Detection")
    print(f"   Processing image: {enhanced_image_path}")
    
    try:
        # Call the actual YOLO API via ngrok
        damage_results = damage_agent.detect_damage(enhanced_image_path)
        
        # Convert to expected format if needed
        simulated_damage_results = damage_results
        
        print(f"✅ Damage Detection completed:")
        detections = damage_results.get('damage_detections', [])
        print(f"   - Detections found: {len(detections)}")
        for i, detection in enumerate(detections):
            damage_type = detection.get('type', detection.get('class_name', 'unknown'))
            confidence = detection.get('confidence', 0.0)
            print(f"   - Detection {i+1}: {damage_type} (confidence: {confidence:.2f})")
        processing_time = damage_results.get('detection_metadata', {}).get('model_processing_time', 0)
        print(f"   - Processing time: {processing_time:.2f}s")
        
        # Convert to format expected by part identification agent
        simulated_damage_results = {
            'detections': [
                {
                    'class_id': det.get('class_id', 0),
                    'class_name': det.get('type', 'unknown'),
                    'confidence': det.get('confidence', 0.0),
                    'bbox': det.get('bbox', [0, 0, 0, 0])
                }
                for det in detections
            ],
            'image_path': enhanced_image_path,
            'processing_time': processing_time
        }
        print()
        
    except Exception as e:
        print(f"❌ Damage detection failed: {str(e)}")
        return
    
    # Step 4: Initialize Part Identification Agent
    print("🧠 Step 4: Initializing Part Identification Agent")
    
    # Check if OpenAI API key is available
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("⚠️  OpenAI API key not found in environment variables")
        print("   Part identification will run in simulation mode")
        part_agent = None
    else:
        try:
            part_agent = create_part_identification_agent()
            print("✅ Part Identification Agent initialized with OpenAI API")
        except Exception as e:
            print(f"⚠️  Failed to initialize Part Identification Agent: {str(e)}")
            print("   Running in simulation mode")
            part_agent = None
    print()
    
    # Step 5: Run Part Identification
    print("🔧 Step 5: Running Part Identification")
    
    if part_agent:
        try:
            # Real part identification with GPT-4o mini
            part_results = part_agent.identify_damaged_parts(
                enhanced_image_path, 
                simulated_damage_results['detections']
            )
            print("✅ Part Identification completed with GPT-4o mini")
        except Exception as e:
            print(f"⚠️  GPT-4o mini identification failed: {str(e)}")
            print("   Falling back to simulation mode")
            part_agent = None
    
    if not part_agent:
        # Simulated part identification results
        part_results = {
            "identified_parts": [
                {
                    "damage_id": "D01",
                    "part_name": "front bumper",
                    "part_id": "P001",
                    "damage_type": "dent",
                    "damage_percentage": 25,
                    "bbox": [150, 200, 280, 320],
                    "confidence": 0.85,
                    "location_description": "lower center section"
                },
                {
                    "damage_id": "D02",
                    "part_name": "front door (left)",
                    "part_id": "P006",
                    "damage_type": "scratch",
                    "damage_percentage": 15,
                    "bbox": [400, 150, 550, 220],
                    "confidence": 0.72,
                    "location_description": "upper panel near handle"
                },
                {
                    "damage_id": "D03",
                    "part_name": "windshield",
                    "part_id": "P028",
                    "damage_type": "crack",
                    "damage_percentage": 5,
                    "bbox": [100, 50, 200, 120],
                    "confidence": 0.68,
                    "location_description": "lower left corner"
                }
            ],
            "total_parts_identified": 3,
            "processing_time": 2.45
        }
        print("✅ Part Identification completed in simulation mode")
    
    # Handle both real and simulated result structures
    if 'identified_parts' in part_results:
        parts_list = part_results['identified_parts']
        total_parts = part_results.get('total_parts_identified', len(parts_list))
        processing_time = part_results.get('processing_time', 0)
    else:
        # Real GPT-4o mini results structure
        parts_list = part_results.get('parts', [])
        total_parts = len(parts_list)
        processing_time = part_results.get('processing_time', 0)
    
    print(f"   - Parts identified: {total_parts}")
    for part in parts_list:
        if 'damage_id' in part:  # Simulated structure
            print(f"   - {part['damage_id']}: {part['part_name']} ({part['damage_type']}, {part['damage_percentage']}% damage)")
        else:  # Real GPT-4o mini structure
            print(f"   - {part.get('part_name', 'Unknown')}: {part.get('damage_type', 'Unknown')} damage")
    print(f"   - Processing time: {processing_time:.2f}s")
    print()
    
    # Step 6: Save Annotated Image for Debugging
    print("💾 Step 6: Saving Annotated Image for Debugging")
    
    if part_agent:
        try:
            # Create annotated image using the agent
            annotated_image_path = part_agent.create_annotated_image(
                enhanced_image_path, 
                simulated_damage_results['detections']
            )
            print(f"✅ Annotated image saved: {annotated_image_path}")
        except Exception as e:
            print(f"⚠️  Failed to create annotated image with agent: {str(e)}")
            annotated_image_path = None
    else:
        # Create a simple annotated image using basic drawing (simulation)
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Load the image
            image = Image.open(enhanced_image_path)
            draw = ImageDraw.Draw(image)
            
            # Define colors for different damage types
            damage_colors = {
                "dent": "#FF6B6B",
                "scratch": "#4ECDC4", 
                "crack": "#45B7D1",
                "glass shatter": "#96CEB4",
                "tire flat": "#FFEAA7",
                "lamp broken": "#DDA0DD"
            }
            
            # Draw bounding boxes and labels
            for i, detection in enumerate(simulated_damage_results['detections']):
                bbox = detection['bbox']
                damage_type = detection['class_name']
                confidence = detection['confidence']
                
                # Get color for this damage type
                color = damage_colors.get(damage_type, "#FF0000")
                
                # Draw bounding box
                draw.rectangle(bbox, outline=color, width=3)
                
                # Create label
                label = f"D{i+1:02d}: {damage_type} ({confidence:.2f})"
                
                # Draw label background
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except:
                    font = ImageFont.load_default()
                
                text_bbox = draw.textbbox((0, 0), label, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                label_x = bbox[0]
                label_y = bbox[1] - text_height - 5
                
                draw.rectangle(
                    [label_x, label_y, label_x + text_width + 10, label_y + text_height + 5],
                    fill=color
                )
                
                # Draw label text
                draw.text((label_x + 5, label_y + 2), label, fill="white", font=font)
            
            # Save annotated image
            base_name = Path(enhanced_image_path).stem
            annotated_image_path = f"tests/{base_name}_annotated_debug.jpeg"
            image.save(annotated_image_path, "JPEG", quality=95)
            
            print(f"✅ Annotated debug image saved: {annotated_image_path}")
            
        except Exception as e:
            print(f"⚠️  Failed to create annotated debug image: {str(e)}")
            annotated_image_path = None
    
    # Step 7: Save Complete Results
    print("📊 Step 7: Saving Complete Pipeline Results")
    
    complete_results = {
        "pipeline_version": "1.0",
        "input_image": enhanced_image_path,
        "enhancement_data": enhancement_data,
        "damage_detection": simulated_damage_results,
        "part_identification": part_results,
        "annotated_image": annotated_image_path,
        "summary": {
            "total_damages_detected": len(simulated_damage_results['detections']),
            "total_parts_identified": total_parts,
            "damage_types_found": list(set([d['class_name'] for d in simulated_damage_results['detections']])),
            "parts_affected": [p.get('part_name', 'Unknown') for p in parts_list]
        }
    }
    
    results_path = "tests/complete_pipeline_results.json"
    with open(results_path, 'w') as f:
        json.dump(complete_results, f, indent=2)
    
    print(f"✅ Complete results saved: {results_path}")
    print()
    
    # Final Summary
    print("🎉 Pipeline Execution Summary")
    print("=" * 40)
    print(f"✅ Image Enhancement: Quality score {enhancement_data['quality_score']:.3f}")
    print(f"✅ Damage Detection: {len(simulated_damage_results['detections'])} damages found")
    print(f"✅ Part Identification: {total_parts} parts identified")
    if annotated_image_path:
        print(f"✅ Annotated Image: {annotated_image_path}")
    print(f"✅ Results Saved: {results_path}")
    print()
    print("🔍 Damage Summary:")
    for damage_type in complete_results['summary']['damage_types_found']:
        count = sum(1 for d in simulated_damage_results['detections'] if d['class_name'] == damage_type)
        print(f"   - {damage_type}: {count} instance(s)")
    print()
    print("🔧 Parts Affected:")
    for part in complete_results['summary']['parts_affected']:
        print(f"   - {part}")
    print()
    print("🚀 Complete pipeline execution finished successfully!")

if __name__ == "__main__":
    run_complete_pipeline()