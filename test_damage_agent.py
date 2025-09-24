#!/usr/bin/env python3
"""
Test script to use DamageDetectionAgent with the ngrok endpoint
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.DamageDetectionAgent import create_damage_detection_agent
import json
from pathlib import Path

def test_damage_detection_agent():
    """Test the DamageDetectionAgent with ngrok endpoint"""
    
    # Create the damage detection agent
    agent = create_damage_detection_agent(
        model_url="https://b429ee36b989.ngrok-free.app",
        confidence_threshold=0.5,
        timeout=30,
        enable_logging=True
    )
    
    # Test image path
    image_path = Path("tests/front_left-18.jpeg")
    
    if not image_path.exists():
        print(f"Error: Image file not found at {image_path}")
        return
    
    print(f"Testing DamageDetectionAgent with image: {image_path}")
    print(f"Agent endpoint: {agent.detect_endpoint}")
    print(f"Confidence threshold: {agent.confidence_threshold}")
    print(f"Timeout: {agent.timeout}")
    print("\n" + "="*50)
    
    try:
        # Detect damage
        result = agent.detect_damage(str(image_path))
        
        print("\n=== DamageDetectionAgent Result ===")
        print(json.dumps(result, indent=2))
        
        # Print statistics
        stats = agent.get_statistics()
        print("\n=== Agent Statistics ===")
        print(json.dumps(stats, indent=2))
        
    except Exception as e:
        print(f"Error during detection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_damage_detection_agent()