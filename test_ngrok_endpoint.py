#!/usr/bin/env python3
"""
Test script to send image to ngrok endpoint for damage detection
"""

import requests
import base64
import json
from pathlib import Path

def encode_image_to_base64(image_path):
    """Encode image file to base64 string"""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_ngrok_endpoint():
    """Test the ngrok endpoint with the available image"""
    # Configuration
    ngrok_url = "https://b429ee36b989.ngrok-free.app"
    image_path = Path("tests/front_left-18.jpeg")
    
    if not image_path.exists():
        print(f"Error: Image file not found at {image_path}")
        return
    
    try:
        # Encode image to base64
        print(f"Encoding image: {image_path}")
        image_base64 = encode_image_to_base64(image_path)
        
        # Prepare request payload
        payload = {
            "image": image_base64,
            "filename": image_path.name
        }
        
        # Set headers to bypass ngrok warning
        headers = {
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true",
            "User-Agent": "DamageDetectionAgent/1.0"
        }
        
        print(f"Sending request to: {ngrok_url}")
        print(f"Image size: {len(image_base64)} characters (base64)")
        
        # Send POST request
        response = requests.post(
            f"{ngrok_url}/detect",
            json=payload,
            headers=headers,
            timeout=30
        )
        
        print(f"Response Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n=== Detection Result ===")
            print(json.dumps(result, indent=2))
        else:
            print(f"Error Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    test_ngrok_endpoint()