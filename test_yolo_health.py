#!/usr/bin/env python3
"""
YOLO Server Health Check Script

This script pings the YOLO server to check if it's running and accessible.
It tests both the health endpoint and the detect endpoint.
"""

import os
import requests
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

def test_yolo_server():
    """Test YOLO server health and endpoints."""
    
    # Get NGROK URL from environment
    ngrok_url = os.getenv('NGROK_URL')
    
    if not ngrok_url:
        print("❌ NGROK_URL not found in environment variables")
        return False
    
    # Remove trailing slash if present
    base_url = ngrok_url.rstrip('/')
    
    print(f"🔍 Testing YOLO server at: {base_url}")
    print("=" * 50)
    
    # Test 1: Basic connectivity
    print("1️⃣ Testing basic connectivity...")
    try:
        response = requests.get(base_url, timeout=10)
        print(f"   ✅ Server responded with status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Connection failed: {e}")
        return False
    
    # Test 2: Health endpoint (common patterns)
    health_endpoints = ['/health', '/status', '/ping', '/']
    
    print("\n2️⃣ Testing health endpoints...")
    health_found = False
    
    for endpoint in health_endpoints:
        try:
            url = f"{base_url}{endpoint}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                print(f"   ✅ {endpoint}: {response.status_code} - {response.text[:100]}")
                health_found = True
            else:
                print(f"   ⚠️  {endpoint}: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ {endpoint}: {str(e)[:50]}...")
    
    # Test 3: Detect endpoint (the one we actually use)
    print("\n3️⃣ Testing detect endpoint...")
    try:
        detect_url = f"{base_url}/detect"
        
        # Try GET request first (should return method info or 405)
        response = requests.get(detect_url, timeout=5)
        print(f"   📡 GET /detect: {response.status_code}")
        
        if response.status_code == 405:
            print("   ✅ Detect endpoint exists (Method Not Allowed is expected for GET)")
        elif response.status_code == 200:
            print("   ✅ Detect endpoint accessible")
        else:
            print(f"   ⚠️  Unexpected status: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Detect endpoint failed: {e}")
        return False
    
    # Test 4: Server info
    print("\n4️⃣ Server information...")
    try:
        response = requests.get(base_url, timeout=5)
        headers = response.headers
        
        print(f"   🖥️  Server: {headers.get('Server', 'Unknown')}")
        print(f"   📅 Date: {headers.get('Date', 'Unknown')}")
        print(f"   🔧 Content-Type: {headers.get('Content-Type', 'Unknown')}")
        
    except Exception as e:
        print(f"   ⚠️  Could not get server info: {e}")
    
    print("\n" + "=" * 50)
    
    if health_found:
        print("✅ YOLO server appears to be healthy and accessible!")
        return True
    else:
        print("⚠️  YOLO server is reachable but health status unclear")
        print("   The /detect endpoint should still work for POST requests")
        return True

def test_with_sample_request():
    """Test with a sample detection request (if server supports it)."""
    
    ngrok_url = os.getenv('NGROK_URL', '').rstrip('/')
    detect_url = f"{ngrok_url}/detect"
    
    print("\n🧪 Testing sample detection request...")
    
    # Create a minimal test payload (adjust based on your YOLO server's expected format)
    test_data = {
        "image": "test_image_data",
        "confidence": 0.5
    }
    
    try:
        response = requests.post(detect_url, json=test_data, timeout=10)
        
        if response.status_code == 200:
            print("   ✅ Detection endpoint working!")
            print(f"   📄 Response: {response.text[:200]}...")
        elif response.status_code == 400:
            print("   ⚠️  Bad request (expected with test data)")
            print("   ✅ But endpoint is responding to POST requests")
        else:
            print(f"   ⚠️  Status: {response.status_code}")
            print(f"   📄 Response: {response.text[:100]}...")
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Sample request failed: {e}")

def main():
    """Main function to run all health checks."""
    
    print("🤖 YOLO Server Health Check")
    print("=" * 50)
    
    # Basic health check
    server_healthy = test_yolo_server()
    
    if server_healthy:
        # Try sample request
        test_with_sample_request()
        
        print("\n🎉 Health check complete!")
        print("\n💡 Next steps:")
        print("   - If detection fails, check the expected request format")
        print("   - Verify the YOLO model is loaded and ready")
        print("   - Check server logs for any errors")
    else:
        print("\n❌ Server health check failed!")
        print("\n🔧 Troubleshooting:")
        print("   - Check if NGROK tunnel is active")
        print("   - Verify the YOLO server is running")
        print("   - Update NGROK_URL in .env file if needed")

if __name__ == "__main__":
    main()