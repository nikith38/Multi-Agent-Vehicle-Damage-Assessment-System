"""Test script to verify confidence filtering functionality in DamageDetectionAgent.

This script creates mock YOLO responses with various confidence levels to test
that predictions below 0.75 are properly filtered out.
"""

import json
import sys
import os
from unittest.mock import Mock, patch

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.DamageDetectionAgent import DamageDetectionAgent


def create_mock_yolo_response(detections):
    """Create a mock YOLO response with specified detections."""
    return {
        'detections': detections,
        'processing_time': 0.1,
        'image_dimensions': [640, 480]
    }


def test_confidence_filtering():
    """Test that predictions below 0.75 confidence are filtered out."""
    print("Testing confidence filtering functionality...")
    
    # Create a mock agent (we'll mock the HTTP request)
    agent = DamageDetectionAgent(
        model_url="http://mock-url",
        confidence_threshold=0.5,  # This should be overridden by our 0.75 filter
        enable_logging=True
    )
    
    # Create test detections with various confidence levels
    test_detections = [
        {
            'class_name': 'dent',
            'class_id': 0,
            'confidence': 0.95,  # Should be kept
            'bbox': [100, 100, 200, 200]
        },
        {
            'class_name': 'scratch',
            'class_id': 1,
            'confidence': 0.65,  # Should be filtered out
            'bbox': [300, 300, 400, 400]
        },
        {
            'class_name': 'crack',
            'class_id': 2,
            'confidence': 0.80,  # Should be kept
            'bbox': [500, 500, 600, 600]
        },
        {
            'class_name': 'dent',
            'class_id': 0,
            'confidence': 0.45,  # Should be filtered out
            'bbox': [700, 700, 800, 800]
        },
        {
            'class_name': 'scratch',
            'class_id': 1,
            'confidence': 0.75,  # Should be kept (exactly at threshold)
            'bbox': [900, 900, 1000, 1000]
        }
    ]
    
    mock_yolo_response = create_mock_yolo_response(test_detections)
    
    # Test the _process_detection_results method directly
    result = agent._process_detection_results(
        yolo_response=mock_yolo_response,
        image_path="test_image.jpg",
        confidence_threshold=0.5  # This should not affect our 0.75 filter
    )
    
    # Verify results
    print(f"\nOriginal detections: {len(test_detections)}")
    print(f"Filtered detections: {result['total_damages']}")
    print(f"Raw detections count: {result['detection_metadata']['total_raw_detections']}")
    print(f"Filtered low confidence: {result['detection_metadata']['filtered_low_confidence']}")
    
    # Expected: 3 detections should remain (0.95, 0.80, 0.75)
    # Expected: 2 detections should be filtered (0.65, 0.45)
    expected_kept = 3
    expected_filtered = 2
    
    assert result['total_damages'] == expected_kept, f"Expected {expected_kept} detections, got {result['total_damages']}"
    assert result['detection_metadata']['filtered_low_confidence'] == expected_filtered, f"Expected {expected_filtered} filtered, got {result['detection_metadata']['filtered_low_confidence']}"
    
    # Verify that kept detections have correct confidence values
    kept_confidences = [det['confidence'] for det in result['damage_detections']]
    print(f"Kept confidences: {kept_confidences}")
    
    for confidence in kept_confidences:
        assert confidence >= 0.75, f"Detection with confidence {confidence} should have been filtered"
    
    # Verify damage IDs are sequential
    damage_ids = [det['damage_id'] for det in result['damage_detections']]
    expected_ids = ['dmg_001', 'dmg_002', 'dmg_003']
    assert damage_ids == expected_ids, f"Expected {expected_ids}, got {damage_ids}"
    
    print("\n✅ All confidence filtering tests passed!")
    print(f"✅ Successfully filtered out {expected_filtered} low-confidence predictions")
    print(f"✅ Kept {expected_kept} high-confidence predictions")
    print("✅ Damage IDs are properly sequential")
    
    return True


def test_edge_cases():
    """Test edge cases for confidence filtering."""
    print("\nTesting edge cases...")
    
    agent = DamageDetectionAgent(
        model_url="http://mock-url",
        enable_logging=True
    )
    
    # Test with no detections
    empty_response = create_mock_yolo_response([])
    result = agent._process_detection_results(empty_response, "test.jpg", 0.5)
    assert result['total_damages'] == 0
    assert result['detection_metadata']['filtered_low_confidence'] == 0
    print("✅ Empty detections handled correctly")
    
    # Test with all detections below threshold
    low_confidence_detections = [
        {'class_name': 'dent', 'class_id': 0, 'confidence': 0.3, 'bbox': [0, 0, 100, 100]},
        {'class_name': 'scratch', 'class_id': 1, 'confidence': 0.6, 'bbox': [100, 100, 200, 200]}
    ]
    low_response = create_mock_yolo_response(low_confidence_detections)
    result = agent._process_detection_results(low_response, "test.jpg", 0.5)
    assert result['total_damages'] == 0
    assert result['detection_metadata']['filtered_low_confidence'] == 2
    print("✅ All low-confidence detections filtered correctly")
    
    # Test with all detections above threshold
    high_confidence_detections = [
        {'class_name': 'dent', 'class_id': 0, 'confidence': 0.9, 'bbox': [0, 0, 100, 100]},
        {'class_name': 'scratch', 'class_id': 1, 'confidence': 0.85, 'bbox': [100, 100, 200, 200]}
    ]
    high_response = create_mock_yolo_response(high_confidence_detections)
    result = agent._process_detection_results(high_response, "test.jpg", 0.5)
    assert result['total_damages'] == 2
    assert result['detection_metadata']['filtered_low_confidence'] == 0
    print("✅ All high-confidence detections kept correctly")
    
    return True


if __name__ == "__main__":
    try:
        # Run the tests
        test_confidence_filtering()
        test_edge_cases()
        
        print("\n🎉 All tests passed! Confidence filtering is working correctly.")
        print("📊 Summary:")
        print("   - Predictions with confidence < 0.75 are filtered out")
        print("   - Predictions with confidence >= 0.75 are kept")
        print("   - Metadata includes filtering statistics")
        print("   - Damage IDs remain sequential after filtering")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)