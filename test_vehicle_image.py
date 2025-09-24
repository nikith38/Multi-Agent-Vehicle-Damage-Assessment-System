#!/usr/bin/env python3
"""Simple test script to display detailed image quality assessment results"""

import sys
sys.path.append('tools')
from imagetools import assess_image_metrics
import json

def test_vehicle_image():
    """Test the vehicle image and display detailed results"""
    
    # Test with the real vehicle image
    test_input = {
        'image_path': 'tests/front_left-18.jpeg',
        'region_analysis': True,
        'adaptive_thresholds': True
    }

    print('=== IMAGE QUALITY ASSESSMENT RESULTS ===')
    print(f'Image: {test_input["image_path"]}')
    print()

    result = assess_image_metrics(json.dumps(test_input))
    result_dict = json.loads(result)

    print(f'Is Processable: {result_dict["is_processable"]}')
    print(f'Processing Time: {result_dict["processing_time"]:.3f}s')
    print()

    print('OVERALL METRICS:')
    overall = result_dict['overall_metrics']
    print(f'  Blur Score: {overall["blur_score"]:.3f} (higher = sharper)')
    print(f'  Brightness: {overall["brightness_score"]:.1f} (0-255)')
    print(f'  Contrast: {overall["contrast_score"]:.3f} (0-1)')
    print(f'  Noise Level: {overall["noise_level"]:.3f} (lower = cleaner)')
    print(f'  Overall Score: {overall["overall_score"]:.3f} (0-1)')
    print()

    if result_dict['region_metrics']:
        print('REGION ANALYSIS:')
        regions = result_dict['region_metrics']
        for region_name, metrics in regions.items():
            if region_name != 'weighted_average':
                print(f'  {region_name.upper()}:')
                print(f'    Blur: {metrics["blur_score"]:.3f}, Brightness: {metrics["brightness_score"]:.1f}')
                print(f'    Contrast: {metrics["contrast_score"]:.3f}, Noise: {metrics["noise_level"]:.3f}')
        print()

    print('QUALITY ISSUES:')
    for issue in result_dict['quality_issues']:
        print(f'  - {issue}')
    
    print()
    print('RECOMMENDATIONS:')
    for rec in result_dict['recommendations']:
        print(f'  - {rec}')

if __name__ == "__main__":
    test_vehicle_image()