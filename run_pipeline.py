#!/usr/bin/env python3
"""
Run the complete damage assessment pipeline with images from tests folder.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from orchestrator import create_damage_assessment_orchestrator

def main():
    print("🚀 Running Complete Damage Assessment Pipeline")
    print("=" * 60)
    
    # Get API key from environment or use test key
    api_key = os.getenv('OPENAI_API_KEY', 'test-key')
    ngrok_url = 'https://8816baf1bf4f.ngrok-free.app/'
    
    print(f"🔑 Using API Key: {'***' + api_key[-4:] if len(api_key) > 4 else 'test-key'}")
    print(f"🌐 NGROK URL: {ngrok_url}")
    
    # Create orchestrator
    orchestrator = create_damage_assessment_orchestrator(
        openai_api_key=api_key,
        ngrok_url=ngrok_url,
        confidence_threshold=0.5,
        max_retries=3,
        enable_logging=True
    )
    
    # Get images from tests folder, excluding enhanced/processed images
    tests_dir = Path("tests")
    all_image_files = list(tests_dir.glob("*.jpg")) + list(tests_dir.glob("*.jpeg")) + list(tests_dir.glob("*.png"))
    
    # Filter out enhanced images and agent output images
    excluded_suffixes = [
        '_enhanced', '_final', '_contrast_enhanced', '_adjusted', 
        '_sharpened', '_exposure_adjusted', '_damage_detection', 
        '_part_identification', '_severity_assessment'
    ]
    
    image_files = []
    for img_file in all_image_files:
        # Skip images with enhancement suffixes
        if not any(suffix in img_file.stem for suffix in excluded_suffixes):
            image_files.append(img_file)
    
    if not image_files:
        print("❌ No original image files found in tests folder")
        print(f"   Found {len(all_image_files)} total images, but all appear to be processed/enhanced versions")
        return
    
    # Convert to relative paths as strings
    image_paths = [f"tests/{img.name}" for img in image_files]
    
    print(f"\n📸 Found {len(image_paths)} images in tests folder:")
    for img in image_paths:
        print(f"   - {img}")
    
    print(f"\n🎯 Pipeline Configuration:")
    print(f"   - Confidence Threshold: 0.5")
    print(f"   - Max Retries: 3 (ignored in fail-fast mode)")
    print(f"   - Fail-Fast Mode: Enabled")
    
    print(f"\n🚀 Starting pipeline execution...")
    print("-" * 40)
    
    try:
        # Run the pipeline
        results = orchestrator.process_claim(
            images=image_paths,
            confidence_threshold=0.5,
            max_retries=3
        )
        
        print("\n" + "=" * 60)
        print("📊 PIPELINE RESULTS")
        print("=" * 60)
        
        print(f"✅ Overall Success: {results['success']}")
        print(f"📸 Images Processed: {results['images_processed']}/{len(image_paths)}")
        print(f"❌ Images Failed: {results['images_failed']}")
        print(f"⏱️  Total Processing Time: {results['total_processing_time']:.2f}s")
        print(f"📋 Manual Review Required: {results.get('requires_manual_review', False)}")
        
        # Pipeline metadata
        metadata = results.get('pipeline_metadata', {})
        print(f"\n🔍 PIPELINE METADATA:")
        print(f"   🛑 Terminated Due to Error: {metadata.get('workflow_terminated_due_to_error', False)}")
        print(f"   📝 Termination Reason: {metadata.get('termination_reason', 'N/A')}")
        print(f"   📅 Completed At: {metadata.get('processing_completed_at', 'N/A')}")
        
        # Failed images
        if results.get('failed_images'):
            print(f"\n❌ FAILED IMAGES:")
            for failed_img in results['failed_images']:
                print(f"   - {failed_img}")
        
        # Errors
        if results.get('errors'):
            print(f"\n🚨 ERRORS:")
            for error in results['errors']:
                print(f"   - {error}")
        
        # Individual results
        if results.get('individual_results'):
            print(f"\n📋 INDIVIDUAL RESULTS:")
            for i, result in enumerate(results['individual_results']):
                print(f"   Image {i+1}: {result['image_path']}")
                print(f"     Success: {result['success']}")
                print(f"     Confidence: {result['confidence_score']:.2f}")
                print(f"     Damages: {len(result['damage_detections'])}")
                print(f"     Processing Time: {result['processing_time']:.2f}s")
                if result['errors']:
                    print(f"     Errors: {result['errors']}")
        
        # Consolidated assessment
        if results.get('consolidated_assessment'):
            print(f"\n📊 CONSOLIDATED ASSESSMENT:")
            assessment = results['consolidated_assessment']
            print(f"   Total Damages: {assessment.get('total_damages_detected', 0)}")
            print(f"   Total Parts Affected: {assessment.get('total_parts_affected', 0)}")
            print(f"   Overall Severity: {assessment.get('overall_severity', 'N/A')}")
            print(f"   Average Confidence: {assessment.get('average_confidence', 0):.2f}")
        
        print("\n" + "=" * 60)
        
        # Summary based on termination status
        if metadata.get('workflow_terminated_due_to_error', False):
            print("🛑 WORKFLOW TERMINATED GRACEFULLY (FAIL-FAST MODE)")
            print(f"   Reason: {metadata.get('termination_reason', 'Unknown')}")
            print("   This demonstrates the fail-fast functionality working correctly.")
        else:
            print("✅ WORKFLOW COMPLETED NORMALLY")
            print("   All images were processed successfully or the workflow completed without early termination.")
        
    except Exception as e:
        print(f"\n❌ Pipeline execution failed with exception:")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()