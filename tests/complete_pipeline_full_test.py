#!/usr/bin/env python3
"""
Complete Pipeline Test - Runs all 4 agents in sequence
Image Enhancement -> Damage Detection -> Part Identification -> Severity Assessment
"""

import json
import os
import time
from pathlib import Path
import sys
from dotenv import load_dotenv

# Add the parent directory to the path to import our modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
os.chdir(parent_dir)

# Load environment variables
load_dotenv()

from agents.ImageAgent import ImageEnhancementAgent
from agents.DamageDetectionAgent import DamageDetectionAgent
from agents.PartIdentificationAgent import PartIdentificationAgent
from agents.SeverityAssessmentAgent import create_severity_assessment_agent

def run_complete_pipeline():
    """
    Run the complete vehicle damage assessment pipeline
    """
    print("🚗 COMPLETE VEHICLE DAMAGE ASSESSMENT PIPELINE")
    print("=" * 60)
    
    # Configuration
    input_image_path = Path(__file__).parent / "front_left-18.jpeg"
    ngrok_url = os.getenv('NGROK_URL')
    
    if not input_image_path.exists():
        print(f"❌ Input image not found: {input_image_path}")
        return
    
    if not ngrok_url:
        print("❌ NGROK_URL not found in environment variables")
        return
    
    print(f"📸 Input Image: {input_image_path}")
    print(f"🌐 YOLO API URL: {ngrok_url}")
    print()
    
    pipeline_start_time = time.time()
    results = {}
    
    try:
        # STEP 1: Image Enhancement
        print("🔧 STEP 1: IMAGE ENHANCEMENT")
        print("-" * 40)
        
        image_agent = ImageEnhancementAgent()
        enhanced_result = image_agent.enhance_image(str(input_image_path))
        
        if 'error' in enhanced_result:
            print(f"❌ Image enhancement failed: {enhanced_result.get('error', 'Unknown error')}")
            return
        
        # The enhanced image path should be in the same directory with _enhanced suffix
        enhanced_image_path = str(input_image_path).replace('.jpeg', '_enhanced.jpeg')
        results['image_enhancement'] = enhanced_result
        
        print(f"✅ Image enhanced successfully")
        print(f"📁 Enhanced image: {enhanced_image_path}")
        print(f"📝 Agent output: {enhanced_result.get('output', 'No output available')}")
        print()
        
        # STEP 2: Damage Detection
        print("🔍 STEP 2: DAMAGE DETECTION")
        print("-" * 40)
        
        damage_agent = DamageDetectionAgent(ngrok_url)
        damage_result = damage_agent.detect_damage(enhanced_image_path)
        
        if not damage_result['success']:
            print(f"❌ Damage detection failed: {damage_result.get('error', 'Unknown error')}")
            return
        
        results['damage_detection'] = damage_result
        
        print(f"✅ Damage detection completed")
        print(f"🔢 Damages detected: {damage_result['total_damages']}")
        print(f"⏱️  Processing time: {damage_result['processing_time']:.2f}s")
        
        # Display detected damages
        for i, damage in enumerate(damage_result['damage_detections'], 1):
            print(f"  {i}. {damage['type']} (confidence: {damage['confidence']:.2f})")
        print()
        
        # STEP 3: Part Identification
        print("🎯 STEP 3: PART IDENTIFICATION")
        print("-" * 40)
        
        part_agent = PartIdentificationAgent()
        part_result = part_agent.identify_damaged_parts(
            enhanced_image_path, 
            damage_result['damage_detections']
        )
        
        if not part_result['success']:
            print(f"❌ Part identification failed: {part_result.get('error', 'Unknown error')}")
            return
        
        results['part_identification'] = part_result
        
        print(f"✅ Part identification completed")
        print(f"🔧 Parts identified: {part_result['total_parts_identified']}")
        print(f"⏱️  Processing time: {part_result['processing_time']:.2f}s")
        
        # Display identified parts
        for i, part in enumerate(part_result['damaged_parts'], 1):
            print(f"  {i}. {part['part_name']} - {part['damage_type']} ({part['damage_percentage']:.1f}% damage)")
        print()
        
        # STEP 4: Severity Assessment
        print("⚖️  STEP 4: SEVERITY ASSESSMENT")
        print("-" * 40)
        
        severity_agent = create_severity_assessment_agent()
        severity_result = severity_agent.assess_damage_severity(
            damage_detection_results=damage_result,
            part_identification_results=part_result
        )
        
        if not severity_result['success']:
            print(f"❌ Severity assessment failed: {severity_result.get('error', 'Unknown error')}")
            return
        
        results['severity_assessment'] = severity_result
        
        print(f"✅ Severity assessment completed")
        print(f"📊 Overall Severity: {severity_result['severity_classification']}")
        print(f"📈 Severity Score: {severity_result['severity_score']}/100")
        print(f"⏱️  Processing time: {severity_result['processing_time']:.2f}s")
        print()
        
        # PIPELINE SUMMARY
        total_pipeline_time = time.time() - pipeline_start_time
        
        print("📋 PIPELINE SUMMARY")
        print("=" * 60)
        print(f"🎯 Total Damages Detected: {damage_result['total_damages']}")
        print(f"🔧 Total Parts Identified: {part_result['total_parts_identified']}")
        print(f"📊 Severity Classification: {severity_result['severity_classification']}")
        print(f"📈 Severity Score: {severity_result['severity_score']}/100")
        print(f"⏱️  Total Pipeline Time: {total_pipeline_time:.2f}s")
        print()
        
        # DETAILED RESULTS
        print("📝 DETAILED ASSESSMENT")
        print("-" * 40)
        
        # Cost Analysis
        if 'repair_cost_estimate' in severity_result:
            cost_data = severity_result['repair_cost_estimate']
            print(f"💰 Estimated Repair Cost: ₹{cost_data.get('estimated_repair_cost_min', 0):,.0f} - ₹{cost_data.get('estimated_repair_cost_max', 0):,.0f}")
            print(f"⏰ Estimated Repair Time: {cost_data.get('estimated_repair_days_min', 0)}-{cost_data.get('estimated_repair_days_max', 0)} days")
        
        # Safety Assessment
        if 'safety_concerns' in severity_result and severity_result['safety_concerns']:
            print(f"⚠️  Safety Concerns: {len(severity_result['safety_concerns'])} identified")
            for concern in severity_result['safety_concerns']:
                print(f"   • {concern.get('concern_type', 'Unknown')}: {concern.get('description', 'No description')}")
        else:
            print("✅ No safety concerns identified")
        
        # Explanations
        if 'severity_explanation' in severity_result:
            print(f"\n📖 Assessment Explanation:")
            print(f"   {severity_result['severity_explanation']}")
        
        print()
        
        # Save complete results
        output_file = Path(__file__).parent / "complete_pipeline_full_results.json"
        results['pipeline_summary'] = {
            'total_damages_detected': damage_result['total_damages'],
            'total_parts_identified': part_result['total_parts_identified'],
            'severity_classification': severity_result['severity_classification'],
            'severity_score': severity_result['severity_score'],
            'total_pipeline_time': total_pipeline_time,
            'processing_times': {
                'damage_detection': damage_result['processing_time'],
                'part_identification': part_result['processing_time'],
                'severity_assessment': severity_result['processing_time']
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Complete results saved to: {output_file}")
        print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """
    Main function to run the complete pipeline test
    """
    print("Starting Complete Vehicle Damage Assessment Pipeline...")
    print()
    run_complete_pipeline()

if __name__ == "__main__":
    main()