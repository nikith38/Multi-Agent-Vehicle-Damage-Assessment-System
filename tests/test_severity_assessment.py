#!/usr/bin/env python3
"""
Test script for SeverityAssessmentAgent
Demonstrates how to use the severity assessment functionality with sample data.
"""

import json
import asyncio
from pathlib import Path
import sys
import os

# Add the parent directory to the path to import our modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Change to the project directory
os.chdir(parent_dir)

from agents.SeverityAssessmentAgent import create_severity_assessment_agent

# Sample damage detection results (from YOLO)
sample_damage_detection = {
    "success": True,
    "damages_detected": [
        {
            "damage_id": "damage_1",
            "damage_type": "dent",
            "confidence": 0.85,
            "bbox": [100, 150, 200, 250],
            "damage_percentage": 15.2
        },
        {
            "damage_id": "damage_2",
            "damage_type": "paint_damage",
            "confidence": 0.92,
            "bbox": [300, 100, 450, 200],
            "damage_percentage": 8.7
        }
    ],
    "total_damages": 2,
    "processing_time": 2.3
}

# Sample part identification results (from GPT-4o mini)
sample_part_identification = {
    "success": True,
    "damaged_parts": [
        {
            "damage_id": "damage_1",
            "part_name": "front bumper",
            "part_id": "FB001",
            "damage_type": "dent",
            "damage_percentage": 15.2,
            "bbox": [100, 150, 200, 250],
            "confidence": 0.85,
            "location_description": "Lower left section of front bumper showing significant denting"
        },
        {
            "damage_id": "damage_2",
            "part_name": "front fender (right)",
            "part_id": "FF002",
            "damage_type": "paint_damage",
            "damage_percentage": 8.7,
            "bbox": [300, 100, 450, 200],
            "confidence": 0.92,
            "location_description": "Surface paint scratches on the right front fender"
        }
    ],
    "total_parts_identified": 2,
    "processing_time": 1.8
}

def test_severity_assessment():
    """
    Test the SeverityAssessmentAgent with sample data
    """
    print("🚗 Testing Severity Assessment Agent")
    print("=" * 50)
    
    try:
        # Create the severity assessment agent
        agent = create_severity_assessment_agent()
        print("✅ Severity Assessment Agent created successfully")
        
        # Perform severity assessment
        print("\n🔍 Analyzing damage severity...")
        result = agent.assess_damage_severity(
            damage_detection_results=sample_damage_detection,
            part_identification_results=sample_part_identification
        )
        
        if result["success"]:
            print("\n✅ Severity Assessment Completed Successfully!")
            print(f"⏱️  Processing Time: {result['processing_time']:.2f} seconds")
            
            # Display core assessment
            print("\n📊 CORE ASSESSMENT")
            print("-" * 30)
            print(f"Overall Severity: {result['severity_classification']}")
            print(f"Severity Score: {result['severity_score']}/100")
            print(f"Repairability: {result.get('repairability', 'N/A')}")
            print(f"Vehicle Safety Status: {result.get('vehicle_safety_status', 'N/A')}")
            
            # Display available assessment data
            print("\n📋 ASSESSMENT DETAILS")
            print("-" * 30)
            
            # Show all available keys in the result
            print("Available assessment data:")
            for key, value in result.items():
                if key not in ['success', 'processing_time']:
                    if isinstance(value, (dict, list)):
                        print(f"  {key}: {type(value).__name__} with {len(value)} items")
                    else:
                        print(f"  {key}: {value}")
            
            # Display cost analysis if available
            if 'cost_analysis' in result:
                cost = result['cost_analysis']
                print("\n💰 COST ANALYSIS")
                print("-" * 30)
                print(f"Estimated Repair Cost: ₹{cost.get('estimated_repair_cost_min', 0):,.2f} - ₹{cost.get('estimated_repair_cost_max', 0):,.2f}")
                print(f"Labor Hours: {cost.get('estimated_labor_hours_min', 0)} - {cost.get('estimated_labor_hours_max', 0)}")
            
            # Display repair recommendations if available
            if 'repair_recommendations' in result and result['repair_recommendations']:
                print("\n🔧 REPAIR RECOMMENDATIONS")
                print("-" * 30)
                for i, rec in enumerate(result['repair_recommendations'], 1):
                    print(f"{i}. {rec.get('recommendation_type', 'Unknown')}: {rec.get('description', 'No description')}")
            
            # Display safety concerns if available
            if 'safety_concerns' in result and result['safety_concerns']:
                print("\n⚠️  SAFETY CONCERNS")
                print("-" * 30)
                for concern in result['safety_concerns']:
                    print(f"• {concern.get('concern_type', 'Unknown')}: {concern.get('description', 'No description')}")
            
            # Display explanations if available
            if 'explanations' in result:
                explanations = result['explanations']
                print("\n📝 EXPLANATIONS")
                print("-" * 30)
                print(f"Severity Reasoning: {explanations.get('severity_reasoning', 'N/A')}")
                print(f"Cost Reasoning: {explanations.get('cost_reasoning', 'N/A')}")
                print(f"Repair Reasoning: {explanations.get('repair_reasoning', 'N/A')}")
            
            # Display statistics
            stats = agent.get_statistics()
            print("\n📈 AGENT STATISTICS")
            print("-" * 30)
            print(f"Total Assessments: {stats['total_assessments']}")
            print(f"Successful Assessments: {stats['successful_assessments']}")
            print(f"Average Processing Time: {stats['average_processing_time']:.2f}s")
            
            # Save results to file
            output_file = Path(__file__).parent / "severity_assessment_results.json"
            with open(output_file, 'w') as f:
                # Save the complete result with statistics
                json.dump({
                    "assessment_result": result,
                    "agent_statistics": stats
                }, f, indent=2)
            
            print(f"\n💾 Results saved to: {output_file}")
            
        else:
            print(f"❌ Assessment failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """
    Main function to run the severity assessment test
    """
    print("Starting Severity Assessment Agent Test...")
    test_severity_assessment()

if __name__ == "__main__":
    main()