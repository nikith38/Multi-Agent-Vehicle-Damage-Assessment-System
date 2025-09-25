"""Severity Assessment Agent

This agent analyzes vehicle damage data from damage detection and part identification
to provide comprehensive severity assessments including cost estimates, repair recommendations,
and safety concern identification.
"""

import json
import os
import time
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

from .severity_models import SeverityAssessment
from .prompts.severity_assessment_prompt import SEVERITY_ASSESSMENT_PROMPT

# Load environment variables
load_dotenv(override=True)


class SeverityAssessmentAgent:
    """Agent for assessing vehicle damage severity using LangChain and ChatOpenAI."""
    
    def __init__(self, openai_api_key: Optional[str] = None, model: str = "gpt-4o-mini",
                 temperature: float = 0.1, timeout: int = 60, enable_logging: bool = True):
        """Initialize the Severity Assessment Agent.
        
        Args:
            openai_api_key: OpenAI API key (if None, loads from OPENAI_API_KEY env var)
            model: OpenAI model to use (default: gpt-4o-mini)
            temperature: Model temperature for consistency (default: 0.1)
            timeout: Request timeout in seconds
            enable_logging: Enable detailed logging
        """
        # Load API key from environment if not provided
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key must be provided either as parameter or OPENAI_API_KEY environment variable")
        
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.enable_logging = enable_logging
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            openai_api_key=self.openai_api_key,
            request_timeout=self.timeout
        )
        
        # Initialize Pydantic output parser
        self.output_parser = PydanticOutputParser(pydantic_object=SeverityAssessment)
        
        # Create prompt template
        self.prompt_template = PromptTemplate(
            template=SEVERITY_ASSESSMENT_PROMPT + "\n\n{format_instructions}",
            input_variables=["damage_data", "part_data"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )
        
        # Statistics tracking
        self.stats = {
            'total_assessments': 0,
            'successful_assessments': 0,
            'failed_assessments': 0,
            'total_processing_time': 0.0,
            'average_severity_score': 0.0,
            'severity_distribution': {
                'minor': 0,
                'moderate': 0,
                'major': 0,
                'severe': 0
            }
        }
        
        if self.enable_logging:
            print(f"SeverityAssessmentAgent initialized with model: {self.model}")
    
    def assess_damage_severity(self, damage_detection_results: Dict[str, Any], 
                             part_identification_results: Dict[str, Any]) -> Dict[str, Any]:
        """Main method to assess damage severity from detection and identification results.
        
        Args:
            damage_detection_results: Results from DamageDetectionAgent
            part_identification_results: Results from PartIdentificationAgent
            
        Returns:
            Dictionary containing severity assessment results
        """
        start_time = time.time()
        self.stats['total_assessments'] += 1
        
        try:
            # Prepare input data
            damage_data = json.dumps(damage_detection_results, indent=2)
            part_data = json.dumps(part_identification_results, indent=2)
            
            if self.enable_logging:
                print(f"Analyzing {len(damage_detection_results.get('damage_detections', []))} damages and {len(part_identification_results.get('damaged_parts', []))} parts")
            
            # Format prompt with data
            formatted_prompt = self.prompt_template.format(
                damage_data=damage_data,
                part_data=part_data
            )
            
            # Get LLM response
            response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
            
            # Parse response using Pydantic
            assessment = self.output_parser.parse(response.content)
            
            # Convert to dictionary and add metadata
            assessment_dict = assessment.dict()
            assessment_dict['success'] = True
            assessment_dict['processing_time'] = time.time() - start_time
            
            # Update statistics
            self._update_statistics(assessment, time.time() - start_time)
            
            if self.enable_logging:
                print(f"Severity assessment completed: {assessment.severity_classification.value}")
                print(f"Severity score: {assessment.severity_score}/100")
                print(f"Safety concerns: {len(assessment.safety_concerns)}")
                print(f"Processing time: {assessment_dict['processing_time']:.2f}s")
            
            # Save output JSON file
            image_path = damage_detection_results.get('image_path', 'unknown')
            self._save_output_json(assessment_dict, image_path)
            
            return assessment_dict
            
        except Exception as e:
            self.stats['failed_assessments'] += 1
            error_msg = f"Severity assessment failed: {str(e)}"
            
            if self.enable_logging:
                print(error_msg)
            
            return {
                'success': False,
                'error': error_msg,
                'severity_classification': 'unknown',
                'severity_score': 0,
                'processing_time': time.time() - start_time
            }
    
    def _update_statistics(self, assessment: SeverityAssessment, processing_time: float):
        """Update internal statistics with assessment results.
        
        Args:
            assessment: The completed severity assessment
            processing_time: Time taken for the assessment
        """
        self.stats['successful_assessments'] += 1
        self.stats['total_processing_time'] += processing_time
        
        # Update severity distribution
        severity_level = assessment.severity_classification.value
        if severity_level in self.stats['severity_distribution']:
            self.stats['severity_distribution'][severity_level] += 1
        
        # Update average severity score
        total_successful = self.stats['successful_assessments']
        current_avg = self.stats['average_severity_score']
        self.stats['average_severity_score'] = (
            (current_avg * (total_successful - 1) + assessment.severity_score) / total_successful
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent performance statistics.
        
        Returns:
            Dictionary containing performance metrics
        """
        success_rate = (
            self.stats['successful_assessments'] / 
            max(1, self.stats['total_assessments'])
        ) * 100
        
        avg_processing_time = (
            self.stats['total_processing_time'] / 
            max(1, self.stats['successful_assessments'])
        )
        
        return {
            'total_assessments': self.stats['total_assessments'],
            'successful_assessments': self.stats['successful_assessments'],
            'failed_assessments': self.stats['failed_assessments'],
            'success_rate_percent': round(success_rate, 2),
            'total_processing_time': round(self.stats['total_processing_time'], 2),
            'average_processing_time': round(avg_processing_time, 2),
            'average_severity_score': round(self.stats['average_severity_score'], 2),
            'severity_distribution': self.stats['severity_distribution'].copy()
        }
    
    def reset_statistics(self):
        """Reset all statistics counters."""
        self.stats = {
            'total_assessments': 0,
            'successful_assessments': 0,
            'failed_assessments': 0,
            'total_processing_time': 0.0,
            'average_severity_score': 0.0,
            'severity_distribution': {
                'minor': 0,
                'moderate': 0,
                'major': 0,
                'severe': 0
            }
        }
        
        if self.enable_logging:
            print("Statistics reset successfully")
    
    def _save_output_json(self, result: Dict[str, Any], image_path: str) -> Optional[str]:
        """Save severity assessment results to JSON file.
        
        Args:
            result: Severity assessment results dictionary
            image_path: Original image path for naming the output file
            
        Returns:
            Path to saved JSON file or None if failed
        """
        try:
            # Generate output filename based on image path
            # If image_path is in enhanced folder, use that for naming
            if 'enhanced' in image_path:
                base_name = os.path.splitext(image_path)[0]
            else:
                # For original images, create enhanced folder path
                base_name = os.path.splitext(image_path)[0]
                if not base_name.startswith('enhanced'):
                    filename = os.path.basename(base_name)
                    base_name = f"enhanced/{filename}_enhanced"
            
            output_file = f"{base_name}_severity_assessment_output.json"
            
            # Ensure enhanced directory exists
            os.makedirs('enhanced', exist_ok=True)
            
            # Write JSON file
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=4)
            
            if self.enable_logging:
                print(f"Severity assessment output saved: {output_file}")
            
            return output_file
        except Exception as e:
            if self.enable_logging:
                print(f"Failed to save severity assessment output JSON: {str(e)}")
            return None


def create_severity_assessment_agent(openai_api_key: Optional[str] = None, **kwargs) -> SeverityAssessmentAgent:
    """Factory function to create a SeverityAssessmentAgent instance.
    
    Args:
        openai_api_key: OpenAI API key (if None, loads from OPENAI_API_KEY env var)
        **kwargs: Additional configuration options
        
    Returns:
        Configured SeverityAssessmentAgent instance
    """
    return SeverityAssessmentAgent(openai_api_key=openai_api_key, **kwargs)


# Example usage
if __name__ == "__main__":
    # Example damage detection results
    sample_damage_results = {
        'success': True,
        'damage_detections': [
            {
                'damage_id': 'D01',
                'class': 'dent',
                'confidence': 0.85,
                'bbox': [100, 150, 200, 250]
            },
            {
                'damage_id': 'D02',
                'class': 'scratch',
                'confidence': 0.72,
                'bbox': [300, 100, 450, 180]
            }
        ],
        'total_damages_detected': 2,
        'processing_time': 0.24
    }
    
    # Example part identification results
    sample_part_results = {
        'success': True,
        'damaged_parts': [
            {
                'damage_id': 'D01',
                'part_name': 'front_bumper',
                'part_id': 'P001',
                'damage_type': 'dent',
                'damage_percentage': 30,
                'bbox': [100, 150, 200, 250],
                'confidence': 0.85,
                'location_description': 'lower section near fog light area'
            },
            {
                'damage_id': 'D02',
                'part_name': 'front_fender_right',
                'part_id': 'P011',
                'damage_type': 'scratch',
                'damage_percentage': 15,
                'bbox': [300, 100, 450, 180],
                'confidence': 0.72,
                'location_description': 'upper section near headlight'
            }
        ],
        'total_parts_identified': 2,
        'processing_time': 5.45
    }
    
    # Create agent (requires OpenAI API key)
    # agent = create_severity_assessment_agent(
    #     openai_api_key="your-openai-api-key",
    #     enable_logging=True
    # )
    
    # Assess damage severity
    # results = agent.assess_damage_severity(
    #     damage_detection_results=sample_damage_results,
    #     part_identification_results=sample_part_results
    # )
    
    # print(json.dumps(results, indent=2))
    # print("\nAgent Statistics:")
    # print(json.dumps(agent.get_statistics(), indent=2))
    
    print("SeverityAssessmentAgent implementation complete!")