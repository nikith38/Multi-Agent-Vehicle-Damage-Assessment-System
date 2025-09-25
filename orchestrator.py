"""LangGraph-based Damage Assessment Orchestrator

This orchestrator coordinates all agents in the vehicle damage assessment pipeline,
implementing routing logic, multi-image correlation, confidence-based workflows,
and graceful error handling using LangGraph's state management.
"""

import json
import os
import time
from typing import Dict, List, Optional, Any, TypedDict, Annotated
from datetime import datetime
from pathlib import Path

from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Always load environment variables
load_dotenv(override=True)

from agents.ImageAgent import ImageEnhancementAgent, create_image_agent
from agents.DamageDetectionAgent import DamageDetectionAgent, create_damage_detection_agent
from agents.PartIdentificationAgent import PartIdentificationAgent, create_part_identification_agent
from agents.SeverityAssessmentAgent import SeverityAssessmentAgent, create_severity_assessment_agent


class ImageProcessingResult(BaseModel):
    """Result structure for individual image processing."""
    image_path: str
    original_path: str
    enhanced_path: Optional[str] = None
    enhancement_metadata: Optional[Dict[str, Any]] = None
    damage_detections: List[Dict[str, Any]] = Field(default_factory=list)
    damaged_parts: List[Dict[str, Any]] = Field(default_factory=list)
    severity_assessment: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0
    confidence_score: float = 0.0
    errors: List[str] = Field(default_factory=list)
    success: bool = True


class PipelineState(TypedDict):
    """State schema for the damage assessment pipeline."""
    # Input data
    images: List[str]  # List of image paths
    current_image_index: int
    
    # Processing results
    image_results: List[ImageProcessingResult]
    current_result: Optional[ImageProcessingResult]
    
    # Pipeline configuration
    confidence_threshold: float
    retry_count: int
    max_retries: int
    
    # Final assessment
    consolidated_assessment: Optional[Dict[str, Any]]
    pipeline_metadata: Dict[str, Any]
    
    # Error handling
    errors: List[str]
    failed_images: List[str]
    workflow_terminated_due_to_error: bool
    termination_reason: str
    
    # Routing decisions
    next_step: str
    skip_enhancement: bool
    require_manual_review: bool


class DamageAssessmentOrchestrator:
    """LangGraph-based orchestrator for vehicle damage assessment pipeline."""
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 ngrok_url: Optional[str] = None,
                 confidence_threshold: float = 0.7,
                 max_retries: int = 2,
                 enable_logging: bool = True):
        """
        Initialize the orchestrator with all agents.
        
        Args:
            openai_api_key: OpenAI API key for LLM-based agents
            ngrok_url: NGROK URL for YOLO damage detection
            confidence_threshold: Minimum confidence for processing decisions
            max_retries: Maximum retry attempts for failed operations
            enable_logging: Enable detailed logging
        """
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries
        self.enable_logging = enable_logging
        
        # Initialize agents
        self.image_agent = create_image_agent(
            api_key=openai_api_key,
            enable_logging=enable_logging
        )
        
        self.damage_agent = create_damage_detection_agent(
            model_url=ngrok_url or "http://localhost:8000",  # Default fallback URL
            enable_logging=enable_logging
        )
        
        self.part_agent = create_part_identification_agent(
            openai_api_key=openai_api_key,
            enable_logging=enable_logging
        )
        
        self.severity_agent = create_severity_assessment_agent(
            openai_api_key=openai_api_key,
            enable_logging=enable_logging
        )
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for damage assessment."""
        workflow = StateGraph(PipelineState)
        
        # Add nodes
        workflow.add_node("initialize", self._initialize_pipeline)
        workflow.add_node("enhance_image", self._enhance_image_node)
        workflow.add_node("detect_damage", self._detect_damage_node)
        workflow.add_node("identify_parts", self._identify_parts_node)
        workflow.add_node("assess_severity", self._assess_severity_node)
        workflow.add_node("consolidate_results", self._consolidate_results_node)
        workflow.add_node("handle_error", self._handle_error_node)
        workflow.add_node("finalize", self._finalize_pipeline)
        
        # Define edges and routing logic
        workflow.set_entry_point("initialize")
        
        # From initialize, decide whether to enhance or skip to damage detection
        workflow.add_conditional_edges(
            "initialize",
            self._route_after_initialize,
            {
                "enhance": "enhance_image",
                "detect": "detect_damage",
                "error": "handle_error",
                "consolidate": "consolidate_results",
                "finalize": "finalize"
            }
        )
        
        # From enhancement, always go to damage detection
        workflow.add_conditional_edges(
            "enhance_image",
            self._route_after_enhancement,
            {
                "detect": "detect_damage",
                "error": "handle_error",
                "initialize": "initialize"
            }
        )
        
        # From damage detection, decide based on confidence
        workflow.add_conditional_edges(
            "detect_damage",
            self._route_after_damage_detection,
            {
                "identify_parts": "identify_parts",
                "error": "handle_error",
                "initialize": "initialize",
                "manual_review": "finalize",
                "next_image": "initialize"
            }
        )
        
        # From part identification to severity assessment
        workflow.add_conditional_edges(
            "identify_parts",
            self._route_after_part_identification,
            {
                "assess_severity": "assess_severity",
                "error": "handle_error",
                "initialize": "initialize"
            }
        )
        
        # From severity assessment, decide next steps
        workflow.add_conditional_edges(
            "assess_severity",
            self._route_after_severity_assessment,
            {
                "initialize": "initialize",
                "consolidate": "consolidate_results",
                "finalize": "finalize",
                "error": "handle_error"
            }
        )
        
        # From consolidation to finalization
        workflow.add_edge("consolidate_results", "finalize")
        
        # Error handling can retry or move to next image
        workflow.add_conditional_edges(
            "handle_error",
            self._route_after_error,
            {
                "initialize": "initialize",
                "finalize": "finalize"
            }
        )
        
        # Finalize ends the workflow
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _initialize_pipeline(self, state: PipelineState) -> PipelineState:
        """Initialize pipeline for current image."""
        if self.enable_logging:
            print(f"\n🚀 Initializing pipeline for image {state['current_image_index'] + 1}/{len(state['images'])}")
        
        # Check if we have more images to process
        if state['current_image_index'] >= len(state['images']):
            state['next_step'] = 'consolidate'
            return state
        
        # Get current image path
        current_image = state['images'][state['current_image_index']]
        
        # Initialize result for current image
        state['current_result'] = ImageProcessingResult(
            image_path=current_image,
            original_path=current_image
        )
        
        # Reset error handling
        state['retry_count'] = 0
        
        # Determine if enhancement is needed (simple heuristic)
        # In a real implementation, you might do a quick quality check
        state['skip_enhancement'] = False  # Always enhance for now
        state['next_step'] = 'enhance' if not state['skip_enhancement'] else 'detect'
        
        return state
    
    def _enhance_image_node(self, state: PipelineState) -> PipelineState:
        """Image enhancement node."""
        try:
            if self.enable_logging:
                print("🎨 Enhancing image...")
            
            start_time = time.time()
            current_image = state['current_result'].image_path
            
            # Enhance the image
            enhancement_result = self.image_agent.enhance_image(current_image)
            
            processing_time = time.time() - start_time
            
            # Handle the response from ImageEnhancementAgent
            if 'error' in enhancement_result:
                state['current_result'].errors.append(f"Enhancement failed: {enhancement_result['error']}")
                state['current_result'].success = False
                state['next_step'] = 'error'
            else:
                # The agent returns a response dict with 'output' containing the final answer
                agent_output = enhancement_result.get('output', '')
                enhanced_image_path = enhancement_result.get('enhanced_image_path')
                
                # Update the current result to use the enhanced image path for subsequent processing
                if enhanced_image_path:
                    state['current_result'].image_path = enhanced_image_path
                    if self.enable_logging:
                        print(f"📸 Using enhanced image for further processing: {enhanced_image_path}")
                
                state['current_result'].enhancement_metadata = {
                    'processing_time': processing_time,
                    'agent_output': agent_output,
                    'enhanced_image_path': enhanced_image_path,
                    'intermediate_steps': len(enhancement_result.get('intermediate_steps', []))
                }
                state['next_step'] = 'detect'
            
        except Exception as e:
            state['current_result'].errors.append(f"Enhancement error: {str(e)}")
            state['current_result'].success = False
            state['next_step'] = 'error'
        
        return state
    
    def _detect_damage_node(self, state: PipelineState) -> PipelineState:
        """Damage detection node."""
        try:
            if self.enable_logging:
                print("🔍 Detecting damage...")
            
            current_image = state['current_result'].image_path
            
            # Detect damage
            damage_result = self.damage_agent.detect_damage(
                image_path=current_image,
                confidence_threshold=state['confidence_threshold'],
                enhancement_metadata=state['current_result'].enhancement_metadata
            )
            
            if damage_result.get('success', False):
                state['current_result'].damage_detections = damage_result.get('damage_detections', [])
                
                # Calculate confidence score based on detections
                detections = damage_result.get('damage_detections', [])
                if detections:
                    avg_confidence = sum(d.get('confidence', 0) for d in detections) / len(detections)
                    state['current_result'].confidence_score = avg_confidence
                    
                    if avg_confidence >= state['confidence_threshold']:
                        state['next_step'] = 'identify_parts'
                    else:
                        state['require_manual_review'] = True
                        state['next_step'] = 'manual_review'
                else:
                    # No damage detected
                    state['current_result'].confidence_score = 1.0
                    state['next_step'] = 'next_image'
            else:
                state['current_result'].errors.append("Damage detection failed")
                state['current_result'].success = False
                state['next_step'] = 'error'
                
        except Exception as e:
            state['current_result'].errors.append(f"Damage detection error: {str(e)}")
            state['current_result'].success = False
            state['next_step'] = 'error'
        
        return state
    
    def _identify_parts_node(self, state: PipelineState) -> PipelineState:
        """Part identification node."""
        try:
            if self.enable_logging:
                print("🔧 Identifying damaged parts...")
            
            current_image = state['current_result'].image_path
            detections = state['current_result'].damage_detections
            
            # Convert detections to YOLO format for part identification
            yolo_detections = [
                {
                    'class': d.get('type', d.get('class', 'unknown')),
                    'confidence': d.get('confidence', 0),
                    'bbox': d.get('bbox', [])
                }
                for d in detections
            ]
            
            # Identify parts
            parts_result = self.part_agent.identify_damaged_parts(
                image_path=current_image,
                yolo_detections=yolo_detections
            )
            
            if parts_result.get('success', False):
                state['current_result'].damaged_parts = parts_result.get('damaged_parts', [])
                state['next_step'] = 'assess_severity'
            else:
                state['current_result'].errors.append("Part identification failed")
                state['current_result'].success = False
                state['next_step'] = 'error'
                
        except Exception as e:
            state['current_result'].errors.append(f"Part identification error: {str(e)}")
            state['current_result'].success = False
            state['next_step'] = 'error'
        
        return state
    
    def _assess_severity_node(self, state: PipelineState) -> PipelineState:
        """Severity assessment node."""
        try:
            if self.enable_logging:
                print("⚖️ Assessing damage severity...")
            
            # Prepare damage detection results
            damage_results = {
                'success': True,
                'damage_detections': state['current_result'].damage_detections,
                'total_damages': len(state['current_result'].damage_detections),
                'image_path': state['current_result'].image_path
            }
            
            # Prepare part identification results
            part_results = {
                'success': True,
                'damaged_parts': state['current_result'].damaged_parts,
                'total_parts_identified': len(state['current_result'].damaged_parts)
            }
            
            # Assess severity
            severity_result = self.severity_agent.assess_damage_severity(
                damage_detection_results=damage_results,
                part_identification_results=part_results
            )
            
            if severity_result.get('success', False):
                state['current_result'].severity_assessment = severity_result
                state['next_step'] = 'next_image'
            else:
                state['current_result'].errors.append("Severity assessment failed")
                state['current_result'].success = False
                state['next_step'] = 'error'
                
        except Exception as e:
            state['current_result'].errors.append(f"Severity assessment error: {str(e)}")
            state['current_result'].success = False
            state['next_step'] = 'error'
        
        return state
    
    def _consolidate_results_node(self, state: PipelineState) -> PipelineState:
        """Consolidate results from all processed images."""
        if self.enable_logging:
            print("📊 Consolidating multi-image results...")
        
        # Aggregate results from all images
        all_damages = []
        all_parts = []
        total_processing_time = 0
        confidence_scores = []
        
        for result in state['image_results']:
            all_damages.extend(result.damage_detections)
            all_parts.extend(result.damaged_parts)
            total_processing_time += result.processing_time
            if result.confidence_score > 0:
                confidence_scores.append(result.confidence_score)
        
        # Calculate overall metrics
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # Merge overlapping damage detections (simple approach)
        unique_damages = self._merge_similar_damages(all_damages)
        unique_parts = self._merge_similar_parts(all_parts)
        
        # Determine overall severity
        overall_severity = self._calculate_overall_severity(state['image_results'])
        
        state['consolidated_assessment'] = {
            'total_images_processed': len([r for r in state['image_results'] if r.success]),
            'total_damages_detected': len(unique_damages),
            'total_parts_affected': len(unique_parts),
            'overall_confidence': avg_confidence,
            'overall_severity': overall_severity,
            'consolidated_damages': unique_damages,
            'consolidated_parts': unique_parts,
            'processing_metadata': {
                'total_processing_time': total_processing_time,
                'images_with_errors': len([r for r in state['image_results'] if not r.success]),
                'confidence_threshold_used': state['confidence_threshold'],
                'timestamp': datetime.now().isoformat()
            }
        }
        
        return state
    
    def _handle_error_node(self, state: PipelineState) -> PipelineState:
        """Handle errors by stopping workflow gracefully (fail-fast approach)."""
        if self.enable_logging:
            print(f"❌ Agent failure detected for image {state['current_image_index'] + 1}")
            print(f"❌ Stopping workflow gracefully (fail-fast mode)...")
        
        # Add current result to failed images
        current_image = state['images'][state['current_image_index']]
        state['failed_images'].append(current_image)
        state['errors'].extend(state['current_result'].errors)
        
        # Set workflow termination flags
        state['workflow_terminated_due_to_error'] = True
        state['termination_reason'] = f"Agent failure on image: {current_image}. Errors: {', '.join(state['current_result'].errors)}"
        
        # Go directly to finalize to stop workflow immediately
        state['next_step'] = 'finalize'
        
        return state
    
    def _finalize_pipeline(self, state: PipelineState) -> PipelineState:
        """Finalize the pipeline and prepare final results."""
        if self.enable_logging:
            if state.get('workflow_terminated_due_to_error', False):
                print("❌ Pipeline terminated due to error")
                print(f"❌ Termination reason: {state.get('termination_reason', 'Unknown error')}")
            else:
                print("✅ Finalizing pipeline...")
        
        # If we have multiple successful images, consolidate (only if not terminated due to error)
        successful_results = [r for r in state['image_results'] if r.success]
        
        if len(successful_results) > 1 and not state['consolidated_assessment'] and not state.get('workflow_terminated_due_to_error', False):
            state = self._consolidate_results_node(state)
        
        # Prepare final metadata
        state['pipeline_metadata'] = {
            'total_images_submitted': len(state['images']),
            'successfully_processed': len(successful_results),
            'failed_images': len(state['failed_images']),
            'requires_manual_review': state.get('require_manual_review', False),
            'confidence_threshold': state['confidence_threshold'],
            'processing_completed_at': datetime.now().isoformat(),
            'workflow_terminated_due_to_error': state.get('workflow_terminated_due_to_error', False),
            'termination_reason': state.get('termination_reason', None)
        }
        
        return state
    
    # Routing functions
    def _route_after_initialize(self, state: PipelineState) -> str:
        """Route after initialization."""
        if state['current_image_index'] >= len(state['images']):
            # All images processed - decide whether to consolidate or finalize
            if len(state['image_results']) > 1:
                return 'consolidate'
            else:
                return 'finalize'
        return state['next_step']
    
    def _route_after_enhancement(self, state: PipelineState) -> str:
        """Route after image enhancement."""
        return state['next_step']
    
    def _route_after_damage_detection(self, state: PipelineState) -> str:
        """Route after damage detection."""
        next_step = state['next_step']
        
        # Handle the case when no damage is detected
        if next_step == 'next_image':
            # Save current result (no damage detected) and move to next image
            state['image_results'].append(state['current_result'])
            state['current_image_index'] += 1
            
            # Check if we've processed all images
            if state['current_image_index'] >= len(state['images']):
                return 'consolidate' if len(state['image_results']) > 1 else 'finalize'
            return 'initialize'
        
        return next_step
    
    def _route_after_part_identification(self, state: PipelineState) -> str:
        """Route after part identification."""
        return state['next_step']
    
    def _route_after_severity_assessment(self, state: PipelineState) -> str:
        """Route after severity assessment."""
        # Save current result and move to next image
        state['image_results'].append(state['current_result'])
        state['current_image_index'] += 1
        
        if state['current_image_index'] >= len(state['images']):
            return 'consolidate' if len(state['image_results']) > 1 else 'finalize'
        return 'initialize'
    
    def _route_after_error(self, state: PipelineState) -> str:
        """Route after error handling."""
        return state['next_step']
    
    # Helper methods
    def _merge_similar_damages(self, damages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge similar damage detections from multiple images."""
        # Simple implementation - in practice, you'd use more sophisticated merging
        unique_damages = []
        for damage in damages:
            # Check if similar damage already exists
            is_duplicate = False
            for existing in unique_damages:
                if (existing.get('type') == damage.get('type') and 
                    self._calculate_bbox_overlap(existing.get('bbox', []), damage.get('bbox', [])) > 0.5):
                    # Update confidence to higher value
                    existing['confidence'] = max(existing.get('confidence', 0), damage.get('confidence', 0))
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_damages.append(damage)
        
        return unique_damages
    
    def _merge_similar_parts(self, parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge similar part identifications from multiple images."""
        # Group by part name and merge damage percentages
        part_groups = {}
        for part in parts:
            part_name = part.get('part_name', 'unknown')
            if part_name not in part_groups:
                part_groups[part_name] = part.copy()
            else:
                # Merge damage percentages (take maximum)
                existing_damage = part_groups[part_name].get('damage_percentage', 0)
                new_damage = part.get('damage_percentage', 0)
                part_groups[part_name]['damage_percentage'] = max(existing_damage, new_damage)
                
                # Update confidence
                existing_conf = part_groups[part_name].get('confidence', 0)
                new_conf = part.get('confidence', 0)
                part_groups[part_name]['confidence'] = max(existing_conf, new_conf)
        
        return list(part_groups.values())
    
    def _calculate_bbox_overlap(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate overlap between two bounding boxes."""
        if len(bbox1) != 4 or len(bbox2) != 4:
            return 0.0
        
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_overall_severity(self, results: List[ImageProcessingResult]) -> Dict[str, Any]:
        """Calculate overall severity from multiple image results."""
        severity_scores = []
        severity_levels = []
        cost_estimates = []
        
        for result in results:
            if result.severity_assessment and result.success:
                assessment = result.severity_assessment.get('assessment', {})
                if 'severity_score' in assessment:
                    severity_scores.append(assessment['severity_score'])
                if 'overall_severity' in assessment:
                    severity_levels.append(assessment['overall_severity'])
                if 'cost_estimate' in assessment:
                    cost_est = assessment['cost_estimate']
                    if isinstance(cost_est, dict) and 'estimated_cost_range' in cost_est:
                        cost_estimates.append(cost_est['estimated_cost_range'])
        
        # Calculate overall metrics
        overall_score = max(severity_scores) if severity_scores else 0
        
        # Determine overall severity level
        severity_mapping = {'minor': 1, 'moderate': 2, 'severe': 3, 'total_loss': 4}
        max_severity_level = 'minor'
        max_severity_value = 0
        
        for level in severity_levels:
            if level in severity_mapping and severity_mapping[level] > max_severity_value:
                max_severity_value = severity_mapping[level]
                max_severity_level = level
        
        # Calculate cost range
        if cost_estimates:
            min_costs = [est[0] if isinstance(est, list) and len(est) >= 2 else est for est in cost_estimates]
            max_costs = [est[1] if isinstance(est, list) and len(est) >= 2 else est for est in cost_estimates]
            overall_cost_range = [min(min_costs), max(max_costs)]
        else:
            overall_cost_range = [0, 0]
        
        return {
            'overall_severity': max_severity_level,
            'severity_score': overall_score,
            'estimated_cost_range': overall_cost_range,
            'confidence': sum(severity_scores) / len(severity_scores) if severity_scores else 0
        }
    
    def process_claim(self, images: List[str], 
                     confidence_threshold: Optional[float] = None,
                     max_retries: Optional[int] = None) -> Dict[str, Any]:
        """
        Main method to process a damage claim with multiple images.
        
        Args:
            images: List of image file paths (must be in tests folder)
            confidence_threshold: Override default confidence threshold
            max_retries: Override default max retries
            
        Returns:
            Comprehensive assessment results
        """
        start_time = time.time()
        
        # Validate that all images are in tests folder
        tests_folder = Path("tests")
        validated_images = []
        
        for image_path in images:
            image_path_obj = Path(image_path)
            
            # Check if path is in tests folder
            if not image_path_obj.is_absolute():
                # For relative paths, check if they start with 'tests/'
                if not str(image_path).startswith('tests/'):
                    raise ValueError(f"Image must be in tests folder: {image_path}")
            else:
                # For absolute paths, check if they're within tests folder
                try:
                    image_path_obj.relative_to(tests_folder.absolute())
                except ValueError:
                    raise ValueError(f"Image must be in tests folder: {image_path}")
            
            validated_images.append(image_path)
        
        # Continue with validated images
        images = validated_images
        
        # Initialize state
        initial_state: PipelineState = {
            'images': images,
            'current_image_index': 0,
            'image_results': [],
            'current_result': None,
            'confidence_threshold': confidence_threshold or self.confidence_threshold,
            'retry_count': 0,
            'max_retries': max_retries or self.max_retries,
            'consolidated_assessment': None,
            'pipeline_metadata': {},
            'errors': [],
            'failed_images': [],
            'workflow_terminated_due_to_error': False,
            'termination_reason': '',
            'next_step': 'enhance',
            'skip_enhancement': False,
            'require_manual_review': False
        }
        
        if self.enable_logging:
            print(f"\n🚀 Starting damage assessment pipeline for {len(images)} images")
            print(f"📊 Confidence threshold: {initial_state['confidence_threshold']}")
            print(f"🔄 Max retries: {initial_state['max_retries']}")
        
        # Execute the workflow
        try:
            final_state = self.workflow.invoke(initial_state)
            
            # Calculate total processing time
            total_time = time.time() - start_time
            
            # Prepare final results
            results = {
                'success': True,
                'total_processing_time': total_time,
                'images_processed': len([r for r in final_state['image_results'] if r.success]),
                'images_failed': len(final_state['failed_images']),
                'individual_results': [r.dict() for r in final_state['image_results']],
                'consolidated_assessment': final_state.get('consolidated_assessment'),
                'pipeline_metadata': final_state['pipeline_metadata'],
                'errors': final_state['errors'],
                'failed_images': final_state['failed_images'],
                'requires_manual_review': final_state.get('require_manual_review', False)
            }
            
            if self.enable_logging:
                print(f"\n✅ Pipeline completed in {total_time:.2f}s")
                print(f"📊 Successfully processed: {results['images_processed']}/{len(images)} images")
                if results['images_failed'] > 0:
                    print(f"❌ Failed images: {results['images_failed']}")
            
            return results
            
        except Exception as e:
            error_msg = f"Pipeline execution failed: {str(e)}"
            if self.enable_logging:
                print(f"❌ {error_msg}")
            
            return {
                'success': False,
                'error': error_msg,
                'total_processing_time': time.time() - start_time,
                'images_processed': 0,
                'images_failed': len(images)
            }


def create_damage_assessment_orchestrator(openai_api_key: Optional[str] = None,
                                        ngrok_url: Optional[str] = None,
                                        confidence_threshold: float = 0.7,
                                        max_retries: int = 2,
                                        enable_logging: bool = True) -> DamageAssessmentOrchestrator:
    """Factory function to create a damage assessment orchestrator."""
    return DamageAssessmentOrchestrator(
        openai_api_key=openai_api_key,
        ngrok_url=ngrok_url,
        confidence_threshold=confidence_threshold,
        max_retries=max_retries,
        enable_logging=enable_logging
    )


if __name__ == "__main__":
    # Example usage
    print("🤖 LangGraph Damage Assessment Orchestrator")
    print("=" * 50)
    
    # This would be used in practice:
    # orchestrator = create_damage_assessment_orchestrator(
    #     openai_api_key="your-openai-key",
    #     ngrok_url="your-ngrok-url",
    #     confidence_threshold=0.75,
    #     enable_logging=True
    # )
    # 
    # results = orchestrator.process_claim([
    #     "path/to/image1.jpg",
    #     "path/to/image2.jpg"
    # ])
    # 
    # print(json.dumps(results, indent=2))
    
    print("Orchestrator implementation complete!")