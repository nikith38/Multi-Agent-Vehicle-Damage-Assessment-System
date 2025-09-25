"""Pydantic models for Severity Assessment Agent output structure.

This module defines the structured output format for vehicle damage severity assessment,
including severity classification, cost estimation, repair recommendations, and safety concerns.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class SeverityLevel(str, Enum):
    """Enumeration of damage severity levels"""
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    SEVERE = "severe"


class RepairCostEstimate(BaseModel):
    """Repair cost estimation with range and currency."""
    min_cost: float = Field(
        description="Minimum estimated repair cost",
        ge=0
    )
    max_cost: float = Field(
        description="Maximum estimated repair cost",
        ge=0
    )
    currency: str = Field(
        default="USD",
        description="Currency code for the cost estimate"
    )
    confidence_level: str = Field(
        description="Confidence level of the cost estimate (low/medium/high)"
    )


class SafetyConcern(BaseModel):
    """Individual safety concern with priority level."""
    concern: str = Field(
        description="Description of the safety concern"
    )
    priority: str = Field(
        description="Priority level of the safety concern (low/medium/high/critical)"
    )
    affected_systems: List[str] = Field(
        default_factory=list,
        description="List of vehicle systems affected by this safety concern"
    )


class RepairRecommendation(BaseModel):
    """Repair approach recommendation with details."""
    approach: str = Field(
        description="Primary repair approach (repair/replace/total_loss)"
    )
    description: str = Field(
        description="Detailed description of the recommended repair approach"
    )
    estimated_time: str = Field(
        description="Estimated time to complete repairs (e.g., '3-5 days')"
    )
    required_expertise: str = Field(
        description="Level of expertise required (basic/intermediate/advanced/specialist)"
    )
    parts_needed: List[str] = Field(
        default_factory=list,
        description="List of parts that may need replacement"
    )


class SeverityAssessment(BaseModel):
    """Complete severity assessment output structure."""
    
    # Core Assessment
    severity_classification: SeverityLevel = Field(
        description="Overall damage severity classification"
    )
    
    severity_score: float = Field(
        description="Numerical severity score from 0-100",
        ge=0,
        le=100
    )
    
    # Cost Analysis
    repair_cost_estimate: RepairCostEstimate = Field(
        description="Estimated repair costs with range and confidence"
    )
    
    # Repair Recommendations
    repair_recommendation: RepairRecommendation = Field(
        description="Recommended repair approach and details"
    )
    
    # Safety Analysis
    safety_concerns: List[SafetyConcern] = Field(
        default_factory=list,
        description="List of identified safety concerns"
    )
    
    has_safety_issues: bool = Field(
        description="Whether any safety-critical issues were identified"
    )
    
    # Explanations
    severity_explanation: str = Field(
        description="Detailed explanation of why this severity level was assigned"
    )
    
    cost_breakdown: str = Field(
        description="Breakdown of how repair costs were estimated"
    )
    
    # Metadata
    assessment_confidence: str = Field(
        description="Overall confidence in the assessment (low/medium/high)"
    )
    
    factors_considered: List[str] = Field(
        description="List of key factors that influenced the assessment"
    )
    
    additional_notes: Optional[str] = Field(
        default=None,
        description="Any additional notes or considerations"
    )
    
    # Processing Info
    total_damages_analyzed: int = Field(
        description="Number of individual damages analyzed"
    )
    
    total_parts_affected: int = Field(
        description="Number of vehicle parts affected by damage"
    )
    
    processing_time: float = Field(
        description="Time taken to complete the assessment in seconds"
    )