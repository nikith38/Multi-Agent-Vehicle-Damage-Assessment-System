"""Pydantic models for orchestrator and pipeline management.

This module defines the structured data formats for managing the vehicle damage
assessment pipeline state and results.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


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