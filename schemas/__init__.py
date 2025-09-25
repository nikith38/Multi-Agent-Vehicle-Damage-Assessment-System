"""Schemas package for Multi-Agent Vehicle Damage Assessment System.

This package contains all Pydantic models used throughout the system for data validation
and serialization. Models are organized by their functional domain.
"""

# Severity Assessment Schemas
from .severity import (
    SeverityLevel,
    RepairCostEstimate,
    SafetyConcern,
    RepairRecommendation,
    SeverityAssessment
)

# Image Processing Schemas
from .image_processing import (
    ImageMetricsInput,
    QualityMetrics,
    RegionMetrics,
    ImageMetricsOutput,
    SharpenImageInput,
    DenoiseImageInput,
    AdjustExposureInput,
    EnhanceContrastInput,
    ColorResolutionInput
)

# Orchestrator Schemas
from .orchestrator import (
    ImageProcessingResult
)

__all__ = [
    # Severity Assessment
    'SeverityLevel',
    'RepairCostEstimate',
    'SafetyConcern',
    'RepairRecommendation',
    'SeverityAssessment',
    
    # Image Processing
    'ImageMetricsInput',
    'QualityMetrics',
    'RegionMetrics',
    'ImageMetricsOutput',
    'SharpenImageInput',
    'DenoiseImageInput',
    'AdjustExposureInput',
    'EnhanceContrastInput',
    'ColorResolutionInput',
    
    # Orchestrator
    'ImageProcessingResult',
]