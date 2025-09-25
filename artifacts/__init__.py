"""Artifacts package for the Multi-Agent Vehicle Damage Assessment System.

This package contains the core components, pipeline processors, and user interfaces
for the vehicle damage assessment system.

Structure:
- core/: Core orchestration components (orchestrator)
- pipeline/: Pipeline processing components (batch processor, run pipeline)
- ui/: User interface components (streamlit app)
- docs/: Documentation files
"""

# Import main components for easy access
from .core import DamageAssessmentOrchestrator
from .pipeline import BatchProcessor

__all__ = [
    'DamageAssessmentOrchestrator',
    'BatchProcessor'
]