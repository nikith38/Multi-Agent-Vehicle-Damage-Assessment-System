"""Pydantic models for image processing and enhancement tools.

This module defines the structured input/output formats for image quality assessment
and enhancement operations used throughout the system.
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field


class ImageMetricsInput(BaseModel):
    """Input model for image metrics assessment"""
    image_path: str = Field(description="Path to the image file")
    region_analysis: bool = Field(default=True, description="Enable multi-region analysis")
    adaptive_thresholds: bool = Field(default=True, description="Use adaptive quality thresholds")


class QualityMetrics(BaseModel):
    """Individual quality metrics"""
    blur_score: float = Field(description="Blur assessment score (higher = sharper)")
    brightness_score: float = Field(description="Brightness level (0-255)")
    contrast_score: float = Field(description="Contrast measurement (0-1)")
    noise_level: float = Field(description="Noise level (lower = cleaner)")
    overall_score: float = Field(description="Overall quality score (0-1)")


class RegionMetrics(BaseModel):
    """Quality metrics for specific image regions"""
    center: QualityMetrics = Field(description="Center region metrics")
    edges: QualityMetrics = Field(description="Edge regions metrics")
    corners: QualityMetrics = Field(description="Corner regions metrics")
    weighted_average: QualityMetrics = Field(description="Weighted average across regions")


class ImageMetricsOutput(BaseModel):
    """Output model for image metrics assessment"""
    is_processable: bool = Field(description="Whether image meets quality standards")
    overall_metrics: QualityMetrics = Field(description="Overall image quality metrics")
    region_metrics: Optional[RegionMetrics] = Field(description="Region-specific metrics if enabled")
    quality_issues: List[str] = Field(description="List of detected quality issues")
    recommendations: List[str] = Field(description="Recommended enhancement actions")
    processing_time: float = Field(description="Time taken for assessment (seconds)")
    image_properties: Dict[str, Any] = Field(description="Basic image properties")


class SharpenImageInput(BaseModel):
    """Input model for image sharpening tool"""
    image_path: str = Field(description="Path to the input image file")
    output_path: Optional[str] = Field(default=None, description="Path for output image (optional, defaults to input_sharpened.jpg)")
    method: Literal["unsharp_mask", "laplacian", "high_pass"] = Field(
        default="unsharp_mask", 
        description="Sharpening method: unsharp_mask (gentle), laplacian (moderate), high_pass (strong)"
    )
    strength: float = Field(
        default=1.0, 
        ge=0.1, 
        le=3.0, 
        description="Sharpening strength (0.1-3.0, where 1.0 is normal)"
    )
    radius: float = Field(
        default=1.0, 
        ge=0.5, 
        le=5.0, 
        description="Sharpening radius in pixels (0.5-5.0)"
    )
    threshold: int = Field(
        default=0, 
        ge=0, 
        le=255, 
        description="Threshold for edge detection (0-255, 0 means all pixels)"
    )


class DenoiseImageInput(BaseModel):
    """Input model for image denoising tool"""
    image_path: str = Field(description="Path to the input image file")
    output_path: Optional[str] = Field(default=None, description="Path for output image (optional, defaults to input_denoised.jpg)")
    method: Literal["gaussian", "bilateral", "non_local_means", "median"] = Field(
        default="bilateral", 
        description="Denoising method: gaussian (fast), bilateral (edge-preserving), non_local_means (best quality), median (salt-pepper noise)"
    )
    strength: float = Field(
        default=1.0, 
        ge=0.1, 
        le=3.0, 
        description="Denoising strength (0.1-3.0, where 1.0 is normal)"
    )
    preserve_edges: bool = Field(
        default=True, 
        description="Whether to preserve edges during denoising"
    )
    kernel_size: int = Field(
        default=5, 
        ge=3, 
        le=15, 
        description="Kernel size for filtering (3-15, must be odd)"
    )


class AdjustExposureInput(BaseModel):
    """Input model for exposure adjustment tool"""
    image_path: str = Field(description="Path to the input image file")
    output_path: Optional[str] = Field(default=None, description="Path for output image (optional, defaults to input_exposure.jpg)")
    exposure_compensation: float = Field(
        default=0.0, 
        ge=-2.0, 
        le=2.0, 
        description="Exposure compensation in stops (-2.0 to +2.0, where 0 is no change)"
    )
    gamma_correction: float = Field(
        default=1.0, 
        ge=0.3, 
        le=3.0, 
        description="Gamma correction (0.3-3.0, where 1.0 is no change, <1 brightens, >1 darkens)"
    )
    auto_adjust: bool = Field(
        default=False, 
        description="Whether to automatically adjust exposure based on histogram analysis"
    )
    preserve_highlights: bool = Field(
        default=True, 
        description="Whether to preserve highlight details during adjustment"
    )
    preserve_shadows: bool = Field(
        default=True, 
        description="Whether to preserve shadow details during adjustment"
    )


class EnhanceContrastInput(BaseModel):
    """Input model for contrast enhancement tool"""
    image_path: str = Field(description="Path to the input image file")
    output_path: Optional[str] = Field(default=None, description="Path for output image (optional, defaults to input_contrast.jpg)")
    method: Literal["histogram_equalization", "adaptive_histogram", "linear_stretch", "gamma_correction"] = Field(
        default="adaptive_histogram", 
        description="Contrast enhancement method: histogram_equalization (global), adaptive_histogram (local), linear_stretch (simple), gamma_correction (tone curve)"
    )
    strength: float = Field(
        default=1.0, 
        ge=0.1, 
        le=3.0, 
        description="Enhancement strength (0.1-3.0, where 1.0 is normal)"
    )
    clip_limit: float = Field(
        default=2.0, 
        ge=1.0, 
        le=10.0, 
        description="Clipping limit for adaptive histogram equalization (1.0-10.0)"
    )
    tile_grid_size: int = Field(
        default=8, 
        ge=4, 
        le=16, 
        description="Grid size for adaptive histogram equalization (4-16)"
    )
    preserve_color: bool = Field(
        default=True, 
        description="Whether to preserve original color relationships"
    )


class ColorResolutionInput(BaseModel):
    """Input model for color and resolution enhancement tool"""
    image_path: str = Field(description="Path to the input image file")
    output_path: Optional[str] = Field(default=None, description="Path for output image (optional, defaults to input_enhanced.jpg)")
    color_enhancement: Literal["none", "saturation", "vibrance", "color_balance", "auto_color"] = Field(
        default="vibrance", 
        description="Color enhancement method: none, saturation (uniform boost), vibrance (selective boost), color_balance (white balance), auto_color (automatic correction)"
    )
    saturation_factor: float = Field(
        default=1.2, 
        ge=0.5, 
        le=2.0, 
        description="Saturation enhancement factor (0.5-2.0, where 1.0 is no change)"
    )
    resolution_enhancement: Literal["none", "bicubic", "lanczos", "super_resolution"] = Field(
        default="none", 
        description="Resolution enhancement method: none, bicubic (standard), lanczos (high quality), super_resolution (AI-based)"
    )
    scale_factor: float = Field(
        default=1.0, 
        ge=1.0, 
        le=4.0, 
        description="Scale factor for resolution enhancement (1.0-4.0)"
    )
    white_balance_correction: bool = Field(
        default=False, 
        description="Whether to apply automatic white balance correction"
    )
    noise_reduction: bool = Field(
        default=False,
        description="Whether to apply light noise reduction during enhancement"
    )