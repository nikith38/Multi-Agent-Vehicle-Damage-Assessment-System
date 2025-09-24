"""Image Quality Assessment Tools for Vehicle Damage Detection System

This module provides computer vision tools for assessing and enhancing image quality
using classic CV techniques, decorated with LangChain @tool for ReACT agent integration.
"""

import cv2
import numpy as np
import asyncio
import concurrent.futures
import functools
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Literal
from pydantic import BaseModel, Field
from langchain.tools import tool
import json
import time
import pywt
import logging
from scipy import ndimage
from PIL import Image, ImageEnhance, ImageFilter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class ImageQualityAssessor:
    """Enhanced image quality assessment with robustness improvements and async processing"""
    
    def __init__(self):
        self.quality_thresholds = {
            'blur_min': 100.0,  # Laplacian variance threshold
            'brightness_min': 30.0,
            'brightness_max': 225.0,
            'contrast_min': 0.15,  # Michelson contrast
            'noise_max': 0.3,
            'overall_min': 0.6
        }
        # Thread pool for CPU-intensive operations
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    
    def _load_and_validate_image(self, image_path: str) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Load and validate image with comprehensive error handling"""
        try:
            # Check if file exists
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Basic properties
            height, width = image.shape[:2]
            channels = image.shape[2] if len(image.shape) == 3 else 1
            
            properties = {
                'width': width,
                'height': height,
                'channels': channels,
                'total_pixels': width * height,
                'aspect_ratio': width / height,
                'file_size': Path(image_path).stat().st_size
            }
            
            # Validate minimum requirements
            if width < 100 or height < 100:
                raise ValueError(f"Image too small: {width}x{height} (minimum 100x100)")
            
            return image, properties
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return None, {'error': str(e)}
    
    def _assess_blur_laplacian(self, image: np.ndarray) -> float:
        """Robust blur detection using Laplacian variance"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()
            return float(variance)
        except Exception as e:
            logger.warning(f"Blur assessment failed: {e}")
            return 0.0
    
    async def _assess_blur_laplacian_async(self, image: np.ndarray) -> float:
        """Async wrapper for blur assessment"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._assess_blur_laplacian, image)
    
    def _assess_brightness(self, image: np.ndarray) -> float:
        """Robust brightness assessment with edge case handling"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Handle extreme cases
            if np.all(gray == 0):
                return 0.0  # Completely black image
            if np.all(gray == 255):
                return 255.0  # Completely white image
            
            # Use mean brightness
            brightness = np.mean(gray)
            return float(brightness)
        except Exception as e:
            logger.warning(f"Brightness assessment failed: {e}")
            return 0.0
    
    async def _assess_brightness_async(self, image: np.ndarray) -> float:
        """Async wrapper for brightness assessment"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._assess_brightness, image)
    
    def _assess_contrast_michelson(self, image: np.ndarray) -> float:
        """Enhanced contrast using Michelson contrast formula"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Handle edge cases
            if np.all(gray == gray[0, 0]):
                return 0.0  # Uniform image
            
            # Michelson contrast: (Imax - Imin) / (Imax + Imin)
            i_max = np.max(gray)
            i_min = np.min(gray)
            
            if i_max + i_min == 0:
                return 0.0
            
            michelson_contrast = (i_max - i_min) / (i_max + i_min)
            return float(michelson_contrast)
        except Exception as e:
            logger.warning(f"Contrast assessment failed: {e}")
            return 0.0
    
    async def _assess_contrast_michelson_async(self, image: np.ndarray) -> float:
        """Async wrapper for contrast assessment"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._assess_contrast_michelson, image)
    
    def _assess_noise_multiscale(self, image: np.ndarray) -> float:
        """Multi-scale noise detection combining wavelet and Laplacian methods"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Primary method: Wavelet-based noise estimation
            wavelet_score = self._assess_noise_level_wavelet(gray)
            
            # Secondary method: Laplacian variance (edge-based noise detection)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian_var = laplacian.var()
            laplacian_score = min(1.0, laplacian_var / 500.0)  # Typical range 0-500
            
            # Tertiary method: Local patch analysis
            patch_size = 8
            h, w = gray.shape
            local_stds = []
            
            for i in range(0, h - patch_size, patch_size):
                for j in range(0, w - patch_size, patch_size):
                    patch = gray[i:i+patch_size, j:j+patch_size]
                    local_stds.append(np.std(patch))
            
            local_std_var = np.var(local_stds) if local_stds else 0
            texture_score = min(1.0, local_std_var / 100.0)  # Typical range 0-100
            
            # Weighted combination (wavelet gets highest weight as it's most robust)
            combined_score = (
                wavelet_score * 0.6 +     # Wavelet-based estimation (most reliable)
                laplacian_score * 0.3 +   # Edge preservation
                texture_score * 0.1       # Texture consistency
            )
            
            # Convert to noise level (invert the score)
            noise_level = 1.0 - max(0.0, min(1.0, combined_score))
            
            return float(noise_level)
        except Exception as e:
            logger.warning(f"Noise assessment failed: {e}")
            return 0.5  # Default moderate noise level
    
    def _assess_noise_level_wavelet(self, gray_image: np.ndarray) -> float:
        """Assess noise level using wavelet-based estimation.
        
        Args:
            gray_image: Grayscale input image
            
        Returns:
            Noise score (0-1, higher means less noise)
        """
        try:
            # Method 1: Single-level robust estimation
            coeffs = pywt.dwt2(gray_image, 'db1')
            cA, (cH, cV, cD) = coeffs
            
            # Robust noise estimation using diagonal detail coefficients
            # Diagonal coefficients are less affected by image structure
            noise_sigma = np.median(np.abs(cD)) / 0.6745
            
            # Method 2: Multi-orientation analysis
            noise_h = np.median(np.abs(cH)) / 0.6745
            noise_v = np.median(np.abs(cV)) / 0.6745
            
            # Combine estimates (diagonal is most reliable)
            combined_noise = (noise_sigma * 0.6 + noise_h * 0.2 + noise_v * 0.2)
            
            # Normalize for vehicle images (empirically determined)
            # Clean vehicle images: sigma ~ 2-8
            # Noisy vehicle images: sigma ~ 15-30+
            noise_score = max(0.0, min(1.0, (12.0 - combined_noise) / 10.0))
            
            return noise_score
            
        except Exception:
            # Fallback to simple standard deviation if wavelet fails
            return max(0.0, min(1.0, np.std(gray_image) / 50.0))
    
    async def _assess_noise_multiscale_async(self, image: np.ndarray) -> float:
        """Async wrapper for noise assessment"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._assess_noise_multiscale, image)
    
    def _get_adaptive_thresholds(self, image: np.ndarray, properties: Dict[str, Any]) -> Dict[str, float]:
        """Adaptive thresholds based on image context"""
        thresholds = self.quality_thresholds.copy()
        
        try:
            # Analyze image characteristics
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            mean_brightness = np.mean(gray)
            
            # Adjust for low-light conditions (night images, dark vehicles)
            if mean_brightness < 50:
                thresholds['brightness_min'] = 10.0  # More lenient for dark images
                thresholds['contrast_min'] = 0.08   # Lower contrast acceptable
                thresholds['noise_max'] = 0.4       # Allow more noise in low light
            
            # Adjust for high-brightness conditions
            elif mean_brightness > 200:
                thresholds['brightness_max'] = 240.0
                thresholds['contrast_min'] = 0.1     # Bright images may have lower contrast
            
            # Adjust for image size
            total_pixels = properties.get('total_pixels', 0)
            if total_pixels < 500000:  # Small images (< 0.5MP)
                thresholds['blur_min'] = 50.0       # More lenient blur threshold
            
            return thresholds
        except Exception as e:
            logger.warning(f"Adaptive threshold calculation failed: {e}")
            return self.quality_thresholds
    
    def _assess_region_quality(self, image: np.ndarray, region_mask: np.ndarray) -> QualityMetrics:
        """Assess quality for a specific image region"""
        try:
            # Apply mask to extract region
            masked_image = cv2.bitwise_and(image, image, mask=region_mask)
            
            # Only assess non-zero pixels
            if np.sum(region_mask) == 0:
                return QualityMetrics(
                    blur_score=0.0, brightness_score=0.0, 
                    contrast_score=0.0, noise_level=1.0, overall_score=0.0
                )
            
            blur = self._assess_blur_laplacian(masked_image)
            brightness = self._assess_brightness(masked_image)
            contrast = self._assess_contrast_michelson(masked_image)
            noise = self._assess_noise_multiscale(masked_image)
            
            # Calculate region-specific overall score
            overall = self._calculate_overall_score(blur, brightness, contrast, noise)
            
            return QualityMetrics(
                blur_score=blur,
                brightness_score=brightness,
                contrast_score=contrast,
                noise_level=noise,
                overall_score=overall
            )
        except Exception as e:
            logger.warning(f"Region quality assessment failed: {e}")
            return QualityMetrics(
                blur_score=0.0, brightness_score=0.0,
                contrast_score=0.0, noise_level=1.0, overall_score=0.0
            )
    
    async def _assess_region_quality_async(self, image: np.ndarray, region_mask: np.ndarray) -> QualityMetrics:
        """Async version of region quality assessment"""
        try:
            # Apply mask to extract region
            masked_image = cv2.bitwise_and(image, image, mask=region_mask)
            
            # Only assess non-zero pixels
            if np.sum(region_mask) == 0:
                return QualityMetrics(
                    blur_score=0.0, brightness_score=0.0, 
                    contrast_score=0.0, noise_level=1.0, overall_score=0.0
                )
            
            # Run all assessments concurrently
            blur_task = self._assess_blur_laplacian_async(masked_image)
            brightness_task = self._assess_brightness_async(masked_image)
            contrast_task = self._assess_contrast_michelson_async(masked_image)
            noise_task = self._assess_noise_multiscale_async(masked_image)
            
            blur, brightness, contrast, noise = await asyncio.gather(
                blur_task, brightness_task, contrast_task, noise_task
            )
            
            # Calculate region-specific overall score
            overall = self._calculate_overall_score(blur, brightness, contrast, noise)
            
            return QualityMetrics(
                blur_score=blur,
                brightness_score=brightness,
                contrast_score=contrast,
                noise_level=noise,
                overall_score=overall
            )
        except Exception as e:
            logger.warning(f"Async region quality assessment failed: {e}")
            return QualityMetrics(
                blur_score=0.0, brightness_score=0.0,
                contrast_score=0.0, noise_level=1.0, overall_score=0.0
            )
    
    async def _assess_overall_quality_async(self, image: np.ndarray) -> Tuple[float, float, float, float]:
        """Assess overall image quality metrics concurrently"""
        try:
            # Run all assessments concurrently for maximum speed
            blur_task = self._assess_blur_laplacian_async(image)
            brightness_task = self._assess_brightness_async(image)
            contrast_task = self._assess_contrast_michelson_async(image)
            noise_task = self._assess_noise_multiscale_async(image)
            
            blur, brightness, contrast, noise = await asyncio.gather(
                blur_task, brightness_task, contrast_task, noise_task
            )
            
            return blur, brightness, contrast, noise
        except Exception as e:
            logger.warning(f"Async overall quality assessment failed: {e}")
            return 0.0, 0.0, 0.0, 1.0
    
    def _create_region_masks(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Create masks for different image regions"""
        height, width = image.shape[:2]
        
        # Center region (40% of image)
        center_h, center_w = int(height * 0.3), int(width * 0.3)
        center_mask = np.zeros((height, width), dtype=np.uint8)
        center_mask[center_h:height-center_h, center_w:width-center_w] = 255
        
        # Edge regions (outer 20% border)
        edge_mask = np.ones((height, width), dtype=np.uint8) * 255
        edge_border = min(int(height * 0.2), int(width * 0.2))
        edge_mask[edge_border:height-edge_border, edge_border:width-edge_border] = 0
        
        # Corner regions
        corner_size = min(int(height * 0.15), int(width * 0.15))
        corner_mask = np.zeros((height, width), dtype=np.uint8)
        # Top-left and bottom-right corners
        corner_mask[0:corner_size, 0:corner_size] = 255
        corner_mask[height-corner_size:height, width-corner_size:width] = 255
        
        return {
            'center': center_mask,
            'edges': edge_mask,
            'corners': corner_mask
        }
    
    def _calculate_overall_score(self, blur: float, brightness: float, 
                               contrast: float, noise: float, 
                               thresholds: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted overall quality score"""
        if thresholds is None:
            thresholds = self.quality_thresholds
        
        try:
            # Normalize each metric to 0-1 scale
            blur_norm = min(blur / thresholds['blur_min'], 1.0)
            
            # Brightness: penalize values outside optimal range
            brightness_norm = 1.0
            if brightness < thresholds['brightness_min']:
                brightness_norm = brightness / thresholds['brightness_min']
            elif brightness > thresholds['brightness_max']:
                brightness_norm = (255 - brightness) / (255 - thresholds['brightness_max'])
            
            contrast_norm = min(contrast / thresholds['contrast_min'], 1.0)
            noise_norm = max(1.0 - (noise / thresholds['noise_max']), 0.0)
            
            # Weighted combination (blur and contrast are most important for damage detection)
            weights = {'blur': 0.35, 'brightness': 0.2, 'contrast': 0.3, 'noise': 0.15}
            
            overall = (
                weights['blur'] * blur_norm +
                weights['brightness'] * brightness_norm +
                weights['contrast'] * contrast_norm +
                weights['noise'] * noise_norm
            )
            
            return float(np.clip(overall, 0.0, 1.0))
        except Exception as e:
            logger.warning(f"Overall score calculation failed: {e}")
            return 0.0
    
    def _identify_quality_issues(self, metrics: QualityMetrics, 
                               thresholds: Dict[str, float]) -> List[str]:
        """Identify specific quality issues"""
        issues = []
        
        if metrics.blur_score < thresholds['blur_min']:
            issues.append(f"Image is blurry (score: {metrics.blur_score:.1f}, min: {thresholds['blur_min']})")
        
        if metrics.brightness_score < thresholds['brightness_min']:
            issues.append(f"Image is too dark (brightness: {metrics.brightness_score:.1f})")
        elif metrics.brightness_score > thresholds['brightness_max']:
            issues.append(f"Image is too bright (brightness: {metrics.brightness_score:.1f})")
        
        if metrics.contrast_score < thresholds['contrast_min']:
            issues.append(f"Low contrast (score: {metrics.contrast_score:.3f}, min: {thresholds['contrast_min']})")
        
        if metrics.noise_level > thresholds['noise_max']:
            issues.append(f"High noise level (score: {metrics.noise_level:.3f}, max: {thresholds['noise_max']})")
        
        if metrics.overall_score < thresholds['overall_min']:
            issues.append(f"Overall quality insufficient (score: {metrics.overall_score:.3f})")
        
        return issues
    
    def _generate_recommendations(self, issues: List[str]) -> List[str]:
        """Generate enhancement recommendations based on issues"""
        recommendations = []
        
        for issue in issues:
            if "blurry" in issue.lower():
                recommendations.append("Apply unsharp masking or edge enhancement")
            elif "dark" in issue.lower():
                recommendations.append("Increase brightness using gamma correction or CLAHE")
            elif "bright" in issue.lower():
                recommendations.append("Reduce brightness and enhance shadows")
            elif "contrast" in issue.lower():
                recommendations.append("Apply histogram equalization or CLAHE")
            elif "noise" in issue.lower():
                recommendations.append("Apply noise reduction filtering")
        
        if not recommendations:
            recommendations.append("Image quality is acceptable for processing")
        
        return recommendations


async def _assess_image_metrics_async(input_model: ImageMetricsInput) -> ImageMetricsOutput:
    """Async implementation of image metrics assessment for optimal performance"""
    import time
    
    start_time = time.time()
    
    # Initialize assessor
    assessor = ImageQualityAssessor()
    
    # Load and validate image
    image, properties = assessor._load_and_validate_image(input_model.image_path)
    
    if image is None:
        return ImageMetricsOutput(
            is_processable=False,
            overall_metrics=QualityMetrics(
                blur_score=0.0, brightness_score=0.0,
                contrast_score=0.0, noise_level=1.0, overall_score=0.0
            ),
            region_metrics=None,
            quality_issues=[f"Failed to load image: {properties.get('error', 'Unknown error')}"],
            recommendations=["Verify image file exists and is in supported format"],
            processing_time=time.time() - start_time,
            image_properties=properties
        )
    
    # Get adaptive thresholds if enabled
    thresholds = (
        assessor._get_adaptive_thresholds(image, properties) 
        if input_model.adaptive_thresholds 
        else assessor.quality_thresholds
    )
    
    # Assess overall image quality concurrently
    blur, brightness, contrast, noise = await assessor._assess_overall_quality_async(image)
    
    # Calculate overall score
    overall_score = assessor._calculate_overall_score(blur, brightness, contrast, noise, thresholds)
    
    overall_metrics = QualityMetrics(
        blur_score=blur,
        brightness_score=brightness,
        contrast_score=contrast,
        noise_level=noise,
        overall_score=overall_score
    )
    
    # Region analysis if enabled (concurrent processing)
    region_metrics = None
    if input_model.region_analysis:
        try:
            masks = assessor._create_region_masks(image)
            
            # Run region assessments concurrently
            center_task = assessor._assess_region_quality_async(image, masks['center'])
            edge_task = assessor._assess_region_quality_async(image, masks['edges'])
            corner_task = assessor._assess_region_quality_async(image, masks['corners'])
            
            center_metrics, edge_metrics, corner_metrics = await asyncio.gather(
                center_task, edge_task, corner_task
            )
            
            # Calculate weighted average (center region is most important)
            weights = {'center': 0.5, 'edges': 0.3, 'corners': 0.2}
            weighted_blur = (weights['center'] * center_metrics.blur_score + 
                           weights['edges'] * edge_metrics.blur_score + 
                           weights['corners'] * corner_metrics.blur_score)
            weighted_brightness = (weights['center'] * center_metrics.brightness_score + 
                                 weights['edges'] * edge_metrics.brightness_score + 
                                 weights['corners'] * corner_metrics.brightness_score)
            weighted_contrast = (weights['center'] * center_metrics.contrast_score + 
                               weights['edges'] * edge_metrics.contrast_score + 
                               weights['corners'] * corner_metrics.contrast_score)
            weighted_noise = (weights['center'] * center_metrics.noise_level + 
                            weights['edges'] * edge_metrics.noise_level + 
                            weights['corners'] * corner_metrics.noise_level)
            weighted_overall = (weights['center'] * center_metrics.overall_score + 
                              weights['edges'] * edge_metrics.overall_score + 
                              weights['corners'] * corner_metrics.overall_score)
            
            weighted_average = QualityMetrics(
                blur_score=weighted_blur,
                brightness_score=weighted_brightness,
                contrast_score=weighted_contrast,
                noise_level=weighted_noise,
                overall_score=weighted_overall
            )
            
            region_metrics = RegionMetrics(
                center=center_metrics,
                edges=edge_metrics,
                corners=corner_metrics,
                weighted_average=weighted_average
            )
            
            # Use weighted average for final assessment
            overall_metrics = weighted_average
            
        except Exception as e:
            logger.warning(f"Async region analysis failed: {e}")
    
    # Identify issues and generate recommendations
    quality_issues = assessor._identify_quality_issues(overall_metrics, thresholds)
    recommendations = assessor._generate_recommendations(quality_issues)
    
    # Determine if image is processable
    is_processable = overall_metrics.overall_score >= thresholds['overall_min']
    
    # Create output
    return ImageMetricsOutput(
        is_processable=is_processable,
        overall_metrics=overall_metrics,
        region_metrics=region_metrics,
        quality_issues=quality_issues,
        recommendations=recommendations,
        processing_time=time.time() - start_time,
        image_properties=properties
    )


@tool
def assess_image_metrics(input_data: str) -> str:
    """Assess comprehensive image quality metrics for vehicle damage detection with async optimization.
    
    This tool evaluates image quality across multiple dimensions including blur,
    brightness, contrast, and noise levels. It provides both overall and region-specific
    analysis with adaptive thresholds based on image characteristics.
    
    Optimized with async processing for minimum latency - all quality metrics
    are assessed concurrently using thread pools for CPU-intensive operations.
    
    Args:
        input_data: JSON string containing ImageMetricsInput parameters
        
    Returns:
        JSON string containing ImageMetricsOutput with quality assessment results
    """
    import json
    import time
    
    start_time = time.time()
    
    try:
        # Clean input data - remove quotes, backticks and whitespace that agent might add
        cleaned_input = input_data.strip()
        
        # Remove backticks if present
        if cleaned_input.startswith('`') and cleaned_input.endswith('`'):
            cleaned_input = cleaned_input[1:-1]
        
        # Remove single quotes if present (agent sometimes wraps JSON in single quotes)
        if cleaned_input.startswith("'") and cleaned_input.endswith("'"):
            cleaned_input = cleaned_input[1:-1]
        
        cleaned_input = cleaned_input.strip()  # Remove any remaining whitespace
        
        # Parse input
        input_dict = json.loads(cleaned_input)
        input_model = ImageMetricsInput(**input_dict)
        
        # Run async assessment
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create a new thread
                import threading
                result = None
                exception = None
                
                def run_async():
                    nonlocal result, exception
                    try:
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        result = new_loop.run_until_complete(_assess_image_metrics_async(input_model))
                        new_loop.close()
                    except Exception as e:
                        exception = e
                
                thread = threading.Thread(target=run_async)
                thread.start()
                thread.join()
                
                if exception:
                    raise exception
                output = result
            else:
                # Use existing loop
                output = loop.run_until_complete(_assess_image_metrics_async(input_model))
        except RuntimeError:
            # No event loop, create one
            output = asyncio.run(_assess_image_metrics_async(input_model))
        
        return output.model_dump_json()
        
    except Exception as e:
        logger.error(f"Image metrics assessment failed: {str(e)}")
        return json.dumps({
            "is_processable": False,
            "overall_metrics": {
                "blur_score": 0.0, "brightness_score": 0.0,
                "contrast_score": 0.0, "noise_level": 1.0, "overall_score": 0.0
            },
            "region_metrics": None,
            "quality_issues": [f"Assessment failed: {str(e)}"],
            "recommendations": ["Check image file and try again"],
            "processing_time": time.time() - start_time,
            "image_properties": {"error": str(e)}
        })


# Additional Pydantic models for image processing tools
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


@tool
def sharpen_image(input_data: str) -> Dict[str, Any]:
    """
    Sharpens an image using various algorithms to improve edge definition and clarity.
    Addresses blur and sharpness issues detected by image quality assessment.
    
    Args:
        input_data: JSON string containing image path, output path, sharpening method, strength, radius, and threshold
    
    Returns:
        Dict containing success status, output path, applied settings, and quality metrics
    """
    try:
        # Clean and parse input data
        import json
        cleaned_input = input_data.strip().strip('`').strip("'").strip('"').strip()
        parsed_data = json.loads(cleaned_input)
        input_obj = SharpenImageInput(**parsed_data)
        
        # Validate input file
        if not os.path.exists(input_obj.image_path):
            return {"success": False, "error": f"Input image not found: {input_obj.image_path}"}
        
        # Set output path
        if input_obj.output_path is None:
            base, ext = os.path.splitext(input_obj.image_path)
            input_obj.output_path = f"{base}_sharpened{ext}"
        
        # Load image
        image = cv2.imread(input_obj.image_path)
        if image is None:
            return {"success": False, "error": "Failed to load image"}
        
        # Apply sharpening based on method
        if input_obj.method == "unsharp_mask":
            # Unsharp masking - gentle sharpening
            gaussian = cv2.GaussianBlur(image, (0, 0), input_obj.radius)
            unsharp_mask = cv2.addWeighted(image, 1 + input_obj.strength, gaussian, -input_obj.strength, 0)
            sharpened = unsharp_mask
        
        elif input_obj.method == "laplacian":
            # Laplacian sharpening - moderate
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = np.uint8(np.absolute(laplacian))
            
            # Apply threshold if specified
            if input_obj.threshold > 0:
                _, laplacian = cv2.threshold(laplacian, input_obj.threshold, 255, cv2.THRESH_BINARY)
            
            # Convert back to 3-channel and apply
            laplacian_3ch = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
            sharpened = cv2.addWeighted(image, 1.0, laplacian_3ch, input_obj.strength * 0.1, 0)
        
        elif input_obj.method == "high_pass":
            # High-pass filter sharpening - strong
            kernel = np.array([[-1, -1, -1],
                             [-1, 9, -1],
                             [-1, -1, -1]]) * input_obj.strength
            sharpened = cv2.filter2D(image, -1, kernel)
        
        # Ensure values are in valid range
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        # Save result
        cv2.imwrite(input_obj.output_path, sharpened)
        
        return {
            "success": True,
            "output_path": input_obj.output_path,
            "method": input_obj.method,
            "strength": input_obj.strength,
            "radius": input_obj.radius,
            "threshold": input_obj.threshold,
            "message": f"Image sharpened successfully using {input_obj.method} method"
        }
    
    except Exception as e:
        return {"success": False, "error": f"Image sharpening failed: {str(e)}"}


@tool
def denoise_image(input_data: str) -> Dict[str, Any]:
    """
    Removes noise from an image using various denoising algorithms.
    Addresses noise issues detected by image quality assessment.
    
    Args:
        input_data: JSON string containing image path, output path, denoising method, strength, edge preservation, and kernel size
    
    Returns:
        Dict containing success status, output path, applied settings, and quality metrics
    """
    try:
        # Clean and parse input data
        import json
        cleaned_input = input_data.strip().strip('`').strip("'").strip('"').strip()
        parsed_data = json.loads(cleaned_input)
        input_obj = DenoiseImageInput(**parsed_data)
        
        # Validate input file
        if not os.path.exists(input_obj.image_path):
            return {"success": False, "error": f"Input image not found: {input_obj.image_path}"}
        
        # Set output path
        if input_obj.output_path is None:
            base, ext = os.path.splitext(input_obj.image_path)
            input_obj.output_path = f"{base}_denoised{ext}"
        
        # Load image
        image = cv2.imread(input_obj.image_path)
        if image is None:
            return {"success": False, "error": "Failed to load image"}
        
        # Ensure kernel size is odd
        kernel_size = input_obj.kernel_size if input_obj.kernel_size % 2 == 1 else input_obj.kernel_size + 1
        
        # Apply denoising based on method
        if input_obj.method == "gaussian":
            # Gaussian blur - fast but may blur edges
            sigma = input_obj.strength * 2.0
            denoised = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        
        elif input_obj.method == "bilateral":
            # Bilateral filter - preserves edges
            d = kernel_size
            sigma_color = input_obj.strength * 50
            sigma_space = input_obj.strength * 50
            denoised = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        
        elif input_obj.method == "non_local_means":
            # Non-local means - best quality but slower
            h = input_obj.strength * 10
            template_window_size = min(kernel_size, 7)
            search_window_size = min(kernel_size * 3, 21)
            denoised = cv2.fastNlMeansDenoisingColored(image, None, h, h, template_window_size, search_window_size)
        
        elif input_obj.method == "median":
            # Median filter - good for salt-and-pepper noise
            denoised = cv2.medianBlur(image, kernel_size)
        
        # Save result
        cv2.imwrite(input_obj.output_path, denoised)
        
        return {
            "success": True,
            "output_path": input_obj.output_path,
            "method": input_obj.method,
            "strength": input_obj.strength,
            "preserve_edges": input_obj.preserve_edges,
            "kernel_size": kernel_size,
            "message": f"Image denoised successfully using {input_obj.method} method"
        }
    
    except Exception as e:
        return {"success": False, "error": f"Image denoising failed: {str(e)}"}


@tool
def adjust_exposure(input_data: str) -> Dict[str, Any]:
    """
    Adjusts image exposure, brightness, and gamma correction to improve overall lighting.
    Addresses exposure issues detected by image quality assessment.
    
    Args:
        input_data: JSON string containing image path, output path, exposure compensation, gamma correction, auto adjustment, and highlight/shadow preservation settings
    
    Returns:
        Dict containing success status, output path, applied settings, and quality metrics
    """
    try:
        # Clean and parse input data
        import json
        cleaned_input = input_data.strip().strip('`').strip("'").strip('"').strip()
        parsed_data = json.loads(cleaned_input)
        input_obj = AdjustExposureInput(**parsed_data)
        
        # Validate input file
        if not os.path.exists(input_obj.image_path):
            return {"success": False, "error": f"Input image not found: {input_obj.image_path}"}
        
        # Set output path
        if input_obj.output_path is None:
            base, ext = os.path.splitext(input_obj.image_path)
            input_obj.output_path = f"{base}_exposure{ext}"
        
        # Load image
        image = cv2.imread(input_obj.image_path)
        if image is None:
            return {"success": False, "error": "Failed to load image"}
        
        # Convert to float for processing
        image_float = image.astype(np.float32) / 255.0
        
        # Auto-adjust exposure if requested
        if input_obj.auto_adjust:
            # Calculate histogram statistics
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            # Find optimal exposure compensation based on histogram
            total_pixels = gray.shape[0] * gray.shape[1]
            cumsum = np.cumsum(hist.flatten())
            
            # Find percentiles
            p1 = np.where(cumsum >= total_pixels * 0.01)[0][0] / 255.0
            p99 = np.where(cumsum >= total_pixels * 0.99)[0][0] / 255.0
            
            # Calculate auto exposure compensation
            target_mid = 0.5
            current_mid = (p1 + p99) / 2
            auto_compensation = np.log2(target_mid / max(current_mid, 0.01))
            
            # Blend with manual compensation
            final_compensation = (input_obj.exposure_compensation + auto_compensation) / 2
        else:
            final_compensation = input_obj.exposure_compensation
        
        # Apply exposure compensation (in stops)
        if final_compensation != 0:
            exposure_factor = 2 ** final_compensation
            image_float = image_float * exposure_factor
        
        # Apply gamma correction
        if input_obj.gamma_correction != 1.0:
            image_float = np.power(image_float, 1.0 / input_obj.gamma_correction)
        
        # Preserve highlights if requested
        if input_obj.preserve_highlights:
            # Soft clipping for highlights
            highlight_mask = image_float > 0.9
            image_float = np.where(highlight_mask, 
                                 0.9 + 0.1 * np.tanh((image_float - 0.9) * 10), 
                                 image_float)
        
        # Preserve shadows if requested
        if input_obj.preserve_shadows:
            # Lift shadows gently
            shadow_mask = image_float < 0.1
            shadow_lift = 0.05 * (0.1 - image_float) / 0.1
            image_float = np.where(shadow_mask, image_float + shadow_lift, image_float)
        
        # Clip and convert back to uint8
        image_float = np.clip(image_float, 0, 1)
        adjusted = (image_float * 255).astype(np.uint8)
        
        # Save result
        cv2.imwrite(input_obj.output_path, adjusted)
        
        return {
            "success": True,
            "output_path": input_obj.output_path,
            "exposure_compensation": final_compensation,
            "gamma_correction": input_obj.gamma_correction,
            "auto_adjust": input_obj.auto_adjust,
            "preserve_highlights": input_obj.preserve_highlights,
            "preserve_shadows": input_obj.preserve_shadows,
            "message": f"Exposure adjusted successfully with {final_compensation:.2f} stops compensation"
        }
    
    except Exception as e:
        return {"success": False, "error": f"Exposure adjustment failed: {str(e)}"}


@tool
def enhance_contrast(input_data: str) -> Dict[str, Any]:
    """
    Enhances image contrast using various algorithms to improve dynamic range.
    Addresses contrast issues detected by image quality assessment.
    
    Args:
        input_data: JSON string containing image path, output path, contrast method, strength, and CLAHE parameters
    
    Returns:
        Dict containing success status, output path, applied settings, and quality metrics
    """
    try:
        # Clean and parse input data
        import json
        cleaned_input = input_data.strip().strip('`').strip("'").strip('"').strip()
        parsed_data = json.loads(cleaned_input)
        input_obj = EnhanceContrastInput(**parsed_data)
        
        # Validate input file
        if not os.path.exists(input_obj.image_path):
            return {"success": False, "error": f"Input image not found: {input_obj.image_path}"}
        
        # Set output path
        if input_obj.output_path is None:
            base, ext = os.path.splitext(input_obj.image_path)
            input_obj.output_path = f"{base}_contrast{ext}"
        
        # Load image
        image = cv2.imread(input_obj.image_path)
        if image is None:
            return {"success": False, "error": "Failed to load image"}
        
        # Apply contrast enhancement based on method
        if input_obj.method == "histogram_equalization":
            # Global histogram equalization
            if input_obj.preserve_color:
                # Convert to YUV and equalize only Y channel
                yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
                enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            else:
                # Equalize each channel separately
                channels = cv2.split(image)
                eq_channels = [cv2.equalizeHist(ch) for ch in channels]
                enhanced = cv2.merge(eq_channels)
            
            # Blend with original based on strength
            enhanced = cv2.addWeighted(image, 1 - input_obj.strength, enhanced, input_obj.strength, 0)
        
        elif input_obj.method == "adaptive_histogram":
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(
                clipLimit=input_obj.clip_limit,
                tileGridSize=(input_obj.tile_grid_size, input_obj.tile_grid_size)
            )
            
            if input_obj.preserve_color:
                # Apply CLAHE to L channel in LAB color space
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l_clahe = clahe.apply(l)
                
                # Blend with original based on strength
                l_final = cv2.addWeighted(l, 1 - input_obj.strength, l_clahe, input_obj.strength, 0)
                enhanced = cv2.merge([l_final, a, b])
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            else:
                # Apply CLAHE to each channel
                channels = cv2.split(image)
                clahe_channels = [clahe.apply(ch) for ch in channels]
                enhanced = cv2.merge(clahe_channels)
        
        elif input_obj.method == "linear_stretch":
            # Linear contrast stretching
            image_float = image.astype(np.float32)
            
            # Calculate percentiles for stretching
            p2, p98 = np.percentile(image_float, (2, 98))
            
            # Apply linear stretch
            stretched = (image_float - p2) * (255 / (p98 - p2))
            stretched = np.clip(stretched, 0, 255)
            
            # Blend with original based on strength
            enhanced = cv2.addWeighted(image.astype(np.float32), 1 - input_obj.strength, stretched, input_obj.strength, 0)
            enhanced = enhanced.astype(np.uint8)
        
        elif input_obj.method == "gamma_correction":
            # Gamma-based contrast enhancement
            image_float = image.astype(np.float32) / 255.0
            
            # Calculate adaptive gamma based on image statistics
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray) / 255.0
            
            # Adjust gamma based on brightness and strength
            if mean_brightness < 0.5:
                gamma = 1.0 - (input_obj.strength * 0.3)  # Brighten dark images
            else:
                gamma = 1.0 + (input_obj.strength * 0.3)  # Darken bright images
            
            enhanced_float = np.power(image_float, gamma)
            enhanced = (enhanced_float * 255).astype(np.uint8)
        
        # Save result
        cv2.imwrite(input_obj.output_path, enhanced)
        
        return {
            "success": True,
            "output_path": input_obj.output_path,
            "method": input_obj.method,
            "strength": input_obj.strength,
            "clip_limit": input_obj.clip_limit,
            "tile_grid_size": input_obj.tile_grid_size,
            "preserve_color": input_obj.preserve_color,
            "message": f"Contrast enhanced successfully using {input_obj.method} method"
        }
    
    except Exception as e:
        return {"success": False, "error": f"Contrast enhancement failed: {str(e)}"}


@tool
def enhance_color_resolution(input_data: str) -> Dict[str, Any]:
    """
    Enhances image color saturation, vibrancy, and resolution using various algorithms.
    Addresses color and resolution issues detected by image quality assessment.
    
    Args:
        input_data: JSON string containing image path, output path, color enhancement method, saturation factor, resolution enhancement, and scale factor
    
    Returns:
        Dict containing success status, output path, applied settings, and quality metrics
    """
    try:
        # Clean and parse input data
        import json
        cleaned_input = input_data.strip().strip('`').strip("'").strip('"').strip()
        parsed_data = json.loads(cleaned_input)
        input_obj = ColorResolutionInput(**parsed_data)
        
        # Validate input file
        if not os.path.exists(input_obj.image_path):
            return {"success": False, "error": f"Input image not found: {input_obj.image_path}"}
        
        # Set output path
        if input_obj.output_path is None:
            base, ext = os.path.splitext(input_obj.image_path)
            input_obj.output_path = f"{base}_enhanced{ext}"
        
        # Load image
        image = cv2.imread(input_obj.image_path)
        if image is None:
            return {"success": False, "error": "Failed to load image"}
        
        enhanced = image.copy()
        
        # Apply color enhancement
        if input_obj.color_enhancement != "none":
            if input_obj.color_enhancement == "saturation":
                # Uniform saturation boost
                hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 1] = hsv[:, :, 1] * input_obj.saturation_factor
                hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
                enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            
            elif input_obj.color_enhancement == "vibrance":
                # Selective saturation boost (vibrance)
                hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
                
                # Calculate saturation mask (boost less saturated colors more)
                sat_normalized = hsv[:, :, 1] / 255.0
                vibrance_mask = 1.0 - sat_normalized  # Less saturated = more boost
                
                # Apply selective boost
                boost_factor = 1.0 + (input_obj.saturation_factor - 1.0) * vibrance_mask
                hsv[:, :, 1] = hsv[:, :, 1] * boost_factor
                hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
                enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            
            elif input_obj.color_enhancement == "color_balance":
                # Simple color balance using gray world assumption
                enhanced_float = enhanced.astype(np.float32)
                
                # Calculate channel means
                b_mean, g_mean, r_mean = cv2.mean(enhanced)[:3]
                overall_mean = (b_mean + g_mean + r_mean) / 3
                
                # Apply correction factors
                enhanced_float[:, :, 0] *= overall_mean / b_mean  # Blue
                enhanced_float[:, :, 1] *= overall_mean / g_mean  # Green
                enhanced_float[:, :, 2] *= overall_mean / r_mean  # Red
                
                enhanced = np.clip(enhanced_float, 0, 255).astype(np.uint8)
            
            elif input_obj.color_enhancement == "auto_color":
                # Automatic color correction
                enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=10)
        
        # Apply white balance correction if requested
        if input_obj.white_balance_correction:
            # Simple white balance using gray world assumption
            enhanced_float = enhanced.astype(np.float32)
            b_mean, g_mean, r_mean = cv2.mean(enhanced)[:3]
            
            # Use green channel as reference
            enhanced_float[:, :, 0] *= g_mean / b_mean  # Blue
            enhanced_float[:, :, 2] *= g_mean / r_mean  # Red
            
            enhanced = np.clip(enhanced_float, 0, 255).astype(np.uint8)
        
        # Apply light noise reduction if requested
        if input_obj.noise_reduction:
            enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
        
        # Apply resolution enhancement
        if input_obj.resolution_enhancement != "none" and input_obj.scale_factor > 1.0:
            height, width = enhanced.shape[:2]
            new_height = int(height * input_obj.scale_factor)
            new_width = int(width * input_obj.scale_factor)
            
            if input_obj.resolution_enhancement == "bicubic":
                enhanced = cv2.resize(enhanced, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            elif input_obj.resolution_enhancement == "lanczos":
                enhanced = cv2.resize(enhanced, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            elif input_obj.resolution_enhancement == "super_resolution":
                # Placeholder for AI-based super resolution
                # In practice, this would use a trained model like ESRGAN or Real-ESRGAN
                enhanced = cv2.resize(enhanced, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Save result
        cv2.imwrite(input_obj.output_path, enhanced)
        
        return {
            "success": True,
            "output_path": input_obj.output_path,
            "color_enhancement": input_obj.color_enhancement,
            "saturation_factor": input_obj.saturation_factor,
            "resolution_enhancement": input_obj.resolution_enhancement,
            "scale_factor": input_obj.scale_factor,
            "white_balance_correction": input_obj.white_balance_correction,
            "noise_reduction": input_obj.noise_reduction,
            "message": f"Image enhanced successfully with {input_obj.color_enhancement} color enhancement and {input_obj.resolution_enhancement} resolution enhancement"
        }
    
    except Exception as e:
        return {"success": False, "error": f"Color/resolution enhancement failed: {str(e)}"}