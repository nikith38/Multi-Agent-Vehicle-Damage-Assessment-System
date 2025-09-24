"""Image Quality Assessment Tools for Vehicle Damage Detection System

This module provides computer vision tools for assessing and enhancing image quality
using classic CV techniques, decorated with LangChain @tool for ReACT agent integration.
"""

import cv2
import numpy as np
import asyncio
import concurrent.futures
import functools
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field
from langchain.tools import tool
import json
import time
import pywt
import logging
from scipy import ndimage

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
        # Parse input
        input_dict = json.loads(input_data)
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