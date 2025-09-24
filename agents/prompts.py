"""ReACT Prompts for Multi-Agent Vehicle Damage Detection System

This module contains structured prompts for the ReACT (Reasoning + Acting) agents
used in the vehicle damage assessment pipeline.
"""

from typing import Dict, Any


class ImageQualityAgentPrompts:
    """ReACT prompts for the Image Quality Agent"""
    
    SYSTEM_PROMPT = """
You are an expert Image Quality Analyst for vehicle damage assessment in insurance claims.
Your role is to evaluate and enhance image quality to ensure optimal conditions for damage detection.

You operate in a ReACT (Reasoning + Acting) framework with these capabilities:
- Assess image quality metrics (blur, lighting, resolution, noise)
- Enhance images using computer vision techniques
- Detect potential image manipulation/fraud
- Make final processability decisions

Your goal: Ensure images meet quality standards for accurate damage detection while flagging potential fraud.

Processing constraints:
- Maximum 3 iterations per image
- Target processing time: <1 second
- Quality threshold: 0.7+ for processability
- Must maintain image authenticity while enhancing
"""
    
    REASONING_TEMPLATE = """
Current Image Analysis:
===================
Image ID: {image_id}
Iteration: {iteration}/{max_iterations}

Current Quality Metrics:
- Overall Quality Score: {quality_score:.2f}/1.0
- Blur Level: {blur_score:.2f} (lower is better)
- Brightness Score: {brightness_score:.2f} (0.4-0.6 optimal)
- Contrast Score: {contrast_score:.2f} (higher is better)
- Resolution: {width}x{height} pixels
- Noise Level: {noise_level:.2f}

Previous Actions Taken: {previous_actions}
Quality Improvement: {quality_improvement}

Available Tools:
1. assess_image_metrics - Get detailed quality measurements
2. enhance_lighting - Fix brightness/contrast issues (CLAHE, gamma correction)
3. reduce_blur - Apply sharpening techniques (unsharp mask, deconvolution)
4. detect_manipulation - Check for image tampering/fraud indicators
5. validate_processability - Final quality gate and decision

Reasoning Instructions:
1. Analyze the current quality metrics
2. Identify the most critical quality issue
3. Consider previous actions to avoid redundancy
4. Choose the most appropriate tool for improvement
5. If quality is sufficient (>0.7) or max iterations reached, validate processability

Think step by step about what action will most improve image quality for damage detection.
"""
    
    ACTION_SELECTION_PROMPT = """
Based on your analysis, select the most appropriate action:

Decision Criteria:
- If overall quality < 0.4: Focus on primary issue (lighting or blur)
- If 0.4 ≤ quality < 0.7: Address secondary issues or validate improvements
- If quality ≥ 0.7 OR iteration = 3: Run detect_manipulation + validate_processability
- If no improvement in last iteration: Stop and validate

Action Format:
Thought: [Your reasoning for this action]
Action: [tool_name]
Action Input: {"parameter": "value"}

Remember: Each action should measurably improve image quality or provide fraud detection.
"""
    
    QUALITY_THRESHOLDS = {
        "minimum_processable": 0.7,
        "excellent_quality": 0.85,
        "blur_threshold": 100.0,  # Laplacian variance
        "brightness_range": (0.3, 0.7),
        "contrast_minimum": 0.4,
        "noise_maximum": 0.3,
        "min_resolution": (640, 480)
    }
    
    ENHANCEMENT_GUIDELINES = {
        "lighting_issues": {
            "low_brightness": "Apply gamma correction (γ=1.2-2.0) or histogram stretching",
            "high_brightness": "Apply gamma correction (γ=0.5-0.8) or tone mapping",
            "low_contrast": "Use CLAHE with clip limit 2.0-4.0",
            "uneven_lighting": "Apply adaptive histogram equalization"
        },
        "blur_issues": {
            "mild_blur": "Unsharp masking with radius 1-2, amount 0.5-1.5",
            "motion_blur": "Wiener filtering or Richardson-Lucy deconvolution",
            "focus_blur": "Edge enhancement with careful artifact monitoring"
        },
        "fraud_indicators": {
            "compression_artifacts": "ELA analysis for inconsistent compression",
            "metadata_tampering": "EXIF validation and timestamp verification",
            "copy_paste_regions": "Statistical analysis of noise patterns"
        }
    }
    
    TERMINATION_CONDITIONS = [
        "Quality score ≥ 0.7 (processable threshold met)",
        "Maximum iterations (3) reached",
        "No quality improvement (Δ < 0.05) in current iteration",
        "Enhancement artifacts detected (quality degradation)",
        "Image deemed unprocessable due to fundamental issues"
    ]
    
    ERROR_HANDLING = {
        "tool_failure": "Log error, skip to next appropriate tool, continue processing",
        "quality_degradation": "Revert to previous version, try alternative enhancement",
        "processing_timeout": "Return current best version with quality assessment",
        "invalid_image": "Mark as unprocessable, provide detailed failure reason"
    }

    @classmethod
    def format_reasoning_prompt(cls, **kwargs) -> str:
        """Format the reasoning template with current image data"""
        return cls.REASONING_TEMPLATE.format(**kwargs)
    
    @classmethod
    def get_quality_assessment(cls, metrics: Dict[str, Any]) -> str:
        """Generate quality assessment text based on metrics"""
        quality = metrics.get('quality_score', 0.0)
        
        if quality >= cls.QUALITY_THRESHOLDS['excellent_quality']:
            return "Excellent quality - proceed to damage detection"
        elif quality >= cls.QUALITY_THRESHOLDS['minimum_processable']:
            return "Acceptable quality - suitable for damage detection"
        elif quality >= 0.4:
            return "Marginal quality - enhancement recommended"
        else:
            return "Poor quality - significant enhancement required"
    
    @classmethod
    def get_enhancement_recommendation(cls, metrics: Dict[str, Any]) -> str:
        """Recommend specific enhancement based on quality metrics"""
        recommendations = []
        
        # Check brightness
        brightness = metrics.get('brightness_score', 0.5)
        if brightness < cls.QUALITY_THRESHOLDS['brightness_range'][0]:
            recommendations.append("Increase brightness (gamma correction)")
        elif brightness > cls.QUALITY_THRESHOLDS['brightness_range'][1]:
            recommendations.append("Reduce brightness (tone mapping)")
        
        # Check blur
        blur = metrics.get('blur_score', 0.0)
        if blur < cls.QUALITY_THRESHOLDS['blur_threshold']:
            recommendations.append("Apply sharpening (unsharp mask)")
        
        # Check contrast
        contrast = metrics.get('contrast_score', 0.0)
        if contrast < cls.QUALITY_THRESHOLDS['contrast_minimum']:
            recommendations.append("Enhance contrast (CLAHE)")
        
        return "; ".join(recommendations) if recommendations else "No enhancement needed"


# Example usage and testing
if __name__ == "__main__":
    # Example of how the prompts would be used
    sample_metrics = {
        'image_id': 'IMG-001',
        'iteration': 1,
        'max_iterations': 3,
        'quality_score': 0.45,
        'blur_score': 85.0,
        'brightness_score': 0.25,
        'contrast_score': 0.35,
        'width': 1920,
        'height': 1080,
        'noise_level': 0.15,
        'previous_actions': [],
        'quality_improvement': 0.0
    }
    
    prompts = ImageQualityAgentPrompts()
    reasoning_prompt = prompts.format_reasoning_prompt(**sample_metrics)
    print("ReACT Reasoning Prompt:")
    print(reasoning_prompt)
    print("\n" + "="*50 + "\n")
    print("Quality Assessment:", prompts.get_quality_assessment(sample_metrics))
    print("Enhancement Recommendation:", prompts.get_enhancement_recommendation(sample_metrics))