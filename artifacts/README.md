# Image Enhancement ReACT Agent

A sophisticated AI agent built with LangChain's ReACT framework that can assess image quality and automatically apply appropriate enhancements. The agent uses GPT-4o-mini to intelligently analyze images and select the best enhancement tools based on detected quality issues.

## 🚀 Features

### Image Quality Assessment
- **Blur Detection**: Laplacian variance analysis
- **Brightness Analysis**: Histogram-based brightness evaluation
- **Contrast Assessment**: Michelson contrast measurement
- **Noise Detection**: Wavelet-based noise estimation
- **Region Analysis**: Quality assessment for different image regions
- **Overall Quality Scoring**: Comprehensive quality metrics

### Image Enhancement Tools
1. **Sharpen Image**: Reduces blur using unsharp mask, Laplacian, or high-pass filtering
2. **Denoise Image**: Removes noise while preserving edges (Gaussian, bilateral, non-local means, median)
3. **Adjust Exposure**: Optimizes brightness and gamma correction with highlight/shadow preservation
4. **Enhance Contrast**: Improves contrast using histogram equalization, CLAHE, or linear stretching
5. **Color & Resolution Enhancement**: Boosts color saturation, applies white balance, and upscales resolution

### ReACT Agent Capabilities
- **Intelligent Decision Making**: Analyzes quality issues and selects appropriate tools
- **Multi-step Processing**: Can apply multiple enhancements in sequence
- **Adaptive Parameters**: Adjusts enhancement parameters based on image characteristics
- **Reasoning Transparency**: Shows step-by-step decision process

## 📁 Project Structure

```
zoop_main/
├── agents/
│   ├── ImageAgent.py          # Main ReACT agent implementation
│   └── prompts.py            # Custom prompts (if needed)
├── tools/
│   └── imagetools.py         # All image processing tools (assessment and enhancement)
├── tests/
│   └── front_left-18.jpeg    # Sample test image
├── test_image_agent.py       # Agent testing and demo script
├── test_vehicle_image.py     # Vehicle image testing
└── requirements.txt          # Python dependencies
```

## 🛠️ Installation

1. **Clone or download the project**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up OpenAI API key**:
   ```bash
   # Windows
   set OPENAI_API_KEY=your_api_key_here
   
   # Linux/Mac
   export OPENAI_API_KEY=your_api_key_here
   ```

## 🎯 Usage

### Basic Usage

```python
from agents.ImageAgent import create_image_agent

# Create the agent
agent = create_image_agent()

# Automatic enhancement based on quality assessment
response = agent.enhance_image("path/to/your/image.jpg")
print(response['output'])

# Quality assessment only
response = agent.assess_only("path/to/your/image.jpg")
print(response['output'])

# Custom enhancement with specific instructions
response = agent.custom_enhancement(
    "path/to/your/image.jpg",
    "Make the image brighter and reduce noise while preserving natural colors"
)
print(response['output'])
```

### Advanced Usage

```python
# Create agent with custom settings
agent = create_image_agent(
    model="gpt-4o-mini",
    temperature=0.1,
    api_key="your_api_key"
)

# Complex enhancement request
response = agent.enhance_image(
    "vehicle_image.jpg",
    "Analyze this vehicle image for any quality issues. If the image is blurry, apply sharpening. If it's too dark, adjust exposure. If there's noise, apply denoising. Provide a detailed report of what was done."
)

# Access intermediate steps
if 'intermediate_steps' in response:
    for i, (action, observation) in enumerate(response['intermediate_steps']):
        print(f"Step {i+1}: {action.tool}")
        print(f"Result: {observation}")
```

### Testing the Agent

```bash
# Run the comprehensive test suite
python test_image_agent.py

# Choose testing mode:
# 1. Automated test suite
# 2. Interactive demo
# 3. Both
```

## 🧪 Testing Examples

### Test 1: Quality Assessment
```python
# The agent will analyze image quality and provide detailed metrics
response = agent.assess_only("tests/front_left-18.jpeg")
```

### Test 2: Automatic Enhancement
```python
# The agent will assess quality and apply appropriate enhancements
response = agent.enhance_image("tests/front_left-18.jpeg")
```

### Test 3: Custom Instructions
```python
# Specific enhancement requests
response = agent.custom_enhancement(
    "tests/front_left-18.jpeg",
    "Make this vehicle image brighter and sharper for better visibility"
)
```

## 🔧 Tool Parameters

### Sharpen Image
- `method`: "unsharp_mask", "laplacian", "high_pass"
- `strength`: 0.1-3.0 (sharpening intensity)
- `radius`: 0.5-5.0 (sharpening radius)
- `threshold`: 0-255 (edge detection threshold)

### Denoise Image
- `method`: "gaussian", "bilateral", "non_local_means", "median"
- `strength`: 0.1-3.0 (denoising intensity)
- `kernel_size`: 3-15 (filter kernel size)
- `preserve_edges`: true/false

### Adjust Exposure
- `exposure_compensation`: -2.0 to +2.0 stops
- `gamma_correction`: 0.3-3.0
- `auto_adjust`: true/false
- `preserve_highlights`: true/false
- `preserve_shadows`: true/false

### Enhance Contrast
- `method`: "histogram_equalization", "adaptive_histogram", "linear_stretch", "gamma_correction"
- `strength`: 0.1-3.0
- `clip_limit`: 1.0-10.0 (for CLAHE)
- `tile_grid_size`: 4-16 (for CLAHE)

### Color & Resolution Enhancement
- `color_enhancement`: "none", "saturation", "vibrance", "color_balance", "auto_color"
- `saturation_factor`: 0.5-2.0
- `resolution_enhancement`: "none", "bicubic", "lanczos", "super_resolution"
- `scale_factor`: 1.0-4.0

## 🎨 Agent Behavior

The ReACT agent follows this decision process:

1. **Assessment**: Uses `assess_image_metrics` to analyze image quality
2. **Analysis**: Interprets quality metrics to identify issues
3. **Planning**: Determines which enhancement tools to use
4. **Execution**: Applies enhancements with appropriate parameters
5. **Validation**: May re-assess to verify improvements
6. **Reporting**: Provides detailed explanation of actions taken

## 🚨 Troubleshooting

### Common Issues

1. **"OpenAI API key not found"**
   - Set the `OPENAI_API_KEY` environment variable
   - Or pass the API key directly to `create_image_agent(api_key="your_key")`

2. **"Module not found" errors**
   - Install all requirements: `pip install -r requirements.txt`
   - Ensure you're in the correct directory

3. **"Image not found" errors**
   - Check file paths are correct
   - Use absolute paths if relative paths don't work

4. **Tool execution errors**
   - Ensure image files are valid (JPEG, PNG, etc.)
   - Check file permissions
   - Verify OpenCV can read the image format

### Performance Tips

- Use `temperature=0.1` for consistent results
- Set `max_iterations=10` to prevent infinite loops
- Enable `verbose=True` to see agent reasoning
- Use `return_intermediate_steps=True` to debug tool usage

## 📊 Example Output

```
🤖 ImageEnhancementAgent Test Suite
==================================================
✅ Test image found: tests/front_left-18.jpeg
📅 Test started at: 2024-01-15 14:30:25

🔧 Creating ImageEnhancementAgent...
✅ Agent created successfully
🧠 Model: gpt-4o-mini
🌡️  Temperature: 0.1
🛠️  Available tools: 6
   1. assess_image_metrics
   2. sharpen_image
   3. denoise_image
   4. adjust_exposure
   5. enhance_contrast
   6. enhance_color_resolution

📊 Test 1: Quality Assessment Only
------------------------------
✅ Assessment completed successfully
📋 Result: I have analyzed the image quality metrics for 'tests/front_left-18.jpeg'. Here's the detailed assessment:

**Overall Quality Score: 7.2/10**

**Detailed Metrics:**
- Blur Score: 8.1/10 (Good sharpness)
- Brightness Score: 6.8/10 (Slightly underexposed)
- Contrast Score: 7.5/10 (Adequate contrast)
- Noise Score: 6.9/10 (Minor noise present)

**Quality Issues Detected:**
- Slight underexposure in shadow regions
- Minor noise in uniform areas

**Recommendations:**
- Consider slight exposure adjustment (+0.3 stops)
- Light denoising could improve quality
- Overall image quality is acceptable for most uses

🔄 Intermediate steps: 1
```

## 🤝 Contributing

To extend the agent with new tools:

1. Create new tool functions in `tools/imagetools.py`
2. Decorate with `@tool` and proper input validation
3. Add to the tools list in `ImageAgent.py`
4. Update this README with new capabilities

## 📄 License

This project is open source and available under the MIT License.