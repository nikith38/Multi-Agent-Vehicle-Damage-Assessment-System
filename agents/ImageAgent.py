from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain import hub
import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv

# Add tools directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))

# Import our image tools
from imagetools import (
    assess_image_metrics,
    sharpen_image,
    denoise_image,
    adjust_exposure,
    enhance_contrast,
    enhance_color_resolution
)

# Import prompt template
from prompts.image_enhan_prompt import get_image_enhancement_prompt

class ImageEnhancementAgent:
    """
    ReACT agent specialized in image quality assessment and enhancement.
    Uses LangChain's ReACT framework with custom image processing tools.
    """
    
    def __init__(self, api_key=None, model="gpt-4o-mini", temperature=0.1, enable_logging=True):
        """
        Initialize the Image Enhancement Agent.
        
        Args:
            api_key: OpenAI API key (if None, will use environment variable)
            model: OpenAI model to use (default: gpt-4o-mini)
            temperature: Model temperature for consistency (default: 0.1)
            enable_logging: Enable detailed tool call logging (default: True)
        """
        # Setup logging if enabled
        if enable_logging:
            self._setup_logging()
        
        self.enable_logging = enable_logging
        self.tool_call_count = 0
        self.session_start_time = datetime.now()
        self.intermediate_files = []  # Track intermediate files for cleanup
        self.used_tools = set()  # Track tools used in current run

        # Load environment variables from .env file with override=True
        load_dotenv(override=True)
        
        # Use provided API key or load from environment
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key
        )
        
        # Define available tools
        self.tools = [
            assess_image_metrics,
            sharpen_image,
            denoise_image,
            adjust_exposure,
            enhance_contrast,
            enhance_color_resolution
        ]
        
        # Pull the ReACT prompt from LangChain hub
        try:
            self.prompt = hub.pull("hwchase17/react")
        except Exception as e:
            # Fallback to custom prompt if hub is not available
            self.prompt = get_image_enhancement_prompt(self.tools)
        
        # Create the ReACT agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # Wrap with AgentExecutor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
    
    def _setup_logging(self):
        """Setup detailed logging for tool calls."""
        # Create logger
        self.logger = logging.getLogger('ImageEnhancementAgent')
        self.logger.setLevel(logging.INFO)
        
        # Create handlers with formatting
        if not self.logger.handlers:
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            
            # File handler
            file_handler = logging.FileHandler('image_agent.log')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            
            # Add handlers to logger
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)
    
    def _track_intermediate_file(self, file_path):
        """Track an intermediate file for later cleanup."""
        if file_path and os.path.exists(file_path) and file_path not in self.intermediate_files:
            self.intermediate_files.append(file_path)
            if self.enable_logging:
                self.logger.info(f"Tracking intermediate file: {file_path}")
    
    def _extract_output_path_from_response(self, tool_output):
        """Extract output file path from tool response."""
        try:
            import json
            if isinstance(tool_output, str):
                output_data = json.loads(tool_output)
            else:
                output_data = tool_output
            
            if isinstance(output_data, dict) and 'output_path' in output_data:
                return output_data['output_path']
        except:
            pass
        return None
    
    def _cleanup_intermediate_files(self, keep_final=None):
        """Clean up intermediate files, optionally keeping the final result."""
        cleaned_count = 0
        
        for file_path in self.intermediate_files:
            # Skip the final file if specified
            if keep_final and file_path == keep_final:
                continue
                
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    cleaned_count += 1
                    if self.enable_logging:
                        self.logger.info(f"Cleaned up intermediate file: {file_path}")
            except Exception as e:
                if self.enable_logging:
                    self.logger.warning(f"Failed to clean up {file_path}: {e}")
        
        # Clear the tracking list
        if keep_final:
            self.intermediate_files = [keep_final] if keep_final in self.intermediate_files else []
        else:
            self.intermediate_files = []
        
        if self.enable_logging and cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} intermediate files")
    
    def _log_tool_call(self, tool_name, tool_input, tool_output, execution_time=None):
        """Log detailed information about tool calls."""
        if not self.enable_logging:
            return
            
        self.tool_call_count += 1
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"TOOL CALL #{self.tool_call_count}")
        self.logger.info(f"Tool Name: {tool_name}")
        self.logger.info(f"Input: {tool_input}")
        if execution_time:
            self.logger.info(f"Execution Time: {execution_time:.3f}s")
        self.logger.info(f"Output: {tool_output}")
        self.logger.info(f"{'='*60}")
    
    def _log_execution_summary(self, response, total_execution_time):
        """Log a comprehensive summary of the agent execution."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"EXECUTION SUMMARY")
        self.logger.info(f"Total Execution Time: {total_execution_time:.3f}s")
        self.logger.info(f"Total Tool Calls: {len(response.get('intermediate_steps', []))}")
        self.logger.info(f"Used Tools: {sorted(list(self.used_tools))}")
        
        # Log each intermediate step (tool call)
        intermediate_steps = response.get('intermediate_steps', [])
        for i, (action, observation) in enumerate(intermediate_steps, 1):
            self.logger.info(f"\n--- TOOL CALL {i} ---")
            self.logger.info(f"Tool: {action.tool}")
            self.logger.info(f"Input: {action.tool_input}")
            self.logger.info(f"Output: {observation}")
        
        self.logger.info(f"\nFinal Answer: {response.get('output', 'No output')}")
        self.logger.info(f"{'='*80}")
    
    def _generate_final_output_json(self, enhanced_image_path, quality_score=None):
        """Generate final output JSON file with enhancement results."""
        import json
        
        # Static mapping of tools to image quality issues
        TOOL_TO_ISSUE_MAPPING = {
            'sharpen_image': 'blur',
            'denoise_image': 'noise',
            'adjust_exposure': 'low_light',
            'enhance_contrast': 'low_contrast',
            'enhance_color_resolution': 'color_distortion',
            'assess_image_metrics': None  # This tool doesn't fix issues, just assesses
        }
        
        # Extract quality score from the first assess_image_metrics call if available
        if quality_score is None:
            quality_score = None  # Default fallback
        
        # Convert used tools to corresponding issues
        issues = []
        for tool in self.used_tools:
            if tool in TOOL_TO_ISSUE_MAPPING and TOOL_TO_ISSUE_MAPPING[tool] is not None:
                issue = TOOL_TO_ISSUE_MAPPING[tool]
                if issue not in issues:  # Avoid duplicates
                    issues.append(issue)
        
        # Create output data
        output_data = {
            "quality_score": quality_score,
            "issues": sorted(issues),
            "enhanced_image": enhanced_image_path,
            "processable": True
        }
        
        # Generate output filename based on enhanced image
        if enhanced_image_path:
            base_name = os.path.splitext(enhanced_image_path)[0]
            output_file = f"{base_name}_output.json"
        else:
            output_file = "enhancement_output.json"
        
        # Write JSON file
        try:
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=4)
            
            if self.enable_logging:
                self.logger.info(f"Final output JSON saved: {output_file}")
            
            return output_file
        except Exception as e:
            if self.enable_logging:
                self.logger.error(f"Failed to save output JSON: {str(e)}")
            return None
    
    def enhance_image(self, image_path):
        """
        Enhance an image by analyzing its quality and applying appropriate enhancements.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Dict containing the agent's response and any enhanced image paths
        """
        request = f"Analyze the image quality of '{image_path}' and enhance it as needed. First assess the image metrics, then apply appropriate enhancements based on the quality issues found."
        
        if self.enable_logging:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"STARTING IMAGE ENHANCEMENT SESSION")
            self.logger.info(f"Session Start Time: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"Image Path: {image_path}")
            self.logger.info(f"Request: {request}")
            self.logger.info(f"{'='*80}")
        
        try:
            # Clear any previous intermediate files and used tools
            self.intermediate_files = []
            self.used_tools = set()
            
            start_time = datetime.now()
            response = self.agent_executor.invoke({
                "input": request
            })
            end_time = datetime.now()
            total_execution_time = (end_time - start_time).total_seconds()
            
            # Track intermediate files and used tools from tool calls
            final_output_file = None
            quality_score = None
            
            if 'intermediate_steps' in response:
                for action, observation in response['intermediate_steps']:
                    # Track all tools used
                    self.used_tools.add(action.tool)
                    
                    # Extract quality score from assess_image_metrics
                    if action.tool == 'assess_image_metrics':
                        try:
                            import json
                            metrics_data = json.loads(observation)
                            quality_score = metrics_data.get('overall_metrics', {}).get('overall_score', 0.85)
                        except:
                            quality_score = 0.85  # Fallback
                    else:
                        # Track files created by other tools
                        output_path = self._extract_output_path_from_response(observation)
                        if output_path:
                            self._track_intermediate_file(output_path)
                            final_output_file = output_path  # Keep track of the last created file
            
            # Log detailed information about the execution
            if self.enable_logging:
                self._log_execution_summary(response, total_execution_time)
            
            # Clean up intermediate files, keeping only the final result
            if len(self.intermediate_files) > 0:
                self._cleanup_intermediate_files(keep_final=final_output_file)
                if self.enable_logging:
                    self.logger.info(f"Final enhanced image: {final_output_file}")
            
            # Generate final output JSON
            self._generate_final_output_json(final_output_file, quality_score)
            
            return response
        except Exception as e:
            if self.enable_logging:
                self.logger.error(f"Enhancement failed: {str(e)}")
            return {
                "error": f"Enhancement failed: {str(e)}",
                "output": None
            }
    

    



def create_image_agent(api_key=None, model="gpt-4o-mini", temperature=0.1, enable_logging=True):
    """
    Factory function to create an ImageEnhancementAgent instance.
    
    Args:
        api_key: OpenAI API key (if None, will use environment variable)
        model: OpenAI model to use (default: gpt-4o-mini)
        temperature: Model temperature for consistency (default: 0.1)
        enable_logging: Enable detailed tool call logging (default: True)
    
    Returns:
        ImageEnhancementAgent: Configured agent instance
    """
    return ImageEnhancementAgent(api_key=api_key, model=model, temperature=temperature, enable_logging=enable_logging)


# Example usage and testing
if __name__ == "__main__":
    # Create the image enhancement agent
    agent = create_image_agent()
    
    # Single mode: Image Enhancement
    print("=== Image Enhancement Agent ===")
    test_image = "tests/front_left-18.jpeg"
    
    if os.path.exists(test_image):
        print(f"Processing image: {test_image}")
        response = agent.enhance_image(test_image)
        print(f"Response: {response['output']}")
        print(f"Intermediate steps: {len(response.get('intermediate_steps', []))}")
        print(f"Used tools: {sorted(list(agent.used_tools))}")
    else:
        print(f"Test image not found: {test_image}")
    
    print("\n=== Agent Setup Complete ===")
    print(f"Available tools: {[tool.name for tool in agent.tools]}")
    print(f"Model: {agent.llm.model_name}")
    print(f"Temperature: {agent.llm.temperature}")