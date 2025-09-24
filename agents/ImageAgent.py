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
        
        # Log each intermediate step (tool call)
        intermediate_steps = response.get('intermediate_steps', [])
        for i, (action, observation) in enumerate(intermediate_steps, 1):
            self.logger.info(f"\n--- TOOL CALL {i} ---")
            self.logger.info(f"Tool: {action.tool}")
            self.logger.info(f"Input: {action.tool_input}")
            self.logger.info(f"Output: {observation}")
        
        self.logger.info(f"\nFinal Answer: {response.get('output', 'No output')}")
        self.logger.info(f"{'='*80}")
    
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
            start_time = datetime.now()
            response = self.agent_executor.invoke({
                "input": request
            })
            end_time = datetime.now()
            total_execution_time = (end_time - start_time).total_seconds()
            
            # Log detailed information about the execution
            if self.enable_logging:
                self._log_execution_summary(response, total_execution_time)
            
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
    else:
        print(f"Test image not found: {test_image}")
    
    print("\n=== Agent Setup Complete ===")
    print(f"Available tools: {[tool.name for tool in agent.tools]}")
    print(f"Model: {agent.llm.model_name}")
    print(f"Temperature: {agent.llm.temperature}")