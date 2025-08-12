import os
from typing import AsyncGenerator, Literal, Dict, Any, List, TypedDict

from ai_core_sdk.ai_core_v2_client import AICoreV2Client
from gen_ai_hub.proxy.native.openai import embeddings
from llm_commons.langchain.proxy import init_llm
from gen_ai_hub.proxy.langchain.openai import init_chat_model as init_openai_chat_model
from llm_commons.proxy.base import get_proxy_client
from sentence_transformers import SentenceTransformer
from gen_ai_hub.proxy.core import proxy_clients

# LangChain imports
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import BaseOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from langchain.schema.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableParallel, RunnableSequence

from utils.logger_config import setup_logger
logger = setup_logger(__name__)
logger.info("Logger initialized for aicore_utils.py")

# Environment configuration
AICORE_CONFIG = {
    "AUTH_URL": os.environ.get("AICORE_AUTH_URL") + "/oauth/token",
    "CLIENT_ID": os.environ.get("AICORE_CLIENT_ID"),
    "CLIENT_SECRET": os.environ.get("AICORE_CLIENT_SECRET"),
    "RESOURCE_GROUP": os.environ.get("AICORE_RESOURCE_GROUP"),
    "BASE_URL": os.environ.get("AICORE_BASE_URL") + "/v2",
}

base_url = AICORE_CONFIG["BASE_URL"]
auth_url = AICORE_CONFIG["AUTH_URL"]

print(f"Using base URL: {base_url}")

proxy_client = get_proxy_client(
    base_url=base_url,
    auth_url=auth_url,
    client_id=AICORE_CONFIG["CLIENT_ID"],
    client_secret=AICORE_CONFIG["CLIENT_SECRET"],
)
ai_core_client = None
EMBEDDING_MODEL = None
llm = None 

class AIResponseParser(BaseOutputParser):
    """Custom output parser for AI responses"""
    
    def parse(self, text: str) -> str:
        """Parse the output from AI model"""
        return text.strip()

class BtpAICoreService:
    """
    Refactored BTP AI Core Service using LangChain/LangGraph architecture.
    Provides a more structured and maintainable approach to AI operations.
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.5, max_tokens: int = 400):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize proxy client
        self.proxy_client = get_proxy_client(
            base_url=AICORE_CONFIG["BASE_URL"],
            auth_url=AICORE_CONFIG["AUTH_URL"],
            client_id=AICORE_CONFIG["CLIENT_ID"],
            client_secret=AICORE_CONFIG["CLIENT_SECRET"],
        )
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Initialize chains
        self.chat_chain = self._create_chat_chain()
        self.streaming_chain = self._create_streaming_chain()
        
    def _initialize_llm(self):
        """Initialize the LLM with proper configuration"""
        try:
            llm = init_llm(
                self.model_name,
                proxy_client=self.proxy_client,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                init_func=init_openai_chat_model,
            )
            logger.info(f"LLM initialized successfully with model: {self.model_name}")
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise
            
    def _create_chat_chain(self) -> Runnable:
        """Create a LangChain chain for standard chat operations"""
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("{system_prompt}"),
            HumanMessagePromptTemplate.from_template("{user_message}")
        ])
        
        # Create the chain: prompt -> llm -> parser
        chain = (
            prompt
            | self.llm
            | AIResponseParser()
        )
        
        return chain
    
    def _create_streaming_chain(self) -> Runnable:
        """Create a LangChain chain for streaming operations"""
        # For streaming, we'll use a simpler approach
        def format_messages(inputs: Dict[str, Any]) -> List[BaseMessage]:
            return [
                SystemMessage(content=inputs["system_prompt"]),
                HumanMessage(content=inputs["user_message"])
            ]
        
        chain = RunnableLambda(format_messages) | self.llm
        return chain
    
    def call_aicore_api(self, system_prompt: str, user_message: str) -> str:
        """
        Makes an API call to BTP AI Core using LangChain chain and returns the response.

        Args:
            system_prompt (str): The instruction/system prompt
            user_message (str): The user message

        Returns:
            str: AI model's response text
        """
        try:
            logger.info(f"Making API call with system prompt length: {len(system_prompt)}")
            
            # Use the LangChain chain
            response = self.chat_chain.invoke({
                "system_prompt": system_prompt,
                "user_message": user_message
            })
            
            logger.info(f"API call completed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error in AI Core API call: {str(e)}")
            logger.info(f"AI Core ENV: {AICORE_CONFIG}")
            raise

    async def call_aicore_api_stream(
        self,
        system_prompt: str,
        user_message: str,
    ) -> AsyncGenerator[str, None]: 
        """
        Makes a streaming API call to BTP AI Core using LangChain and yields the responses.

        Args:
            system_prompt (str): The instruction/system prompt
            user_message (str): The user message

        Yields:
            str: Chunks of AI model's response text
        """
        try:
            logger.info(f"Making streaming API call with system prompt length: {len(system_prompt)}")
            
            # Use the streaming chain
            inputs = {
                "system_prompt": system_prompt,
                "user_message": user_message
            }
            
            final_response = ""
            async for chunk in self.streaming_chain.astream(inputs):
                if hasattr(chunk, 'content') and chunk.content:
                    final_response += chunk.content
                    yield chunk.content
            
            logger.info(f"Streaming API call completed. Final response length: {len(final_response)}")
            
        except Exception as e:
            logger.error(f"Error in AI Core streaming API call: {str(e)}")
            logger.info(f"AI Core ENV: {AICORE_CONFIG}")
            raise
    
    def create_custom_chain(self, prompt_template: str) -> Runnable:
        """
        Create a custom LangChain chain with a specific prompt template.
        
        Args:
            prompt_template (str): Custom prompt template string
            
        Returns:
            Runnable: Configured LangChain chain
        """
        prompt = PromptTemplate.from_template(prompt_template)
        
        chain = (
            prompt
            | self.llm  
            | AIResponseParser()
        )
        
        return chain
    
    def batch_process(self, inputs: List[Dict[str, str]]) -> List[str]:
        """
        Process multiple requests in batch using LangChain.
        
        Args:
            inputs: List of dictionaries containing system_prompt and user_message
            
        Returns:
            List[str]: List of responses
        """
        try:
            responses = self.chat_chain.batch(inputs)
            logger.info(f"Batch processing completed for {len(inputs)} requests")
            return responses
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            raise
    
    def get_chain_config(self) -> Dict[str, Any]:
        """Get current chain configuration"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "base_url": AICORE_CONFIG["BASE_URL"],
        }

# LangGraph imports for advanced workflows
# Note: LangGraph is optional. If not installed, workflow features will be disabled
# but all basic functionality will still work.
try:
    from langgraph.graph import StateGraph, START, END
    from langchain.tools import BaseTool
    from pydantic import BaseModel, Field
    LANGGRAPH_AVAILABLE = True
    logger.info("LangGraph available - workflow features enabled")
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("LangGraph not available. Advanced workflow features disabled.")

class WorkflowState(BaseModel):
    """State model for LangGraph workflows"""
    system_prompt: str = Field(description="System prompt for the conversation")
    user_message: str = Field(description="User message")
    response: str = Field(default="", description="AI response")
    processing_steps: List[str] = Field(default_factory=list, description="Processing steps taken")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class WorkflowStateDict(TypedDict):
    """Dictionary representation of WorkflowState for LangGraph"""
    system_prompt: str
    user_message: str
    response: str
    processing_steps: List[str]
    metadata: Dict[str, Any]

class BtpAICoreWorkflow:
    """
    Advanced BTP AI Core Service using LangGraph for complex workflows.
    Provides state management and multi-step processing capabilities.
    """
    
    def __init__(self, service: BtpAICoreService):
        self.service = service    
        # Create the workflow graph
        self.workflow = self._create_workflow()
        
    def _create_workflow(self) -> StateGraph:
        """Create a LangGraph workflow for processing requests"""
        
        def preprocess_node(state: dict) -> dict:
            """Preprocess the input and update state"""
            logger.info("Preprocessing input...")
            if "processing_steps" not in state:
                state["processing_steps"] = []
            state["processing_steps"].append("preprocessing")
            
            # Add any preprocessing logic here
            if not state.get("system_prompt"):
                state["system_prompt"] = "You are a helpful AI assistant."
            
            return state
        
        def generate_response_node(state: dict) -> dict:
            """Generate AI response using the service"""
            logger.info("Generating AI response...")
            if "processing_steps" not in state:
                state["processing_steps"] = []
            if "metadata" not in state:
                state["metadata"] = {}
            state["processing_steps"].append("generating_response")
            
            try:
                response = self.service.call_aicore_api(
                    system_prompt=state["system_prompt"],
                    user_message=state["user_message"]
                )
                state["response"] = response
                state["metadata"]["response_length"] = len(response)
                
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                state["response"] = f"Error: {str(e)}"
                state["metadata"]["error"] = str(e)
            
            return state
        
        def postprocess_node(state: dict) -> dict:
            """Postprocess the response"""
            logger.info("Postprocessing response...")
            if "processing_steps" not in state:
                state["processing_steps"] = []
            if "metadata" not in state:
                state["metadata"] = {}
            state["processing_steps"].append("postprocessing")
            
            # Add any postprocessing logic here
            if state.get("response") and not state["response"].endswith('.'):
                state["response"] += '.'
            
            state["metadata"]["final_processing_steps"] = len(state["processing_steps"])
            return state
        
        # Create the workflow graph
        workflow = StateGraph(WorkflowStateDict)
        
        # Add nodes
        workflow.add_node("preprocess", preprocess_node)
        workflow.add_node("generate", generate_response_node)
        workflow.add_node("postprocess", postprocess_node)
        
        # Set entry point and edges
        workflow.add_edge(START, "preprocess")
        workflow.add_edge("preprocess", "generate")
        workflow.add_edge("generate", "postprocess")
        workflow.add_edge("postprocess", END)
        
        return workflow.compile()
    
    def process_request(self, user_message: str, system_prompt: str = None) -> WorkflowState:
        """
        Process a request through the complete workflow.
        
        Args:
            user_message (str): User message to process
            system_prompt (str, optional): System prompt
            
        Returns:
            WorkflowState: Final state with response and metadata
        """
        initial_state = WorkflowState(
            user_message=user_message,
            system_prompt=system_prompt or "You are a helpful AI assistant."
        )
        
        logger.info(f"Starting workflow processing for message: {user_message[:50]}...")
        final_state_dict = self.workflow.invoke(initial_state.dict())
        # Ensure all required fields exist with defaults
        final_state_dict.setdefault("response", "")
        final_state_dict.setdefault("processing_steps", [])
        final_state_dict.setdefault("metadata", {})
        final_state = WorkflowState(**final_state_dict)
        logger.info(f"Workflow completed with {len(final_state.processing_steps)} steps")
        
        return final_state
    
    async def process_request_async(self, user_message: str, system_prompt: str = None) -> WorkflowState:
        """
        Async version of process_request.
        
        Args:
            user_message (str): User message to process
            system_prompt (str, optional): System prompt
            
        Returns:
            WorkflowState: Final state with response and metadata
        """
        initial_state = WorkflowState(
            user_message=user_message,
            system_prompt=system_prompt or "You are a helpful AI assistant."
        )
        
        logger.info(f"Starting async workflow processing for message: {user_message[:50]}...")
        final_state_dict = await self.workflow.ainvoke(initial_state.dict())
        # Ensure all required fields exist with defaults
        final_state_dict.setdefault("response", "")
        final_state_dict.setdefault("processing_steps", [])
        final_state_dict.setdefault("metadata", {})
        final_state = WorkflowState(**final_state_dict)
        logger.info(f"Async workflow completed with {len(final_state.processing_steps)} steps")
        
        return final_state
    
    def get_workflow_visualization(self) -> str:
        """Get a text representation of the workflow structure"""
        return """
        Workflow Structure:
        [Entry] -> preprocess -> generate -> postprocess -> [End]
        
        Nodes:
        - preprocess: Validates and prepares input
        - generate: Calls BTP AI Core for response generation  
        - postprocess: Cleans and formats the response
        """
    
def get_embedding(input_text, model="sentence-transformers/all-MiniLM-L6-v2", platform="btp"):
    """
    Fetches embedding for a given text using the specified model.
    """
    global EMBEDDING_MODEL
    # model = "embed-v4"
    print(f"Generating embedding for: {input_text[:50]}... using model: {model}")
    try:
        if platform == "btp":
            # Use the AI Core client for BTP
            # response = ai_core_client.embeddings.create(model_name=model, input=input_text)
            response = embeddings.create(model_name=model, input=input_text)
            # print(response)
            return response.data[0].embedding
        elif platform == "local":
            # Use local embedding model
            if not EMBEDDING_MODEL:
                EMBEDDING_MODEL = SentenceTransformer(model)
            embeddings_result = EMBEDDING_MODEL.encode([input_text], show_progress_bar=True)
            # print(f"Embedding generated: {embeddings_result[0][:10]}...")  # Print first 10 values for brevity
            return embeddings_result[0].tolist()  # Convert to list for compatibility with HANA
        else:
            raise ValueError("Unsupported platform. Use 'btp' or 'local'.")
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None
