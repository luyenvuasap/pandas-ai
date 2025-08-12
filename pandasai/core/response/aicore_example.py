#!/usr/bin/env python3
"""
Example usage of the refactored BtpAICoreService with LangChain/LangGraph
"""
import dotenv
dotenv.load_dotenv()  # Load environment variables from .env file
import asyncio
import os
try:
    from aicore_utils import BtpAICoreService, BtpAICoreWorkflow, LANGGRAPH_AVAILABLE
except ImportError as e:
    print(f"Import error: {e}")
    from aicore_utils import BtpAICoreService
    LANGGRAPH_AVAILABLE = False

def example_basic_usage():
    """Example of basic usage with the refactored service"""
    print("=== Basic Usage Example ===")
    
    # Initialize the service
    service = BtpAICoreService(
        model_name="gpt-4o-mini",
        temperature=0.7,
        max_tokens=500
    )
    
    # Simple API call
    system_prompt = "You are a helpful AI assistant specializing in Python programming."
    user_message = "Explain the benefits of using LangChain in AI applications."
    
    try:
        response = service.call_aicore_api(system_prompt, user_message)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Get chain configuration
    config = service.get_chain_config()
    print(f"Configuration: {config}")

async def example_streaming():
    """Example of streaming functionality"""
    print("\n=== Streaming Example ===")
    
    service = BtpAICoreService()
    
    system_prompt = "You are a technical writer."
    user_message = "Write a brief introduction to microservices architecture."
    
    try:
        print("Streaming response:")
        async for chunk in service.call_aicore_api_stream(system_prompt, user_message):
            print(chunk, end="", flush=True)
        print()  # New line after streaming
    except Exception as e:
        print(f"Error: {e}")

def example_batch_processing():
    """Example of batch processing"""
    print("\n=== Batch Processing Example ===")
    
    service = BtpAICoreService()
    
    inputs = [
        {
            "system_prompt": "You are a code reviewer.",
            "user_message": "Review this Python function: def add(a, b): return a + b"
        },
        {
            "system_prompt": "You are a documentation writer.",
            "user_message": "Write a docstring for a function that calculates factorial."
        }
    ]
    
    try:
        responses = service.batch_process(inputs)
        for i, response in enumerate(responses):
            print(f"Response {i+1}: {response}")
    except Exception as e:
        print(f"Error: {e}")

def example_custom_chain():
    """Example of creating custom chains"""
    print("\n=== Custom Chain Example ===")
    
    service = BtpAICoreService()
    
    # Create a custom prompt template
    custom_template = """
    You are an expert in {domain}.
    
    Question: {question}
    
    Please provide a detailed answer with examples.
    """
    
    # Create custom chain
    custom_chain = service.create_custom_chain(custom_template)
    
    try:
        response = custom_chain.invoke({
            "domain": "machine learning",
            "question": "What is the difference between supervised and unsupervised learning?"
        })
        print(f"Custom chain response: {response}")
    except Exception as e:
        print(f"Error: {e}")

def example_langgraph_workflow():
    """Example of using LangGraph workflow (if available)"""
    print("\n=== LangGraph Workflow Example ===")
    
    if not LANGGRAPH_AVAILABLE:
        print("LangGraph not available. Install langgraph to use workflow features.")
        return

    try:
        # Initialize basic service first
        service = BtpAICoreService()
        
        # Create workflow
        workflow = BtpAICoreWorkflow(service)
        
        # Process a request through the workflow
        result = workflow.process_request(
            user_message="Explain the benefits of using workflows in AI applications.",
            system_prompt="You are an AI architecture expert."
        )
        
        print(f"Final Response: {result.response}")
        print(f"Processing Steps: {result.processing_steps}")
        print(f"Metadata: {result.metadata}")
        
        # Show workflow structure
        print(workflow.get_workflow_visualization())
        
    except Exception as e:
        print(f"Error: {e}")

async def example_async_workflow():
    """Example of async workflow processing"""
    print("\n=== Async Workflow Example ===")
    
    if not LANGGRAPH_AVAILABLE:
        print("LangGraph not available. Install langgraph to use workflow features.")
        return

    try:
        service = BtpAICoreService()
        workflow = BtpAICoreWorkflow(service)
        
        result = await workflow.process_request_async(
            user_message="What are the best practices for async programming in Python?",
            system_prompt="You are a Python expert."
        )
        
        print(f"Async Final Response: {result.response}")
        print(f"Processing Steps: {result.processing_steps}")
        print(f"Metadata: {result.metadata}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()

def main():
    """Run all examples"""
    print("BTP AI Core Service - LangChain/LangGraph Examples")
    print("=" * 50)
    
    # Check environment variables
    required_vars = ["AICORE_AUTH_URL", "AICORE_CLIENT_ID", "AICORE_CLIENT_SECRET", "AICORE_BASE_URL"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"Missing environment variables: {missing_vars}")
        print("Please set these variables before running the examples.")
        return
    
    # Run examples
    example_basic_usage()
    
    # Run async examples
    asyncio.run(example_streaming())
    
    example_batch_processing()
    example_custom_chain()
    
    if LANGGRAPH_AVAILABLE:
        example_langgraph_workflow()
        # Run async workflow example
        asyncio.run(example_async_workflow())
    else:
        print("\n=== LangGraph Features Skipped ===")
        print("LangGraph not available. Install with: pip install langgraph")

if __name__ == "__main__":
    main()
