import sys
import os
import logging
import asyncio

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

from app import _process_query, QueryRequest

async def main():
    print("Initializing Report Generation Validation...")
    
    # Test Report Query
    query = "Generate a detailed investment report on Dixon."
    request = QueryRequest(query=query)
    
    print(f"\n--- Testing Report Query: {query} ---")
    try:
        response = await _process_query(request)
        print(f"\nResponse Type: {response.answer_type}")
        print(f"Sources Count: {len(response.sources)}")
        print(f"Answer Preview: {response.answer[:200]}...")
        
        if response.answer_type == "REPORT":
            print("SUCCESS: Report generated via Orchestrator pipeline.")
        else:
            print(f"FAILURE: Expected REPORT, got {response.answer_type}")
            
    except Exception as e:
        print(f"Error processing query '{query}': {e}")

if __name__ == "__main__":
    asyncio.run(main())
