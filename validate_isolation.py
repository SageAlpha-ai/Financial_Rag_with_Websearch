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
    print("Initializing Metadata Isolation Validation...")
    
    # Test 1: Known Company (Dixon)
    dixon_query = "Dixon FY2023 revenue"
    print(f"\n--- Testing Known Company: {dixon_query} ---")
    try:
        req = QueryRequest(query=dixon_query)
        response = await _process_query(req)
        print(f"Response Type: {response.answer_type}")
        # Handle SourceItem object
        sources = [s.source for s in response.sources]
        print(f"Sources: {sources}")
        
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: Unknown Company (Oracle)
    oracle_query = "Oracle FY2023 revenue"
    print(f"\n--- Testing Unknown Company: {oracle_query} ---")
    try:
        req = QueryRequest(query=oracle_query)
        response = await _process_query(req)
        print(f"Response Type: {response.answer_type}")
        sources = [s.source for s in response.sources]
        print(f"Sources: {sources}") 

    except Exception as e:
        print(f"Error: {e}")

    # Test 3: No Company (EBITDA)
    ebitda_query = "What is EBITDA?"
    print(f"\n--- Testing No Company: {ebitda_query} ---")
    try:
        req = QueryRequest(query=ebitda_query)
        response = await _process_query(req)
        print(f"Response Type: {response.answer_type}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
