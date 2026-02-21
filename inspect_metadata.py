import sys
import os
import logging
from vectorstore.chroma_client import get_collection

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)

def check_metadata():
    print("Checking metadata for 'Dixon' query...")
    try:
        col = get_collection("dixon-technologies")
        # Use .get() which doesn't need embeddings
        results = col.get(limit=1, include=["metadatas"])
        metas = results['metadatas']
        if metas and len(metas) > 0:
            m = metas[0]
            print(f"Keys: {list(m.keys())}")
            print(f"company: {m.get('company')}")
            print(f"company_name: {m.get('company_name')}")
        else:
            print("No documents found in dixon-technologies collection.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_metadata()
