import sys
import os
import logging

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

from rag.langchain_orchestrator import LangChainOrchestrator

def main():
    print("Initializing Orchestrator...")
    try:
        orchestrator = LangChainOrchestrator()
    except Exception as e:
        print(f"Failed to initialize orchestrator: {e}")
        return

    questions = [
        "Compare FY2023 revenue and current stock price",
        "What is EBITDA?",
        "What was Dixon revenue in FY2022?"
    ]

    for q in questions:
        print(f"\n--- Testing Query: {q} ---")
        try:
            # We only care about the log output which will show [INTENT]
            # but we need to run the method to trigger it.
            orchestrator.answer_query(q)
        except Exception as e:
            print(f"Error processing query '{q}': {e}")

if __name__ == "__main__":
    main()
