import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from rag_langchain.graph import rag_pipeline
from rag_langchain.nodes import GraphState

def run_rag(question: str):
    state = GraphState(question=question)
    step_counter = 0
    max_steps = 50  # Prevent infinite loops

    for output in rag_pipeline.stream(state):
        step_counter += 1
        current_step = output.get("current_step", "Unknown")
        logger.info(f"Step {step_counter}: {current_step}")
        logger.info(f"Current state: {state.dict()}")

        if current_step == "END":
            logger.info("Reached END state. Finalizing answer.")
            return state.generation or "Failed to generate an answer."

        if step_counter >= max_steps:
            logger.warning("Maximum steps reached without reaching END.")
            return "Failed to generate an answer."

        # Optional: Detailed state logging
        logger.debug(f"Step {step_counter}: {current_step}, State: {state}")

    return "Failed to generate an answer."

if __name__ == "__main__":
    question = input("Enter your question: ")
    answer = run_rag(question)
    print(f"Answer: {answer}")