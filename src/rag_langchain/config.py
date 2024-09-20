import os
import logging
from dotenv import load_dotenv
import openai

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_KEY')

# Set the default model to GPT-4 Turbo
DEFAULT_MODEL = "gpt-4-1106-preview"

# Check if the API key is set
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_KEY environment variable is not set")

# Log the loaded API key (first 5 and last 5 characters)
logger.info(f"Loaded OpenAI API Key: {OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-5:]}")

# You can add more configuration variables here if needed




# Set up Tavily API key
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found in environment variables")



