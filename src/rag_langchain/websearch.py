from .config import logger, TAVILY_API_KEY
from langchain_community.tools.tavily_search import TavilySearchResults


web_search_tool = TavilySearchResults(k=3)