# ai_fintech_search/tools/web_search.py
from typing import List, Dict, Any
from langchain_tavily import TavilySearch            
from ai_fintech_search import config                       
config.ensure_required_env(["TAVILY_API_KEY"])             

_tavily = TavilySearch(
    max_results=30,
    tavily_api_key=config.TAVILY_API_KEY                    
)

def web_search(query: str) -> List[Dict[str, Any]]:
    return _tavily.invoke({"query": query})
