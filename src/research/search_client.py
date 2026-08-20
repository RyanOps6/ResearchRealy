import os
import logging
from typing import List, Dict
import httpx
from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)

def query_tavily(query: str, api_key: str, limit: int = 5) -> List[Dict[str, str]]:
    """Performs web search using the Tavily Search API."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": limit
    }
    
    response = httpx.post("https://api.tavily.com/search", json=payload, headers=headers, timeout=15.0)
    response.raise_for_status()
    data = response.json()

    results = []
    for r in data.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", "")
        })
    return results

def query_duckduckgo(query: str, limit: int = 5) -> List[Dict[str, str]]:
    """Performs an anonymous free web search using DuckDuckGo."""
    results = []
    try:
        with DDGS() as ddgs:
            ddg_results = list(ddgs.text(query, max_results=limit))
            for r in ddg_results:
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                })
    except Exception as e:
        logger.error(f"DuckDuckGo search lookup failed: {e}")
    return results

def search_web(query: str, limit: int = 5) -> List[Dict[str, str]]:
    """
    Orchestrates the web search query routing.
    Queries Tavily if a valid API key is present, otherwise falls back to DuckDuckGo.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if api_key and api_key != "your-tavily-api-key-here":
        try:
            logger.info(f"Executing Tavily web search for: '{query}'")
            return query_tavily(query, api_key, limit)
        except Exception as e:
            logger.warning(f"Tavily search failed: {e}. Falling back to DuckDuckGo search.")

    logger.info(f"Executing DuckDuckGo fallback web search for: '{query}'")
    return query_duckduckgo(query, limit)
