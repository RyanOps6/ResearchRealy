import os
import pytest
import dotenv
from src.research.search_client import (
    query_duckduckgo,
    query_tavily,
    search_web
)

# Load environment variables from .env
dotenv.load_dotenv()

def test_duckduckgo_search_lookup():
    """Verify that DuckDuckGo anonymous lookup retrieves search hits or handles rate limits gracefully."""
    try:
        results = query_duckduckgo("Python programming language tutorial", limit=3)
        assert isinstance(results, list)
        
        # If rate limit didn't hit, check keys
        if len(results) > 0:
            for r in results:
                assert "title" in r
                assert "url" in r
                assert "snippet" in r
                assert r["url"].startswith("http")
        else:
            pytest.skip("DuckDuckGo search returned 0 results (potentially rate-limited).")
    except Exception as e:
        if "Ratelimit" in str(e) or "202" in str(e):
            pytest.skip("DuckDuckGo search rate limited by provider.")
        else:
            raise e

def test_tavily_search_integration():
    """Verify Tavily search endpoint if a valid API key is present in environment."""
    api_key = os.getenv("TAVILY_API_KEY")
    
    # Run only if a real key (non-placeholder) is provided
    if not api_key or api_key == "your-tavily-api-key-here":
        pytest.skip("Skipping Tavily integration test: No valid TAVILY_API_KEY found.")

    results = query_tavily("Guido van Rossum Python creator", api_key, limit=2)
    assert isinstance(results, list)
    assert len(results) > 0
    for r in results:
        assert "title" in r
        assert "url" in r
        assert "snippet" in r
        assert r["url"].startswith("http")

def test_search_web_routing():
    """Verify the unified search_web routing works transparently."""
    results = search_web("FastAPI python documentation", limit=3)
    
    assert isinstance(results, list)
    # If Tavily runs successfully, we must get results. If fallback runs and gets rate-limited, skip.
    if len(results) == 0:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key or api_key == "your-tavily-api-key-here":
            pytest.skip("Web search fallback was rate limited.")
        else:
            pytest.fail("Web search returned no results despite Tavily API key configured.")
    else:
        assert "title" in results[0]
        assert "url" in results[0]
        assert "snippet" in results[0]
