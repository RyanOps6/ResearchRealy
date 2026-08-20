import logging
from typing import Optional
import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

def scrape_via_jina(url: str) -> Optional[str]:
    """Queries Jina Reader API to get a formatted Markdown version of any URL."""
    jina_url = f"https://r.jina.ai/{url}"
    try:
        response = httpx.get(jina_url, timeout=15.0)
        if response.status_code == 200:
            return response.text
        else:
            logger.warning(f"Jina Reader returned status code {response.status_code} for {url}")
    except Exception as e:
        logger.warning(f"Jina Reader fetch failed for {url}: {e}")
    return None

def scrape_via_bs4(url: str) -> str:
    """Fetches raw HTML locally and parses it using BeautifulSoup, stripping boilerplate tags."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
        response = httpx.get(url, headers=headers, timeout=15.0)
        response.raise_for_status()
        html_content = response.text
        
        # Parse HTML
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Decompose non-content boilerplate elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe"]):
            element.decompose()
            
        # Extract text content
        raw_text = soup.get_text(separator="\n")
        
        # Clean extra spaces and empty lines
        cleaned_lines = []
        for line in raw_text.splitlines():
            line = line.strip()
            if line:
                cleaned_lines.append(line)
                
        return "\n".join(cleaned_lines)
    except Exception as e:
        logger.error(f"Local BeautifulSoup HTML scraper failed for {url}: {e}")
        return ""

def scrape_url_to_markdown(url: str) -> str:
    """
    Extracts text content from a web URL.
    Attempts Jina Reader first for markdown/SPA execution, falling back to local BeautifulSoup parsing.
    """
    if not url.startswith("http://") and not url.startswith("https://"):
        logger.warning(f"Invalid URL protocol provided for scraping: {url}")
        return ""

    markdown_content = scrape_via_jina(url)
    if markdown_content:
        return markdown_content

    logger.info(f"Jina Reader failed or unavailable. Using local BeautifulSoup fallback for: {url}")
    return scrape_via_bs4(url)
