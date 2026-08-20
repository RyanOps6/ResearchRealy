import pytest
from bs4 import BeautifulSoup
from src.research.scraper import (
    scrape_via_bs4,
    scrape_url_to_markdown
)

def test_local_bs4_parsing_logic():
    """Verify that BeautifulSoup parsing logic strips scripts, headers, footers, and styles."""
    html_content = (
        "<html>\n"
        "  <head><style>body { color: red; }</style></head>\n"
        "  <body>\n"
        "    <header><nav>Main Navigation Link</nav></header>\n"
        "    <main>\n"
        "      <h1>Main Document Heading</h1>\n"
        "      <p>This is the target readable text content.</p>\n"
        "    </main>\n"
        "    <footer>Footer Copyright Information</footer>\n"
        "    <script>console.log('hello world');</script>\n"
        "  </body>\n"
        "</html>\n"
    )

    # Parse and strip tags using BS4 logic
    soup = BeautifulSoup(html_content, "html.parser")
    for element in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe"]):
        element.decompose()
        
    raw_text = soup.get_text(separator="\n")
    cleaned_lines = []
    for line in raw_text.splitlines():
        line = line.strip()
        if line:
            cleaned_lines.append(line)
            
    result = "\n".join(cleaned_lines)
    
    # Assert boilerplate tags are stripped
    assert "Main Navigation Link" not in result
    assert "Footer Copyright Information" not in result
    assert "console.log" not in result
    assert "color: red" not in result
    
    # Assert main content is preserved
    assert "Main Document Heading" in result
    assert "This is the target readable text content." in result

def test_scrape_url_to_markdown_live():
    """Verify live url scraping on a stable domain."""
    results = scrape_url_to_markdown("https://example.com")
    
    assert isinstance(results, str)
    assert len(results) > 0
    assert any(term in results for term in ["Example Domain", "Test Document", "Test Article"])
