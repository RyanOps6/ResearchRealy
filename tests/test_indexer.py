import os
import pytest
import shutil
from qdrant_client import QdrantClient
from src.core.state import CodeReference
from src.rag.indexer import (
    build_sparse_index,
    load_sparse_index,
    index_code_references,
    tokenize_code,
    COLLECTION_NAME,
    SPARSE_INDEX_PATH
)

@pytest.fixture(autouse=True)
def run_around_tests():
    """Setup and teardown directories for tests."""
    # Setup scratch directory
    os.makedirs("scratch", exist_ok=True)
    yield
    # Cleanup
    if os.path.exists(SPARSE_INDEX_PATH):
        os.remove(SPARSE_INDEX_PATH)

def test_tokenize_code():
    """Verify code tokenizer splits symbols and formats lower-case."""
    tokens = tokenize_code("def add_user(username, email_address):")
    assert "def" in tokens
    assert "add" in tokens
    assert "user" in tokens
    assert "username" in tokens
    assert "email" in tokens
    assert "address" in tokens
    # check that symbols/parentheses are ignored
    assert "(" not in tokens
    assert ":" not in tokens

def test_sparse_bm25_indexing():
    """Verify that BM25 sparse indexing creates the index, saves to disk, and ranks queries."""
    chunks = [
        CodeReference(
            file_path="src/auth.py",
            symbol_name="authenticate_user",
            start_line=1,
            end_line=10,
            code_snippet="def authenticate_user(username, password):\n    return True"
        ),
        CodeReference(
            file_path="src/db.py",
            symbol_name="get_database_connection",
            start_line=1,
            end_line=5,
            code_snippet="def get_database_connection():\n    return connection_pool.get()"
        ),
        # Add dummy chunks to enlarge the corpus (avoids 0 IDF math on small test sets)
        CodeReference(
            file_path="src/dummy1.py",
            symbol_name="dummy_func_one",
            start_line=1,
            end_line=2,
            code_snippet="def format_log():\n    print('formatting logs')"
        ),
        CodeReference(
            file_path="src/dummy2.py",
            symbol_name="dummy_func_two",
            start_line=1,
            end_line=2,
            code_snippet="def read_json_data():\n    return {}"
        ),
        CodeReference(
            file_path="src/dummy3.py",
            symbol_name="dummy_func_three",
            start_line=1,
            end_line=2,
            code_snippet="def send_request():\n    pass"
        )
    ]
    
    # 1. Build and save index
    bm25 = build_sparse_index(chunks)
    assert os.path.exists(SPARSE_INDEX_PATH)
    
    # 2. Test search ranking
    query = tokenize_code("database connection pool")
    scores = bm25.get_scores(query)
    
    # The database chunk (index 1) should score higher than the auth chunk (index 0)
    assert scores[1] > scores[0]

def test_qdrant_dense_indexing():
    """Verify Qdrant dense vector uploads and collection provisioning."""
    chunks = [
        CodeReference(
            file_path="src/math.py",
            symbol_name="multiply",
            start_line=1,
            end_line=5,
            code_snippet="def multiply(x, y):\n    return x * y"
        )
    ]
    
    qdrant_url = "http://127.0.0.1:6333"
    client = QdrantClient(url=qdrant_url, timeout=30.0)
    
    try:
        # Check connection
        client.get_collections()
    except Exception as e:
        pytest.fail(
            f"Failed to connect to Qdrant at {qdrant_url}. "
            f"Is the Qdrant Docker container running? "
            f"Please run 'docker compose up -d qdrant' in your terminal.\n"
            f"Error details: {e}"
        )
        
    # 1. Index the chunk
    index_code_references(chunks, qdrant_url=qdrant_url)
    
    # 2. Verify collection exists
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    assert COLLECTION_NAME in collection_names
    
    # 3. Retrieve points and verify payloads
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=10,
        with_payload=True,
        with_vectors=False
    )
    
    assert len(points) > 0
    # Search for our indexed file in scroll results
    matched_point = next((p for p in points if p.payload.get("file_path") == "src/math.py"), None)
    assert matched_point is not None
    assert matched_point.payload["symbol_name"] == "multiply"
    assert "def multiply" in matched_point.payload["code_snippet"]
