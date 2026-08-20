import os
import pytest
from qdrant_client import QdrantClient
from src.core.state import CodeReference
from src.rag.indexer import (
    index_code_references,
    SPARSE_INDEX_PATH
)
from src.rag.retriever import (
    reciprocal_rank_fusion,
    hybrid_search
)

@pytest.fixture(autouse=True)
def run_around_tests():
    """Setup and teardown directories for tests."""
    os.makedirs("scratch", exist_ok=True)
    yield
    # Cleanup
    if os.path.exists(SPARSE_INDEX_PATH):
        os.remove(SPARSE_INDEX_PATH)

def test_reciprocal_rank_fusion_math():
    """Verify that RRF ranks matching documents in both lists higher."""
    # Document A is ranked 1st in dense, and 2nd in sparse
    doc_a = CodeReference(
        file_path="src/file_a.py",
        symbol_name="a",
        start_line=1,
        end_line=5,
        code_snippet="snippet a"
    )
    # Document B is ranked 2nd in dense, but not in sparse
    doc_b = CodeReference(
        file_path="src/file_b.py",
        symbol_name="b",
        start_line=1,
        end_line=5,
        code_snippet="snippet b"
    )
    # Document C is ranked 1st in sparse, but not in dense
    doc_c = CodeReference(
        file_path="src/file_c.py",
        symbol_name="c",
        start_line=1,
        end_line=5,
        code_snippet="snippet c"
    )

    dense_hits = [doc_a, doc_b]
    sparse_hits = [doc_c, doc_a]

    # Run fusion
    fused = reciprocal_rank_fusion(dense_hits, sparse_hits, k=60)
    
    # Assertions
    assert len(fused) == 3
    # doc_a appears in both (rank 1 and 2), so its RRF score should be highest:
    # Score A = 1/(60+1) + 1/(60+2) = 0.01639 + 0.01612 = 0.0325
    # Score C = 1/(60+1) = 0.01639
    # Score B = 1/(60+2) = 0.01612
    assert fused[0].symbol_name == "a"
    assert fused[1].symbol_name == "c"
    assert fused[2].symbol_name == "b"

def test_hybrid_search_end_to_end():
    """Verify end-to-end hybrid retrieval using Qdrant and BM25 indexers."""
    chunks = [
        CodeReference(
            file_path="src/auth.py",
            symbol_name="authenticate_user",
            start_line=1,
            end_line=10,
            code_snippet="def authenticate_user(username, password):\n    # verifies user credentials\n    return True"
        ),
        CodeReference(
            file_path="src/db.py",
            symbol_name="get_database_connection",
            start_line=1,
            end_line=5,
            code_snippet="def get_database_connection():\n    # returns connection pool from database pool\n    return connection_pool.get()"
        ),
        CodeReference(
            file_path="src/dummy1.py",
            symbol_name="dummy_one",
            start_line=1,
            end_line=2,
            code_snippet="def log_format():\n    pass"
        ),
        CodeReference(
            file_path="src/dummy2.py",
            symbol_name="dummy_two",
            start_line=1,
            end_line=2,
            code_snippet="def load_config():\n    return {}"
        ),
        CodeReference(
            file_path="src/dummy3.py",
            symbol_name="dummy_three",
            start_line=1,
            end_line=2,
            code_snippet="def send_request():\n    pass"
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

    # 1. Index the mock chunks
    index_code_references(chunks, qdrant_url=qdrant_url)

    # 2. Query exact keyword terms (matches BM25 sparse search)
    query_keyword = "database connection pool"
    results_keyword = hybrid_search(query_keyword, limit=3, qdrant_url=qdrant_url)
    
    assert len(results_keyword) > 0
    # The database connection chunk should be the first result
    assert results_keyword[0].symbol_name == "get_database_connection"

    # 3. Query semantic terms (matches Qdrant dense vector search)
    query_semantic = "user sign in verification credentials"
    results_semantic = hybrid_search(query_semantic, limit=3, qdrant_url=qdrant_url)
    
    assert len(results_semantic) > 0
    # The authenticate_user chunk should match semantically
    assert results_semantic[0].symbol_name == "authenticate_user"
