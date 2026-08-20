import os
import time
import pytest
import tempfile
import shutil
from qdrant_client import QdrantClient
from src.core.state import CodeReference
from src.rag.indexer import (
    index_code_references,
    load_sparse_index,
    COLLECTION_NAME,
    SPARSE_INDEX_PATH
)
from src.rag.retriever import hybrid_search
from src.rag.watcher import start_watcher

@pytest.fixture
def test_workspace():
    """Setup and teardown a temporary directory for watching tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    if os.path.exists(SPARSE_INDEX_PATH):
        os.remove(SPARSE_INDEX_PATH)

def test_watcher_incremental_indexing(test_workspace):
    """Verify that file creations, modifications, and deletions trigger index updates."""
    qdrant_url = "http://127.0.0.1:6333"
    client = QdrantClient(url=qdrant_url, timeout=30.0)

    try:
        client.get_collections()
    except Exception as e:
        pytest.fail(f"Qdrant not reachable: {e}")

    # Start with a clean Qdrant collection for test stability
    if COLLECTION_NAME in [c.name for c in client.get_collections().collections]:
        client.delete_collection(collection_name=COLLECTION_NAME)

    # 1. Initialize indices with base files
    initial_chunks = [
        CodeReference(
            file_path=os.path.join(test_workspace, "base.py").replace("\\", "/"),
            symbol_name="base_func",
            start_line=1,
            end_line=2,
            code_snippet="def base_func():\n    pass"
        )
    ]
    index_code_references(initial_chunks, qdrant_url=qdrant_url)

    # 2. Start codebase watcher
    observer = start_watcher(test_workspace, qdrant_url=qdrant_url)

    try:
        # A. TEST CREATION EVENT
        new_file_path = os.path.join(test_workspace, "helper.py").replace("\\", "/")
        code_created = "def helper_func():\n    print('hello helper')\n"
        
        with open(new_file_path, "w", encoding="utf-8") as f:
            f.write(code_created)
            
        # Give watchdog background observer thread 1.5 seconds to capture and index
        time.sleep(1.5)

        # Search for helper_func
        res1 = hybrid_search("helper_func", limit=2, qdrant_url=qdrant_url)
        assert len(res1) > 0
        assert any(r.symbol_name == "helper_func" for r in res1)

        # B. TEST MODIFICATION EVENT
        code_modified = "def renamed_helper():\n    print('modified helper')\n"
        with open(new_file_path, "w", encoding="utf-8") as f:
            f.write(code_modified)
            
        time.sleep(1.5)

        # Search for renamed symbol
        res2 = hybrid_search("renamed_helper", limit=2, qdrant_url=qdrant_url)
        assert len(res2) > 0
        assert any(r.symbol_name == "renamed_helper" for r in res2)
        
        # Search for old symbol (should be deleted / overwritten)
        res2_old = hybrid_search("helper_func", limit=5, qdrant_url=qdrant_url)
        assert not any(r.symbol_name == "helper_func" for r in res2_old)

        # C. TEST DELETION EVENT
        os.remove(new_file_path)
        time.sleep(1.5)

        # Search for renamed symbol (should be deleted now)
        res3 = hybrid_search("renamed_helper", limit=5, qdrant_url=qdrant_url)
        assert not any(r.symbol_name == "renamed_helper" for r in res3)

    finally:
        # Clean up background observer thread
        observer.stop()
        observer.join()
