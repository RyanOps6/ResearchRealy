import os
import time
import tempfile
import pytest
from qdrant_client import QdrantClient
from src.core.state import CodeReference
from src.rag.indexer import index_code_references, SPARSE_INDEX_PATH
from src.rag.watcher import start_watcher
from src.rag.retriever import hybrid_search

COLLECTION_NAME = "code_chunks"

@pytest.fixture
def test_workspace():
    """Create and clean up a temporary workspace directory for watcher testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

def test_watcher_incremental_indexing(test_workspace):
    """Verify that file creations, modifications, and deletions trigger index updates."""
    # 0. Clean sparse index file on disk to prevent cross-test leakage
    if os.path.exists(SPARSE_INDEX_PATH):
        try:
            os.remove(SPARSE_INDEX_PATH)
        except Exception:
            pass

    qdrant_url = "http://127.0.0.1:6333"
    client = QdrantClient(url=qdrant_url, timeout=30.0)

    try:
        client.get_collections()
    except Exception as e:
        pytest.fail(f"Qdrant not reachable: {e}")

    # Start with a clean Qdrant collection for test stability
    if COLLECTION_NAME in [c.name for c in client.get_collections().collections]:
        client.delete_collection(collection_name=COLLECTION_NAME)
        time.sleep(1.0)

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
            
        # Poll search until symbol is indexed (max 4.0 seconds)
        indexed = False
        for _ in range(40):
            res1 = hybrid_search("helper_func", limit=2, qdrant_url=qdrant_url)
            if len(res1) > 0 and any(r.symbol_name == "helper_func" for r in res1):
                indexed = True
                break
            time.sleep(0.1)
        assert indexed is True

        # B. TEST MODIFICATION EVENT
        code_modified = "def renamed_helper():\n    print('modified helper')\n"
        with open(new_file_path, "w", encoding="utf-8") as f:
            f.write(code_modified)
            
        # Poll search until modification is indexed (max 4.0 seconds)
        modified = False
        for _ in range(40):
            res2 = hybrid_search("renamed_helper", limit=2, qdrant_url=qdrant_url)
            res2_old = hybrid_search("helper_func", limit=5, qdrant_url=qdrant_url)
            has_new = len(res2) > 0 and any(r.symbol_name == "renamed_helper" for r in res2)
            has_old = any(r.symbol_name == "helper_func" for r in res2_old)
            if has_new and not has_old:
                modified = True
                break
            time.sleep(0.1)
        assert modified is True

        # C. TEST DELETION EVENT
        os.remove(new_file_path)
        
        # Poll search until deletion is indexed (max 4.0 seconds)
        deleted = False
        for _ in range(40):
            res3 = hybrid_search("renamed_helper", limit=5, qdrant_url=qdrant_url)
            if not any(r.symbol_name == "renamed_helper" for r in res3):
                deleted = True
                break
            time.sleep(0.1)
        assert deleted is True

    finally:
        # Clean up background observer thread
        observer.stop()
        observer.join()
