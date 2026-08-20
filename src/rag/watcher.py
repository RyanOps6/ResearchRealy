import os
import time
import logging
import uuid
from typing import Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.rag.ast_parser import parse_python_file
from src.rag.indexer import (
    load_sparse_index,
    build_sparse_index,
    get_dense_embedding,
    COLLECTION_NAME,
    QDRANT_URL
)

logger = logging.getLogger(__name__)

def process_file_update(file_path: str, qdrant_url: str = QDRANT_URL) -> None:
    """Re-indexes a created or modified Python file in Qdrant and BM25."""
    # Standardize path slashes for consistency
    normalized_path = os.path.normpath(file_path).replace("\\", "/")
    
    if not os.path.exists(file_path):
        process_file_deletion(normalized_path, qdrant_url)
        return

    logger.info(f"Incremental indexing triggered for update: {normalized_path}")
    
    # 1. Parse AST chunks from updated file
    new_chunks = parse_python_file(file_path)
    
    # 2. Update Qdrant Collection
    try:
        client = QdrantClient(url=qdrant_url, timeout=30.0)
        collections = client.get_collections().collections
        if COLLECTION_NAME in [c.name for c in collections]:
            # Delete any existing points matching this file_path
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_path",
                            match=models.MatchValue(value=normalized_path)
                        )
                    ]
                )
            )
            
            # Insert the new vector points
            if new_chunks:
                points = []
                for idx, chunk in enumerate(new_chunks):
                    vector = get_dense_embedding(chunk.code_snippet)
                    
                    # Ensure path matches payload standard
                    chunk.file_path = normalized_path
                    payload = {
                        "file_path": chunk.file_path,
                        "symbol_name": chunk.symbol_name,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "code_snippet": chunk.code_snippet
                    }
                    
                    # Generate stable point UUID
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{normalized_path}_{idx}"))
                    points.append(models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    ))
                
                client.upsert(collection_name=COLLECTION_NAME, points=points)
                logger.info(f"Upserted {len(new_chunks)} chunks for {normalized_path} to Qdrant")
    except Exception as e:
        logger.warning(f"Failed to update Qdrant vectors for {normalized_path}: {e}")

    # 3. Update BM25 index on disk
    loaded = load_sparse_index()
    if loaded:
        bm25, old_chunks = loaded
        # Filter out old entries for this file
        updated_chunks = [c for c in old_chunks if os.path.normpath(c.file_path).replace("\\", "/") != normalized_path]
        updated_chunks.extend(new_chunks)
        build_sparse_index(updated_chunks)
    else:
        build_sparse_index(new_chunks)

def process_file_deletion(file_path: str, qdrant_url: str = QDRANT_URL) -> None:
    """Removes a deleted Python file from search indices."""
    normalized_path = os.path.normpath(file_path).replace("\\", "/")
    logger.info(f"Incremental indexing triggered for deletion: {normalized_path}")

    # 1. Delete points from Qdrant
    try:
        client = QdrantClient(url=qdrant_url, timeout=30.0)
        collections = client.get_collections().collections
        if COLLECTION_NAME in [c.name for c in collections]:
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_path",
                            match=models.MatchValue(value=normalized_path)
                        )
                    ]
                )
            )
            logger.info(f"Deleted vectors matching {normalized_path} from Qdrant")
    except Exception as e:
        logger.warning(f"Failed to delete Qdrant vectors for {normalized_path}: {e}")

    # 2. Remove from BM25 sparse index
    loaded = load_sparse_index()
    if loaded:
        bm25, old_chunks = loaded
        updated_chunks = [c for c in old_chunks if os.path.normpath(c.file_path).replace("\\", "/") != normalized_path]
        build_sparse_index(updated_chunks)

class CodebaseWatcherHandler(FileSystemEventHandler):
    """Event handler for monitoring changes in the Python source directory."""
    def __init__(self, qdrant_url: str = QDRANT_URL):
        super().__init__()
        self.qdrant_url = qdrant_url

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(".py"):
            process_file_update(event.src_path, self.qdrant_url)

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".py"):
            process_file_update(event.src_path, self.qdrant_url)

    def on_deleted(self, event):
        if not event.is_directory and event.src_path.endswith(".py"):
            process_file_deletion(event.src_path, self.qdrant_url)

    def on_moved(self, event):
        # Renaming/Moving: delete the old path and update the new path
        if not event.is_directory:
            if event.src_path.endswith(".py"):
                process_file_deletion(event.src_path, self.qdrant_url)
            if event.dest_path.endswith(".py"):
                process_file_update(event.dest_path, self.qdrant_url)

def start_watcher(path: str, qdrant_url: str = QDRANT_URL) -> Observer:
    """Spawns a filesystem observer thread to monitor the target path."""
    event_handler = CodebaseWatcherHandler(qdrant_url=qdrant_url)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    logger.info(f"Background filesystem watcher started for path: {path}")
    return observer
