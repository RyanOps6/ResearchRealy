import os
import logging
import pickle
from typing import List, Tuple, Dict, Any, Optional
from qdrant_client import QdrantClient
from src.core.state import CodeReference
from src.rag.indexer import (
    get_dense_embedding,
    tokenize_code,
    load_sparse_index,
    COLLECTION_NAME,
    QDRANT_URL
)

logger = logging.getLogger(__name__)

def retrieve_dense(query: str, limit: int = 5, qdrant_url: str = QDRANT_URL) -> List[CodeReference]:
    """Retrieves the top dense match CodeReferences from the local Qdrant container."""
    try:
        client = QdrantClient(url=qdrant_url, timeout=30.0)
        # Check if collection exists
        collections = client.get_collections().collections
        if COLLECTION_NAME not in [c.name for c in collections]:
            logger.warning(f"Qdrant collection '{COLLECTION_NAME}' does not exist.")
            return []

        query_vector = get_dense_embedding(query)
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit
        )

        hits = []
        for r in results:
            ref = CodeReference(
                file_path=r.payload["file_path"],
                symbol_name=r.payload["symbol_name"],
                start_line=r.payload["start_line"],
                end_line=r.payload["end_line"],
                code_snippet=r.payload["code_snippet"]
            )
            hits.append(ref)
        return hits
    except Exception as e:
        logger.warning(f"Dense vector retrieval failed: {e}. Falling back to empty dense results.")
        return []

def retrieve_sparse(query: str, limit: int = 5) -> List[CodeReference]:
    """Retrieves the top sparse match CodeReferences using the BM25 index on disk."""
    loaded = load_sparse_index()
    if not loaded:
        logger.warning("BM25 sparse index not found or failed to load.")
        return []

    bm25, chunks = loaded
    tokenized_query = tokenize_code(query)
    scores = bm25.get_scores(tokenized_query)

    scored_chunks = list(zip(chunks, scores))
    # Filter out non-matches (score <= 0.0)
    matching_chunks = [sc for sc in scored_chunks if sc[1] > 0.0]
    matching_chunks.sort(key=lambda x: x[1], reverse=True)

    return [mc[0] for mc in matching_chunks[:limit]]

def reciprocal_rank_fusion(
    dense_hits: List[CodeReference],
    sparse_hits: List[CodeReference],
    k: int = 60
) -> List[CodeReference]:
    """
    Merges dense and sparse search lists using the Reciprocal Rank Fusion (RRF) algorithm.
    RRF Score = Sum( 1 / (k + rank) )
    """
    rrf_scores = {}
    chunk_map = {}

    def fuse_list(hits: List[CodeReference]):
        for rank, chunk in enumerate(hits):
            # Unique identifier for the chunk
            key = (chunk.file_path, chunk.start_line, chunk.end_line)
            chunk_map[key] = chunk
            # 1-indexed rank position
            position = rank + 1
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (k + position))

    fuse_list(dense_hits)
    fuse_list(sparse_hits)

    # Sort keys by fusion score descending
    sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return [chunk_map[key] for key in sorted_keys]

def hybrid_search(query: str, limit: int = 5, qdrant_url: str = QDRANT_URL) -> List[CodeReference]:
    """
    Performs hybrid search combining dense and sparse indices.
    Fuses findings using Reciprocal Rank Fusion (RRF).
    """
    # Fetch double the limit from each index to ensure good fusion candidates
    fetch_limit = limit * 2

    dense_hits = retrieve_dense(query, limit=fetch_limit, qdrant_url=qdrant_url)
    sparse_hits = retrieve_sparse(query, limit=fetch_limit)

    if not dense_hits and not sparse_hits:
        logger.info("Both dense and sparse retrieval returned 0 results.")
        return []

    fused_results = reciprocal_rank_fusion(dense_hits, sparse_hits)
    return fused_results[:limit]
