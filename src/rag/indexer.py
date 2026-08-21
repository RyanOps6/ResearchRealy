import os
import re
import logging
import pickle
import hashlib
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import litellm
from qdrant_client import QdrantClient
from qdrant_client.http import models
from rank_bm25 import BM25Okapi
from src.core.state import CodeReference

logger = logging.getLogger(__name__)

# Default locations
SPARSE_INDEX_PATH = os.path.join("scratch", "bm25_index.pkl")
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION_NAME = "code_chunks"

def get_mock_embedding(text: str, dimension: int = 1536) -> List[float]:
    """Generates a deterministic unit-length mock embedding vector based on text hash."""
    hash_val = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(hash_val[:4], byteorder="big")
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(dimension)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.tolist()

def get_dense_embedding(text: str) -> List[float]:
    """
    Generates a dense vector embedding using litellm.
    Falls back to deterministic mock embeddings if credentials/APIs are missing.
    """
    has_api_key = any(
        os.getenv(k) and os.getenv(k) != "your-api-key-here" for k in [
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "ANTHROPIC_API_KEY",
            "DEEPSEEK_API_KEY"
        ]
    )

    is_testing = "PYTEST_CURRENT_TEST" in os.environ
    if is_testing or not has_api_key:
        return get_mock_embedding(text, 1536)

    try:
        model = os.getenv("LITELLM_EMBEDDING_MODEL")
        api_base = os.getenv("OPENAI_API_BASE")
        api_key = os.getenv("OPENAI_API_KEY")

        if not model:
            # Map default embedding models based on endpoint URL
            if api_base and "nvidia" in api_base.lower():
                model = "nvidia/embeddings-nv-embed-qa-4"
            else:
                model = "text-embedding-3-small"

        # Apply prefix for custom endpoints if needed
        if api_base and "nvidia.com" in api_base.lower() and not model.startswith("openai/"):
            model = f"openai/{model}"

        response = litellm.embedding(
            model=model,
            input=[text],
            api_base=api_base,
            api_key=api_key
        )
        return response.data[0]["embedding"]
    except Exception as e:
        logger.warning(f"Failed to fetch embedding from provider: {e}. Using offline mock embeddings.")
        return get_mock_embedding(text, 1536)

def tokenize_code(code: str) -> List[str]:
    """Tokenizes code by extracting alphanumeric words, converted to lowercase."""
    return re.findall(r'[a-zA-Z0-9]+', code.lower())

def build_sparse_index(chunks: List[CodeReference], path: str = SPARSE_INDEX_PATH) -> BM25Okapi:
    """Builds and serializes a BM25 sparse lexical index on disk."""
    corpus = [tokenize_code(chunk.code_snippet) for chunk in chunks]
    bm25 = BM25Okapi(corpus)
    
    # Save the index and the chunk references together
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    logger.info(f"Successfully saved BM25 sparse index with {len(chunks)} chunks to {path}")
    return bm25

def load_sparse_index(path: str = SPARSE_INDEX_PATH) -> Optional[Tuple[BM25Okapi, List[CodeReference]]]:
    """Loads the serialized BM25 index from disk."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "wb" if not os.path.exists(path) else "rb") as f:
            data = pickle.load(f)
            return data["bm25"], data["chunks"]
    except Exception as e:
        logger.error(f"Failed to load sparse index from {path}: {e}")
        return None

def index_code_references(chunks: List[CodeReference], qdrant_url: str = QDRANT_URL) -> None:
    """
    Builds the sparse BM25 index on disk and uploads dense embeddings to local Qdrant.
    """
    if not chunks:
        logger.warning("No code chunks provided to index.")
        return

    # 1. Build local BM25 Index
    build_sparse_index(chunks)

    # 2. Upload Dense Vectors to Qdrant
    try:
        client = QdrantClient(url=qdrant_url, timeout=30.0)
        
        # Test connection or query collection
        client.get_collections()
        
        # Measure vector dimension dynamically from the first chunk
        sample_vector = get_dense_embedding(chunks[0].code_snippet)
        dim = len(sample_vector)
        
        # Ensure collection exists with matching dimension
        if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
            )
            logger.info(f"Created Qdrant collection '{COLLECTION_NAME}' with dimension {dim}")

        points = []
        for idx, chunk in enumerate(chunks):
            vector = get_dense_embedding(chunk.code_snippet)
            
            # Payload matching CodeReference schema
            payload = {
                "file_path": chunk.file_path,
                "symbol_name": chunk.symbol_name,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "code_snippet": chunk.code_snippet
            }
            
            points.append(models.PointStruct(
                id=idx,
                vector=vector,
                payload=payload
            ))

        # Upload points
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        logger.info(f"Successfully uploaded {len(chunks)} vectors to Qdrant collection '{COLLECTION_NAME}'")
        
    except Exception as e:
        logger.error(f"Failed to upload dense vectors to Qdrant at {qdrant_url}: {e}")
        raise e
