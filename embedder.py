"""
embedder.py — Text to Semantic Embeddings (Dual-Mode)
=======================================================
Supports two embedding backends:
  1. Local  — sentence-transformers 'all-MiniLM-L6-v2' (free, 384-dim, fast)
  2. NVIDIA — llama-nemotron-embed-1b-v2 via API (higher quality, ~4096-dim)

Set EMBEDDING_BACKEND=nvidia in env to use NVIDIA API.
Default: local (no API cost).
"""

import os
import streamlit as st
import numpy as np
import requests
from sentence_transformers import SentenceTransformer


# ─── Configuration ────────────────────────────────────────────────────────────
EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "local").lower()


# ─── Local Model ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the sentence-transformer model (cached across Streamlit reruns)."""
    return SentenceTransformer('all-MiniLM-L6-v2')


# ─── Public API ──────────────────────────────────────────────────────────────

def get_embedding(text: str, input_type: str = "passage") -> np.ndarray:
    """
    Generate a semantic embedding vector for the given text.

    Args:
        text: Input text to embed
        input_type: 'query' for job descriptions, 'passage' for resumes
                    (only matters for NVIDIA backend)

    Returns:
        numpy array — the embedding vector
    """
    backend = os.environ.get("EMBEDDING_BACKEND", EMBEDDING_BACKEND).lower()
    if backend == "nvidia":
        return _get_nvidia_embedding(text, input_type)
    return _get_local_embedding(text)


def get_embeddings_batch(texts: list[str], input_type: str = "passage") -> list[np.ndarray]:
    """
    Generate embeddings for multiple texts in a single batch.

    Args:
        texts: List of text strings to embed
        input_type: 'query' or 'passage' (NVIDIA only)

    Returns:
        List of numpy arrays (embeddings)
    """
    backend = os.environ.get("EMBEDDING_BACKEND", EMBEDDING_BACKEND).lower()
    if backend == "nvidia":
        return _get_nvidia_embeddings_batch(texts, input_type)
    return _get_local_embeddings_batch(texts)


# ─── Local Backend ───────────────────────────────────────────────────────────

def _get_local_embedding(text: str) -> np.ndarray:
    model = load_model()
    max_chars = 10000
    if len(text) > max_chars:
        text = text[:max_chars]
    return model.encode(text, convert_to_numpy=True)


def _get_local_embeddings_batch(texts: list[str]) -> list[np.ndarray]:
    model = load_model()
    max_chars = 10000
    truncated = [t[:max_chars] if len(t) > max_chars else t for t in texts]
    embeddings = model.encode(truncated, convert_to_numpy=True, show_progress_bar=False)
    return list(embeddings)


# ─── NVIDIA Backend (llama-nemotron-embed-1b-v2) ────────────────────────────

def _get_nvidia_embedding(text: str, input_type: str = "passage") -> np.ndarray:
    api_key = os.environ.get("NVIDIA_EMBED_API_KEY", "")
    if not api_key:
        return _get_local_embedding(text)

    max_chars = 10000
    if len(text) > max_chars:
        text = text[:max_chars]

    try:
        response = requests.post(
            "https://integrate.api.nvidia.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "nvidia/llama-nemotron-embed-1b-v2",
                "input": text,
                "input_type": input_type,
                "encoding_format": "float",
                "truncate": "END"
            },
            timeout=30
        )
        response.raise_for_status()
        embedding = response.json()["data"][0]["embedding"]
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        print(f"⚠️ NVIDIA Embed error: {e} — falling back to local model")
        return _get_local_embedding(text)


def _get_nvidia_embeddings_batch(texts: list[str], input_type: str = "passage") -> list[np.ndarray]:
    api_key = os.environ.get("NVIDIA_EMBED_API_KEY", "")
    if not api_key:
        return _get_local_embeddings_batch(texts)

    max_chars = 10000
    truncated = [t[:max_chars] if len(t) > max_chars else t for t in texts]

    try:
        response = requests.post(
            "https://integrate.api.nvidia.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "nvidia/llama-nemotron-embed-1b-v2",
                "input": truncated,
                "input_type": input_type,
                "encoding_format": "float",
                "truncate": "END"
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()["data"]
        data.sort(key=lambda x: x["index"])
        return [np.array(item["embedding"], dtype=np.float32) for item in data]
    except Exception as e:
        print(f"⚠️ NVIDIA Batch Embed error: {e} — falling back to local model")
        return _get_local_embeddings_batch(texts)
