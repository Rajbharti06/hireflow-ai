"""
embedder.py — Text to Semantic Embeddings
============================================
Converts text into dense vector representations using sentence-transformers.
Uses 'all-MiniLM-L6-v2' — fast, lightweight, great for similarity tasks.

The model maps text to a 384-dimensional vector space where semantically
similar texts are close together. This powers the matching engine.
"""

import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer


@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load the sentence-transformer model (cached across Streamlit reruns).
    
    Returns:
        SentenceTransformer model instance
    """
    return SentenceTransformer('all-MiniLM-L6-v2')


def get_embedding(text: str) -> np.ndarray:
    """
    Generate a semantic embedding vector for the given text.
    
    Args:
        text: Input text to embed
    
    Returns:
        numpy array of shape (384,) — the embedding vector
    """
    model = load_model()
    
    # Truncate very long texts to avoid memory issues
    # MiniLM has a max sequence length of 256 tokens, but we send more
    # and let the model handle truncation internally
    max_chars = 10000
    if len(text) > max_chars:
        text = text[:max_chars]
    
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding


def get_embeddings_batch(texts: list[str]) -> list[np.ndarray]:
    """
    Generate embeddings for multiple texts in a single batch (faster).
    
    Args:
        texts: List of text strings to embed
    
    Returns:
        List of numpy arrays (embeddings)
    """
    model = load_model()
    
    # Truncate each text
    max_chars = 10000
    truncated = [t[:max_chars] if len(t) > max_chars else t for t in texts]
    
    embeddings = model.encode(truncated, convert_to_numpy=True, show_progress_bar=False)
    return list(embeddings)
