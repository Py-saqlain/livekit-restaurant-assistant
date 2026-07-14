"""
Embedding wrapper for the restaurant RAG pipeline.

Uses FastEmbed (same library used in the RepoMind AI project) - runs
locally, no API key needed, free.
"""

from fastembed import TextEmbedding

_model = None


def get_embedding_model() -> TextEmbedding:
    """Lazily load the embedding model once, reused across all calls."""
    global _model
    if _model is None:
        _model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of text chunks, returns a list of vectors."""
    model = get_embedding_model()
    return [vec.tolist() for vec in model.embed(texts)]


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]