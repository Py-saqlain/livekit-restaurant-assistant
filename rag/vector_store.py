"""
Per-restaurant vector store for menu + policy retrieval.

Each restaurant gets its own FAISS index, built once from its menu.json
and policy.txt, so retrieval never accidentally mixes items or policies
between restaurants.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from rag.embeddings import embed_query, embed_texts

DATA_DIR = Path(__file__).parent / "data"


@dataclass
class Chunk:
    text: str
    kind: str  # "menu_item" or "policy"


class RestaurantIndex:
    """Holds a FAISS index + the original chunks for one restaurant."""

    def __init__(self, restaurant_id: str):
        self.restaurant_id = restaurant_id
        self.chunks: list[Chunk] = []
        self.index: faiss.IndexFlatL2 | None = None
        self._build()

    def _build(self) -> None:
        restaurant_dir = DATA_DIR / self.restaurant_id

        # Load menu items - one chunk per item, with category + price + description
        menu_path = restaurant_dir / "menu.json"
        with open(menu_path, encoding="utf-8") as f:
            menu_data = json.load(f)

        for item in menu_data["menu"]:
            text = (
                f"{item['item']} ({item['category']}) - Rs. {item['price']}. "
                f"{item['description']}"
            )
            self.chunks.append(Chunk(text=text, kind="menu_item"))

        # Load policy - split into paragraphs for finer-grained retrieval
        policy_path = restaurant_dir / "policy.txt"
        with open(policy_path, encoding="utf-8") as f:
            policy_text = f.read()

        for para in policy_text.split("\n\n"):
            para = para.strip()
            if para:
                self.chunks.append(Chunk(text=para, kind="policy"))

        # Embed all chunks and build the FAISS index
        texts = [c.text for c in self.chunks]
        vectors = embed_texts(texts)
        vectors_np = np.array(vectors, dtype="float32")

        dimension = vectors_np.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(vectors_np)

    def search(self, query: str, top_k: int = 4) -> list[Chunk]:
        """Return the top_k most relevant chunks for a query."""
        query_vec = np.array([embed_query(query)], dtype="float32")
        distances, indices = self.index.search(query_vec, top_k)
        return [self.chunks[i] for i in indices[0] if i != -1]


# Cache of built indexes, keyed by restaurant_id - built once, reused
_indexes: dict[str, RestaurantIndex] = {}


def get_index(restaurant_id: str) -> RestaurantIndex:
    if restaurant_id not in _indexes:
        _indexes[restaurant_id] = RestaurantIndex(restaurant_id)
    return _indexes[restaurant_id]


def search_restaurant(restaurant_id: str, query: str, top_k: int = 4) -> str:
    """
    Main entry point used by the voice agent's search_menu tool.
    Returns a plain-text block of the most relevant menu/policy info.
    """
    index = get_index(restaurant_id)
    results = index.search(query, top_k=top_k)
    if not results:
        return "No matching information found."
    return "\n".join(f"- {c.text}" for c in results)