"""
embedder.py — Semantic Embedding Engine
========================================
Replaces TF-IDF with a dense sentence-transformer model (all-MiniLM-L6-v2).

Benefits over TF-IDF:
  - Understands synonyms & paraphrases without query rewriting
  - Fixed-dimension output (384-d) — no re-fit required when new PDFs are added
  - Zero vocabulary mismatch crashes
  - Removes the need for a slow LLM Query Rewriting step on every query
  - Batch encoding: thousands of chunks embedded in seconds via GPU/CPU batching
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Dense semantic embedder using all-MiniLM-L6-v2.

    Advantages over TF-IDF:
      ✅ Understands synonyms (e.g., "FR" ~ "Fundamental Rights")
      ✅ Fixed 384-d output — no re-fit needed when corpus changes
      ✅ Eliminates the need for LLM-based query rewriting
      ✅ Batch encoding: embed thousands of chunks in one call
    """

    MODEL_NAME = "all-MiniLM-L6-v2"
    BATCH_SIZE = 256  # Tune down to 64 if RAM is tight

    def __init__(self):
        self.model = SentenceTransformer(self.MODEL_NAME)
        self.is_fitted = True  # Always ready — no corpus-fitting needed

    def fit(self, texts: list[str]):
        """No-op: dense embedders don't need corpus fitting."""
        pass  # Kept for API compatibility with the ingestion flow in app.py

    def embed(self, text: str) -> np.ndarray:
        """Return a 384-dimensional L2-normalised embedding vector (single text)."""
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.astype(np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """
        Batch-encode a list of texts.
        Dramatically faster than calling embed() in a loop:
          15 000 chunks: ~12 min one-by-one  vs  ~15 s batched
        Returns a list of 384-d float32 numpy arrays.
        """
        if not texts:
            return []
        vecs = self.model.encode(
            texts,
            batch_size=self.BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return [v.astype(np.float32) for v in vecs]
