"""
semantic_cache.py — Query-Level Semantic Cache
================================================
Sits in front of the FDL pipeline. Before running ANY LLM calls, the cache
checks whether a semantically equivalent query was already answered.

If cosine similarity between the new query embedding and any cached query
embedding exceeds the threshold, the stored answer is returned instantly —
bypassing the entire LLM pipeline (System-1, judge, System-2, etc.).

Performance impact:
  Cache HIT  → ~2 ms   (one embed + cosine scan)
  Cache MISS → normal pipeline (unchanged)

Design choices:
  - Uses the same dense embedder as the rest of the system (all-MiniLM-L6-v2)
    so similarity judgements are semantically meaningful, not lexical.
  - Only faithful answers are cached (no point caching low-quality results).
  - TTL-based expiry prevents stale answers from surviving long sessions.
  - LRU-style eviction keeps memory bounded when max_size is reached.
"""

import time
import numpy as np
from typing import Callable, Optional


class SemanticCache:
    """
    Semantic similarity cache for FDL pipeline responses.

    Lookup is O(n) over cached entries (n is typically small, ≤500).
    For larger deployments, replace the linear scan with FAISS.
    """

    def __init__(
        self,
        embed_func: Callable[[str], np.ndarray],
        similarity_threshold: float = 0.92,
        max_size: int = 500,
        ttl_seconds: float = 3600.0,
    ):
        """
        Args:
            embed_func:           Same embed() from Embedder — ensures vectors are
                                  in the same space as the rest of the system.
            similarity_threshold: Cosine similarity required for a cache hit.
                                  0.92 catches near-paraphrases while rejecting
                                  distinct questions on the same broad topic.
            max_size:             Maximum number of cached entries before LRU eviction.
            ttl_seconds:          Entries older than this are considered expired.
        """
        self._embed = embed_func
        self.threshold = similarity_threshold
        self.max_size = max_size
        self.ttl = ttl_seconds

        # Each entry: {query, embedding, result, timestamp}
        self._entries: list[dict] = []

        # Stats
        self.hits = 0
        self.misses = 0

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def lookup(self, query: str) -> Optional[dict]:
        """
        Check whether a semantically equivalent query was already answered.

        Returns the cached result dict (with '_cache_hit' and
        '_cache_similarity' added) if a match is found, otherwise None.
        """
        now = time.time()
        q_emb = self._embed(query)

        best_sim = -1.0
        best_entry = None

        for entry in self._entries:
            if now - entry["timestamp"] > self.ttl:
                continue  # Expired — skip but don't purge inline (done in store)
            sim = self._cosine(q_emb, entry["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_entry is not None and best_sim >= self.threshold:
            self.hits += 1
            cached = dict(best_entry["result"])
            cached["cache_hit"] = True
            cached["cache_similarity"] = round(float(best_sim), 4)
            return cached

        self.misses += 1
        return None

    def store(self, query: str, result: dict) -> None:
        """
        Cache a faithful FDL result.

        Only call this when result["faithful"] is True — there's no value
        in caching answers the system itself flagged as low-quality.
        """
        now = time.time()

        # Purge expired entries
        self._entries = [e for e in self._entries if now - e["timestamp"] <= self.ttl]

        # Evict oldest 25 % if at capacity
        if len(self._entries) >= self.max_size:
            self._entries.sort(key=lambda x: x["timestamp"])
            keep = int(self.max_size * 0.75)
            self._entries = self._entries[-keep:]

        self._entries.append({
            "query":     query,
            "embedding": self._embed(query),
            "result":    {k: v for k, v in result.items() if k not in ("cache_hit", "cache_similarity")},
            "timestamp": now,
        })

    def invalidate_all(self) -> int:
        """Clear the entire cache (e.g., after a new PDF is uploaded)."""
        count = len(self._entries)
        self._entries.clear()
        self.hits = 0
        self.misses = 0
        return count

    # ──────────────────────────────────────────────────────────────────
    # Stats
    # ──────────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "size":           self.size,
            "hits":           self.hits,
            "misses":         self.misses,
            "hit_rate":       round(self.hit_rate, 3),
            "threshold":      self.threshold,
            "ttl_seconds":    self.ttl,
        }

    # ──────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
