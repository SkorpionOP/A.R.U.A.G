"""
semantic_cache_faiss.py — Vectorized Query-Level Semantic Cache
===============================================================
A two-tier semantic similarity cache bypassing LLM pipelines on cache hits.
Upgraded with FAISS for highly scalable, optimized vector lookups.
"""

import random
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Optional, List, Dict

import numpy as np
import faiss


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures & Helpers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    id: int
    query: str
    result: dict
    timestamp: float
    last_access: float

    def is_expired(self, now: float, ttl: float) -> bool:
        return (now - self.timestamp) > ttl

    def update_access(self, now: float) -> None:
        self.last_access = now


def _token_overlap(a: str, b: str) -> float:
    """Jaccard similarity over unigram token sets (case-insensitive intent guard)."""
    def _tokens(s: str) -> set:
        return set(re.findall(r"\w+", s.lower()))

    ta, tb = _tokens(a), _tokens(b)
    union = ta | tb
    return len(ta & tb) / len(union) if union else 0.0


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    """FAISS Inner Product (IP) behaves as Cosine Similarity when L2 normalized."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return (vector / norm).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main Cache Class
# ─────────────────────────────────────────────────────────────────────────────

class SemanticCache:
    _PRUNE_PROB = 0.01

    def __init__(
        self,
        embed_func: Callable[[str], np.ndarray],
        embedding_dim: int = 384,  # Default for all-MiniLM-L6-v2
        similarity_threshold: float = 0.92,
        intent_overlap_threshold: float = 0.30,
        max_size: int = 5000,      # FAISS easily handles much larger sizes
        ttl_seconds: float = 3600.0,
    ):
        self._embed_func = embed_func
        self.threshold = similarity_threshold
        self.intent_threshold = intent_overlap_threshold
        self.max_size = max_size
        self.ttl = ttl_seconds

        # ── Tier 1: Exact Store ──
        self._exact: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # ── Tier 2: FAISS Semantic Store ──
        # IndexIDMap allows us to attach our own custom integer IDs to vectors
        base_index = faiss.IndexFlatIP(embedding_dim)
        self._index = faiss.IndexIDMap(base_index)
        
        self._entries: Dict[int, CacheEntry] = {}
        self._next_id = 0
        self._emb_memo: Dict[str, np.ndarray] = {}

        # ── Telemetry & Stats ──
        self.hits = self.exact_hits = self.semantic_hits = self.misses = 0
        self.evictions = self.expired_removed = 0
        self._hit_similarities: List[float] = []
        self._lookup_times: List[float] = []

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def lookup(self, query: str) -> Optional[dict]:
        t0 = time.perf_counter()
        result = self._lookup_internal(query)
        
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._lookup_times.append(elapsed_ms)
        if len(self._lookup_times) > 1000:
            self._lookup_times.pop(0)
            
        return result

    def store(self, query: str, result: dict) -> None:
        now = time.time()
        if self._is_duplicate(query, now):
            return

        clean_result = {k: v for k, v in result.items() 
                        if k not in ("cache_hit", "cache_similarity")}

        # Tier 1 Store
        self._store_exact(query, clean_result, now)
        
        # Tier 2 Store
        self._sweep_expired(now)
        self._evict_semantic_if_full()
        self._store_semantic(query, clean_result, now)

    def invalidate_all(self) -> int:
        count = len(self._entries)
        self._exact.clear()
        self._entries.clear()
        self._index.reset()
        self._emb_memo.clear()
        
        self.hits = self.exact_hits = self.semantic_hits = self.misses = 0
        self.evictions = self.expired_removed = 0
        self._hit_similarities.clear()
        self._lookup_times.clear()
        return count

    # ──────────────────────────────────────────────────────────────────────────
    # Internal Pipeline
    # ──────────────────────────────────────────────────────────────────────────

    def _lookup_internal(self, query: str) -> Optional[dict]:
        now = time.time()

        # Tier 1 Exact Lookup
        exact_hit = self._check_exact(query, now)
        if exact_hit:
            return exact_hit

        # Lazy maintenance
        if random.random() < self._PRUNE_PROB:
            self._sweep_expired(now)

        # Tier 2 FAISS Lookup
        semantic_hit = self._check_semantic(query, now)
        if semantic_hit:
            return semantic_hit

        self.misses += 1
        return None

    def _check_exact(self, query: str, now: float) -> Optional[dict]:
        norm = self._normalise(query)
        if norm in self._exact:
            entry = self._exact[norm]
            if not entry.is_expired(now, self.ttl):
                entry.update_access(now)
                self._exact.move_to_end(norm)
                
                self._record_hit(exact=True, sim=1.0)
                return self._annotate(entry.result, 1.0)
            
            del self._exact[norm]
            self.expired_removed += 1
        return None

    def _check_semantic(self, query: str, now: float) -> Optional[dict]:
        if self._index.ntotal == 0:
            return None

        q_emb = self._get_embedding(query)
        
        # Search FAISS for top 5 closest vectors
        # Returns distances (cosine similarities due to IP + L2) and FAISS IDs
        similarities, idxs = self._index.search(q_emb.reshape(1, -1), k=min(5, self._index.ntotal))

        best_entry, best_sim = None, -1.0

        for sim, entry_id in zip(similarities[0], idxs[0]):
            if entry_id == -1 or entry_id not in self._entries:
                continue
                
            entry = self._entries[entry_id]
            if entry.is_expired(now, self.ttl):
                continue
            
            if sim > best_sim and sim >= self.threshold:
                # Lexical intent guard
                if self.intent_threshold > 0.0 and _token_overlap(query, entry.query) < self.intent_threshold:
                    continue
                best_sim = float(sim)
                best_entry = entry

        if best_entry:
            best_entry.update_access(now)
            self._record_hit(exact=False, sim=best_sim)
            return self._annotate(best_entry.result, best_sim)
            
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Internal Storage & Eviction (FAISS logic)
    # ──────────────────────────────────────────────────────────────────────────

    def _store_exact(self, query: str, result: dict, now: float) -> None:
        norm = self._normalise(query)
        # Using -1 for ID since exact cache doesn't need FAISS tracking
        self._exact[norm] = CacheEntry(-1, query, result, now, now)
        self._exact.move_to_end(norm)
        
        if len(self._exact) > self.max_size:
            self._exact.popitem(last=False)
            self.evictions += 1

    def _store_semantic(self, query: str, result: dict, now: float) -> None:
        emb = self._get_embedding(query).reshape(1, -1)
        entry_id = self._next_id
        self._next_id += 1
        
        # Add to FAISS index mapped to our generated entry_id
        ids_array = np.array([entry_id], dtype=np.int64)
        self._index.add_with_ids(emb, ids_array)
        
        # Add to dictionary mapping
        self._entries[entry_id] = CacheEntry(entry_id, query, result, now, now)

    def _evict_semantic_if_full(self) -> None:
        if len(self._entries) >= self.max_size:
            # Sort by least recently used
            sorted_entries = sorted(self._entries.values(), key=lambda x: x.last_access)
            keep_count = int(self.max_size * 0.75)
            entries_to_remove = sorted_entries[:-keep_count]
            
            ids_to_remove = [e.id for e in entries_to_remove]
            self._remove_from_faiss(ids_to_remove)
            self.evictions += len(ids_to_remove)

    def _sweep_expired(self, now: float) -> None:
        expired_ids = [e.id for e in self._entries.values() if e.is_expired(now, self.ttl)]
        if expired_ids:
            self._remove_from_faiss(expired_ids)
            self.expired_removed += len(expired_ids)

    def _remove_from_faiss(self, ids: List[int]) -> None:
        """Helper to cleanly remove entries from both the dict and the FAISS index."""
        if not ids: return
        self._index.remove_ids(np.array(ids, dtype=np.int64))
        for i in ids:
            self._entries.pop(i, None)

    def _is_duplicate(self, query: str, now: float) -> bool:
        norm = self._normalise(query)
        if norm in self._exact and not self._exact[norm].is_expired(now, self.ttl):
            return True

        if self._index.ntotal == 0:
            return False

        q_emb = self._get_embedding(query)
        # We only need to check the closest vector to see if it breaches the threshold
        similarities, idxs = self._index.search(q_emb.reshape(1, -1), k=1)
        
        if len(similarities[0]) > 0:
            best_sim, best_id = similarities[0][0], idxs[0][0]
            if best_id != -1 and best_sim >= self.threshold:
                entry = self._entries.get(best_id)
                if entry and not entry.is_expired(now, self.ttl):
                    return True
        return False

    # ──────────────────────────────────────────────────────────────────────────
    # Utilites
    # ──────────────────────────────────────────────────────────────────────────

    def _get_embedding(self, query: str) -> np.ndarray:
        if query not in self._emb_memo:
            # Generate and immediately L2 normalize for Inner Product Cosine math
            raw_emb = self._embed_func(query)
            self._emb_memo[query] = _l2_normalize(raw_emb)
        return self._emb_memo[query]

    def _record_hit(self, exact: bool, sim: float) -> None:
        self.hits += 1
        if exact: self.exact_hits += 1
        else: self.semantic_hits += 1
        self._hit_similarities.append(sim)

    @staticmethod
    def _normalise(query: str) -> str:
        return query.strip().lower()

    @staticmethod
    def _annotate(result: dict, similarity: float) -> dict:
        return {**result, "cache_hit": True, "cache_similarity": round(float(similarity), 4)}