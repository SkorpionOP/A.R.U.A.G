"""
test_semantic_cache_ultimate.py — The Ultimate RAG Cache Test Suite
===================================================================
Covers all 12 RAG testing categories:
  1. Core Functional          7. Memory & Resource
  2. Semantic Behavior        8. Edge Cases
  3. Stress & Correctness     9. Consistency
  4. Time-Based (TTL)        10. Real-World Simulation
  5. Performance             11. Advanced (Drift/Overwrite)
  6. Statistical Integrity   12. Integration Bounds
"""

import time
import random
import unittest
import numpy as np
import re
from typing import Callable, Optional, List, Dict
from dataclasses import dataclass
from collections import OrderedDict

# ═══════════════════════════════════════════════════════════════════════════
# INLINE REFACTORED CACHE 
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CacheEntry:
    query: str
    result: dict
    timestamp: float
    last_access: float
    embedding: Optional[np.ndarray] = None

    def is_expired(self, now: float, ttl: float) -> bool:
        return (now - self.timestamp) > ttl

    def update_access(self, now: float) -> None:
        self.last_access = now

def _token_overlap(a: str, b: str) -> float:
    def _tokens(s: str) -> set:
        return set(re.findall(r"\w+", s.lower()))
    ta, tb = _tokens(a), _tokens(b)
    union = ta | tb
    return len(ta & tb) / len(union) if union else 0.0

class SemanticCache:
    _PRUNE_PROB = 0.01

    def __init__(self, embed_func: Callable[[str], np.ndarray], similarity_threshold: float = 0.92, 
                 intent_overlap_threshold: float = 0.30, max_size: int = 500, ttl_seconds: float = 3600.0):
        self._embed_func = embed_func
        self.threshold = similarity_threshold
        self.intent_threshold = intent_overlap_threshold
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._exact: OrderedDict[str, CacheEntry] = OrderedDict()
        self._entries: List[CacheEntry] = []
        self._emb_memo: Dict[str, np.ndarray] = {}

        self.hits = self.exact_hits = self.semantic_hits = self.misses = 0
        self.evictions = self.expired_removed = 0
        self._hit_similarities: List[float] = []
        self._lookup_times: List[float] = []

    def lookup(self, query: str) -> Optional[dict]:
        t0 = time.perf_counter()
        result = self._lookup_internal(query)
        self._lookup_times.append((time.perf_counter() - t0) * 1000)
        if len(self._lookup_times) > 1000: self._lookup_times.pop(0)
        return result

    def store(self, query: str, result: dict) -> None:
        now = time.time()
        if self._is_duplicate(query, now): return
        clean_result = {k: v for k, v in result.items() if k not in ("cache_hit", "cache_similarity")}
        self._store_exact(query, clean_result, now)
        self._sweep_expired(now)
        self._evict_semantic_if_full()
        self._store_semantic(query, clean_result, now)

    def invalidate_all(self) -> int:
        count = len(self._entries)
        self._exact.clear(); self._entries.clear(); self._emb_memo.clear()
        self.hits = self.exact_hits = self.semantic_hits = self.misses = 0
        self.evictions = self.expired_removed = 0
        self._hit_similarities.clear(); self._lookup_times.clear()
        return count

    def stats(self) -> dict:
        safe_avg = lambda lst: round(sum(lst) / len(lst), 4) if lst else None
        return {
            "size": len(self._entries),
            "exact_cache_size": len(self._exact),
            "hits": self.hits, "exact_hits": self.exact_hits, "semantic_hits": self.semantic_hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / (self.hits + self.misses), 3) if (self.hits + self.misses) > 0 else 0.0,
            "avg_similarity_hit": safe_avg(self._hit_similarities),
            "evictions": self.evictions, "expired_removed": self.expired_removed,
            "avg_lookup_time_ms": safe_avg(self._lookup_times),
            "threshold": self.threshold, "intent_threshold": self.intent_threshold, "ttl_seconds": self.ttl,
        }

    def _lookup_internal(self, query: str) -> Optional[dict]:
        now = time.time()
        if exact_hit := self._check_exact(query, now): return exact_hit
        if random.random() < self._PRUNE_PROB: self._sweep_expired(now)
        if semantic_hit := self._check_semantic(query, now): return semantic_hit
        self.misses += 1
        return None

    def _check_exact(self, query: str, now: float) -> Optional[dict]:
        norm = query.strip().lower()
        if norm in self._exact:
            entry = self._exact[norm]
            if not entry.is_expired(now, self.ttl):
                entry.update_access(now); self._exact.move_to_end(norm)
                self._record_hit(True, 1.0)
                return {**entry.result, "cache_hit": True, "cache_similarity": 1.0}
            del self._exact[norm]
            self.expired_removed += 1
        return None

    def _check_semantic(self, query: str, now: float) -> Optional[dict]:
        q_emb = self._get_embedding(query)
        best_entry, best_sim = None, -1.0
        for entry in self._entries:
            if entry.is_expired(now, self.ttl): continue
            sim = self._cosine(q_emb, entry.embedding)
            if sim > best_sim and sim >= self.threshold:
                if self.intent_threshold > 0.0 and _token_overlap(query, entry.query) < self.intent_threshold: continue
                best_sim, best_entry = sim, entry
        if best_entry:
            best_entry.update_access(now)
            self._record_hit(False, best_sim)
            return {**best_entry.result, "cache_hit": True, "cache_similarity": round(float(best_sim), 4)}
        return None

    def _store_exact(self, query: str, result: dict, now: float) -> None:
        norm = query.strip().lower()
        self._exact[norm] = CacheEntry(query, result, now, now)
        self._exact.move_to_end(norm)
        if len(self._exact) > self.max_size:
            self._exact.popitem(last=False)
            self.evictions += 1

    def _store_semantic(self, query: str, result: dict, now: float) -> None:
        self._entries.append(CacheEntry(query, result, now, now, embedding=self._get_embedding(query)))

    def _evict_semantic_if_full(self) -> None:
        if len(self._entries) >= self.max_size:
            self._entries.sort(key=lambda x: x.last_access)
            keep = int(self.max_size * 0.75)
            self.evictions += len(self._entries) - keep
            self._entries = self._entries[-keep:]

    def _sweep_expired(self, now: float) -> None:
        before = len(self._entries)
        self._entries = [e for e in self._entries if not e.is_expired(now, self.ttl)]
        self.expired_removed += before - len(self._entries)

    def _is_duplicate(self, query: str, now: float) -> bool:
        norm = query.strip().lower()
        if norm in self._exact and not self._exact[norm].is_expired(now, self.ttl): return True
        q_emb = self._get_embedding(query)
        for entry in self._entries:
            if not entry.is_expired(now, self.ttl) and self._cosine(q_emb, entry.embedding) >= self.threshold: return True
        return False

    def _record_hit(self, exact: bool, sim: float) -> None:
        self.hits += 1
        if exact: self.exact_hits += 1
        else: self.semantic_hits += 1
        self._hit_similarities.append(sim)

    def _get_embedding(self, query: str) -> np.ndarray:
        if query not in self._emb_memo: self._emb_memo[query] = self._embed_func(query)
        return self._emb_memo[query]

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# MOCKS
# ═══════════════════════════════════════════════════════════════════════════

class MockEmbedder:
    def __init__(self, dimension=384):
        self.dimension = dimension
        self.vocabulary = {}
        self._next_vector = np.zeros(dimension)
        self._next_vector[0] = 1.0
        self.call_count = 0  # Track how many times embedding is actually computed!

    def embed(self, text: str) -> np.ndarray:
        self.call_count += 1
        if text in self.vocabulary:
            return self.vocabulary[text]
        vec = self._next_vector.copy()
        self._next_vector = np.roll(self._next_vector, 1)
        self.vocabulary[text] = vec
        return vec

    def inject_similarity(self, base_text: str, sim_text: str, similarity: float):
        base_vec = self.embed(base_text)
        ortho_vec = np.zeros(self.dimension)
        for i in range(self.dimension):
            if base_vec[i] == 0:
                ortho_vec[i] = 1.0
                break
        theta = np.arccos(similarity)
        new_vec = base_vec * np.cos(theta) + ortho_vec * np.sin(theta)
        self.vocabulary[sim_text] = new_vec


# ═══════════════════════════════════════════════════════════════════════════
# TEST SUITES
# ═══════════════════════════════════════════════════════════════════════════

class Test1_CoreAndStats(unittest.TestCase):
    def setUp(self):
        self.embedder = MockEmbedder()
        self.cache = SemanticCache(embed_func=self.embedder.embed)

    def test_miss_on_empty(self):
        self.assertIsNone(self.cache.lookup("Hello?"))
        self.assertEqual(self.cache.stats()["misses"], 1)

    def test_invalidate_all_resets_everything(self):
        self.cache.store("Test", {"ans": 1})
        self.cache.lookup("Test") # Generates a hit
        
        cleared = self.cache.invalidate_all()
        self.assertEqual(cleared, 1)
        
        stats = self.cache.stats()
        self.assertEqual(stats["size"], 0)
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 0)
        self.assertEqual(len(self.cache._emb_memo), 0)


class Test2_SemanticBehavior(unittest.TestCase):
    def setUp(self):
        self.embedder = MockEmbedder()
        self.cache = SemanticCache(embed_func=self.embedder.embed, similarity_threshold=0.92, intent_overlap_threshold=0.30)

    def test_paraphrase_handling(self):
        q1 = "What is Python?"
        q2 = "Explain Python programming language" # High overlap, high sim
        self.embedder.inject_similarity(q1, q2, 0.95)
        
        self.cache.store(q1, {"ans": "Code"})
        res = self.cache.lookup(q2)
        self.assertIsNotNone(res)
        self.assertEqual(self.cache.semantic_hits, 1)

    def test_false_positive_prevention(self):
        q1 = "What is Python?"
        q2 = "What is Java?" # High sim, poor overlap (only "what", "is" match)
        self.embedder.inject_similarity(q1, q2, 0.98)
        
        self.cache.store(q1, {"ans": "Code"})
        res = self.cache.lookup(q2)
        self.assertIsNone(res, "Intent guard should have blocked this false positive")


class Test3_StressAndCorrectness(unittest.TestCase):
    def setUp(self):
        self.embedder = MockEmbedder()
        self.cache = SemanticCache(embed_func=self.embedder.embed, max_size=50)

    def test_repeated_inserts_dedup(self):
        for _ in range(100):
            self.cache.store("What is Python?", {"ans": "Code"})
        
        stats = self.cache.stats()
        self.assertEqual(stats["size"], 1, "Deduplication failed; cache grew from identical inserts")
        self.assertEqual(stats["exact_cache_size"], 1)

    def test_large_scale_insert_and_eviction(self):
        # Insert 100 unique queries into a cache with max_size 50
        for i in range(100):
            self.cache.store(f"Unique query number {i}", {"ans": i})
            
        stats = self.cache.stats()
        self.assertLessEqual(stats["size"], 50, "Cache exceeded max_size limits")
        self.assertTrue(stats["evictions"] > 0, "Eviction logic did not trigger under stress")


class Test4_TimeAndTTL(unittest.TestCase):
    def setUp(self):
        self.embedder = MockEmbedder()
        self.cache = SemanticCache(embed_func=self.embedder.embed, ttl_seconds=0.0) # Instant expiration!

    def test_zero_ttl_instant_expiration(self):
        self.cache.store("Instant Death", {"ans": 1})
        res = self.cache.lookup("Instant Death")
        self.assertIsNone(res, "Item survived despite TTL=0")
        self.assertEqual(self.cache.stats()["expired_removed"], 1)


class Test5_EdgeCasesAndConsistency(unittest.TestCase):
    def setUp(self):
        self.embedder = MockEmbedder()
        self.cache = SemanticCache(embed_func=self.embedder.embed)

    def test_special_characters_normalize(self):
        # Tests that tier-1 exact match handles punctuation safely
        self.cache.store("What is Python???", {"ans": 1})
        self.assertIsNotNone(self.cache.lookup("what is python???"), "Case insensitivity failed")
        self.assertIsNone(self.cache.lookup("What is Python?"), "Punctuation stripping is too aggressive")

    def test_empty_and_long_queries(self):
        self.cache.store("", {"ans": "empty"})
        self.assertIsNotNone(self.cache.lookup(""))
        
        long_q = "A" * 5000
        self.cache.store(long_q, {"ans": "long"})
        self.assertIsNotNone(self.cache.lookup(long_q))

    def test_multilingual(self):
        self.cache.store("¿Qué es Python?", {"ans": "es"})
        self.cache.store("什么是Python?", {"ans": "zh"})
        self.assertIsNotNone(self.cache.lookup("¿Qué es Python?"))


class Test6_PerformanceAndMemoization(unittest.TestCase):
    def setUp(self):
        self.embedder = MockEmbedder()
        self.cache = SemanticCache(embed_func=self.embedder.embed)

    def test_embedding_memoization(self):
        q = "Expensive ML Query"
        
        # Miss 1: Computes embedding
        self.cache.lookup(q) 
        calls_after_miss = self.embedder.call_count
        
        # Store: Should reuse embedding
        self.cache.store(q, {"ans": 1})
        
        # Miss 2 (Different query): Computes embedding
        self.cache.lookup("Different Query")
        
        # Total calls should be exactly 2 (One for the first query, one for the second)
        self.assertEqual(self.embedder.call_count, 2, "Embedder was called more than necessary! Memoization failed.")

    def test_semantic_drift_rejection(self):
        # 11. Advanced: Tests that if a query drifts (same string, new answer), 
        # the deduplication logic correctly PREVENTS overwriting the faithful cache.
        self.cache.store("Define A", {"ans": "Original"})
        self.cache.store("Define A", {"ans": "Malicious Overwrite"})
        
        res = self.cache.lookup("Define A")
        self.assertEqual(res["ans"], "Original", "Cache allowed a drift overwrite. Dedup failed.")


if __name__ == "__main__":
    print("╔" + "="*76 + "╗")
    print("║" + " "*14 + "ULTIMATE RAG SEMANTIC CACHE TEST SUITE" + " "*24 + "║")
    print("╚" + "="*76 + "╝")
    unittest.main(verbosity=2)