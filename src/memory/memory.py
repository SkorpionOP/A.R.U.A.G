import math
import uuid
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict


@dataclass
class Memory:
    """Single memory entry with Ebbinghaus decay, FDL tracking, and confidence score."""
    id:               str
    content:          str
    embedding:        np.ndarray  # 384-d dense vector from all-MiniLM-L6-v2

    created_at:       datetime
    last_accessed:    datetime
    access_count:     int   = 0

    importance_score: float = 0.8   # 0.0 – 1.0 (affects decay strength)
    decay_rate:       float = 0.05  # λ in days⁻¹

    success_count:    int   = 0
    failure_count:    int   = 0

    # 🔥 FIX 2: Explicit confidence score, separate from importance
    confidence_score: float = 0.5   # Starts neutral; updated by outcomes

    category:         str   = "learned_fact"
    suppressed:       bool  = False

    def current_strength(self, now: datetime) -> float:
        """S(t) = importance * exp(-λ * days_since_last_access)"""
        days = max(0.0, (now - self.last_accessed).total_seconds() / 86_400)
        return self.importance_score * math.exp(-self.decay_rate * days)

    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return 0.5 if total == 0 else self.success_count / total

    def memory_worth(self) -> float:
        # 🔥 FIX 2: Worth now incorporates confidence_score
        return 0.6 * self.success_rate() + 0.4 * self.confidence_score


@dataclass
class Interaction:
    """One turn in a conversation."""
    user_input:           str
    expected_output:      str
    agent_output:         str
    outcome:              str          # 'success' | 'failure'
    retrieved_memory_ids: List[str]    = field(default_factory=list)
    timestamp:            datetime     = field(default_factory=datetime.now)


class SimpleRAGMemory:
    """Stores everything; no decay; no failure-driven updates."""

    def __init__(self, name: str = "SimpleRAG", embed_func: Callable[[str], np.ndarray] = None):
        self.name      = name
        self.memories: Dict[str, Memory] = {}
        self.interactions: List[Interaction] = []
        self._embed = embed_func if embed_func else self._default_embed

    def store(self, content: str, importance: float = 0.8,
              category: str = "learned_fact",
              decay_rate: float = 0.0) -> str:
        mem = Memory(
            id=str(uuid.uuid4()),
            content=content,
            embedding=self._embed(content),
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            importance_score=importance,
            decay_rate=decay_rate,
            category=category,
        )
        self.memories[mem.id] = mem
        return mem.id

    def retrieve(self, query: str, k: int = 3,
                 decay_enabled: bool = False,
                 now: Optional[datetime] = None) -> List[Tuple[str, float]]:
        if not self.memories:
            return []
        q_emb  = self._embed(query)
        scores = [
            (mid, self._cosine(q_emb, m.embedding))
            for mid, m in self.memories.items()
            if not m.suppressed
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def store_batch(self, contents: List[str], embeddings: List[np.ndarray],
                    importance: float = 0.8,
                    category: str = "learned_fact",
                    decay_rate: float = 0.0) -> List[str]:
        """
        Bulk-insert chunks with pre-computed embeddings.
        Avoids calling _embed() per chunk — critical for large corpora.
        Returns list of inserted memory IDs.
        """
        now = datetime.now()
        ids = []
        for content, emb in zip(contents, embeddings):
            mem = Memory(
                id=str(uuid.uuid4()),
                content=content,
                embedding=emb,
                created_at=now,
                last_accessed=now,
                importance_score=importance,
                decay_rate=decay_rate,
                category=category,
            )
            self.memories[mem.id] = mem
            ids.append(mem.id)
        return ids

    def log_interaction(self, interaction: Interaction,
                        memory_ids: List[str] = None):
        self.interactions.append(interaction)

    def get_memory_size_kb(self) -> float:
        active = sum(1 for m in self.memories.values() if not m.suppressed)
        return active * 1.5

    def _default_embed(self, text: str) -> np.ndarray:
        import hashlib
        raw = hashlib.md5(text.lower().strip().encode()).digest()
        return np.frombuffer(raw, dtype=np.uint8).astype(np.float32)[:16]

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


class EnhancedMemory(SimpleRAGMemory):
    """
    Extends SimpleRAGMemory with:
      - 🔥 FIX 1: Correction Validation (don't store bad corrections)
      - 🔥 FIX 2: Memory Confidence Score (updated on success/failure)
      - 🔥 FIX 3: Hard cap — max 1 memory slot in retrieval (70/30 → 70/~1)
      - 🔥 FIX 4: Document Arbitration (doc always wins over memory)
      - 🔥 FIX 5: Real decay — uses simulated time, fast decay rates, aggressive pruning
    """

    CATEGORY_DECAY = {
        "user_preference": 0.10,   # Moderate decay
        "failure_note":    1.00,   # 🔥 FIX 5: Very fast decay — stale corrections die in ~1 day
        "learned_fact":    0.20,   # 🔥 FIX 5: Faster than before (~5 days half-life)
    }

    # 🔥 FIX 5: Aggressive pruning thresholds
    STRENGTH_PRUNE_THRESHOLD = 0.40   # Was 0.20 — prune much more aggressively
    WORTH_PRUNE_THRESHOLD    = 0.30   # Was 0.17
    MIN_FAILURES_FOR_WORTH   = 2      # Was 3 — prune bad memories sooner

    # 🔥 FIX 2: Confidence update rates
    CONFIDENCE_BOOST = 0.10   # +0.10 per successful retrieval
    CONFIDENCE_PENALTY = 0.15 # -0.15 per failed retrieval

    # 🔥 FIX 3: Hard memory cap — at most 1 memory note per query
    MAX_MEMORY_SLOTS = 1

    # 🔥 FIX 2: Minimum confidence to be retrieved at all
    MIN_CONFIDENCE_THRESHOLD = 0.35  # Memory must have some proven track record

    # Min cosine similarity between correction and doc context to accept it
    CORRECTION_VALIDATION_THRESHOLD = 0.05  # 🔥 FIX 1

    def __init__(self, name: str = "Enhanced", embed_func: Callable[[str], np.ndarray] = None):
        super().__init__(name, embed_func)
        self.current_time: datetime = datetime.now()
        self._query_outcomes: Dict[str, List[str]] = defaultdict(list)
        self._failure_note_ids: Dict[str, str] = {}

    def advance_time(self, days: float = 1.0):
        self.current_time += timedelta(days=days)

    def store(self, content: str, importance: float = 0.95,
              category: str = "learned_fact",
              decay_rate: float = None,
              confidence: float = 0.5) -> str:
        if decay_rate is None:
            decay_rate = self.CATEGORY_DECAY.get(category, 0.20)

        mem = Memory(
            id=str(uuid.uuid4()),
            content=content,
            embedding=self._embed(content),
            created_at=self.current_time,
            last_accessed=self.current_time,
            importance_score=importance,
            decay_rate=decay_rate,
            confidence_score=confidence,
            category=category,
        )
        self.memories[mem.id] = mem
        return mem.id

    def store_batch(self, contents: List[str], embeddings: List[np.ndarray],
                    importance: float = 0.8,
                    category: str = "learned_fact",
                    decay_rate: float = None,
                    confidence: float = 0.5) -> List[str]:
        """
        Bulk-insert chunks with pre-computed embeddings (EnhancedMemory version).
        Mirrors SimpleRAGMemory.store_batch() with EnhancedMemory-specific fields.
        """
        if decay_rate is None:
            decay_rate = self.CATEGORY_DECAY.get(category, 0.20)
        now = self.current_time
        ids = []
        for content, emb in zip(contents, embeddings):
            mem = Memory(
                id=str(uuid.uuid4()),
                content=content,
                embedding=emb,
                created_at=now,
                last_accessed=now,
                importance_score=importance,
                decay_rate=decay_rate,
                confidence_score=confidence,
                category=category,
            )
            self.memories[mem.id] = mem
            ids.append(mem.id)
        return ids

    def _compute_priority(self, mem: Memory, base_sim: float, now: datetime) -> float:
        """Priority = similarity * strength * recency_boost * confidence_weighted_correctness"""
        strength = mem.current_strength(now)
        days_since = max(0.0, (now - mem.last_accessed).total_seconds() / 86_400)
        recency_boost = 1.0 + 0.3 * math.exp(-0.1 * days_since)
        # 🔥 FIX 2: Use confidence_score in priority calculation
        confidence_factor = mem.confidence_score
        return base_sim * strength * recency_boost * confidence_factor

    def retrieve(self, query: str, k: int = 3,
                 decay_enabled: bool = True,
                 now: Optional[datetime] = None) -> List[Tuple[str, float]]:
        """
        Hybrid retrieval:
          FIX 2: Confidence threshold gates memory retrieval
          FIX 3: Hard cap of MAX_MEMORY_SLOTS memory notes (not document chunks)
          FIX 4: Document arbitration — docs fill first, memory is supplemental
        """
        if not self.memories:
            return []

        t     = now if now is not None else self.current_time
        q_emb = self._embed(query)

        doc_scores = []   # learned_fact — original document chunks
        mem_scores = []   # failure_note, user_preference

        for mid, mem in self.memories.items():
            if mem.suppressed:
                continue

            base_sim = self._cosine(q_emb, mem.embedding)

            if decay_enabled:
                score = self._compute_priority(mem, base_sim, t)
            else:
                score = base_sim

            if mem.category == "learned_fact":
                doc_scores.append((mid, score))
            else:
                # 🔥 FIX 2: Gate memory notes by confidence threshold
                if mem.confidence_score >= self.MIN_CONFIDENCE_THRESHOLD:
                    mem_scores.append((mid, score))

        doc_scores.sort(key=lambda x: x[1], reverse=True)
        mem_scores.sort(key=lambda x: x[1], reverse=True)

        # 🔥 FIX 3: Document gets all k-MAX_MEMORY_SLOTS slots. Memory gets at most MAX_MEMORY_SLOTS.
        doc_slots = k - self.MAX_MEMORY_SLOTS
        selected_docs = doc_scores[:doc_slots]
        selected_mems = mem_scores[:self.MAX_MEMORY_SLOTS]

        # 🔥 FIX 4: Document Arbitration
        # If a memory note contradicts the doc context (low cross-similarity), discard it.
        # Document always wins.
        validated_mems = []
        if selected_docs:
            doc_embeddings = [
                self.memories[mid].embedding
                for mid, _ in selected_docs
                if mid in self.memories
            ]
            for mid, score in selected_mems:
                mem = self.memories.get(mid)
                if mem is None:
                    continue
                avg_doc_sim = float(np.mean([
                    self._cosine(mem.embedding, de) for de in doc_embeddings
                ]))
                if avg_doc_sim >= 0.02:  # Compatible with document context
                    validated_mems.append((mid, score))
                # else: contradicts document — DISCARD (FIX 4)
        else:
            validated_mems = selected_mems

        merged = selected_docs + validated_mems
        merged.sort(key=lambda x: x[1], reverse=True)
        return merged[:k]

    def validate_correction(self, correction: str, doc_context_embeddings: List[np.ndarray]) -> bool:
        """
        🔥 FIX 1: Before storing a correction, check it is actually related
        to the document context (not a hallucination from training data).
        Returns True if valid to store, False if should be rejected.
        """
        if not doc_context_embeddings:
            return True  # No context to compare, allow with caution
        corr_emb = self._embed(correction)
        avg_sim = float(np.mean([
            self._cosine(corr_emb, de) for de in doc_context_embeddings
        ]))
        return avg_sim >= self.CORRECTION_VALIDATION_THRESHOLD

    def log_interaction(self, interaction: Interaction,
                        memory_ids: List[str] = None,
                        doc_context_embeddings: List[np.ndarray] = None):
        """
        Log an interaction with FDL.
        doc_context_embeddings: embeddings of the document chunks used in retrieval,
                                 used to validate corrections before storing.
        """
        super().log_interaction(interaction, memory_ids)

        query_key = interaction.user_input.strip().lower()
        self._query_outcomes[query_key].append(interaction.outcome)

        if memory_ids:
            for mid in memory_ids:
                mem = self.memories.get(mid)
                if mem is None:
                    continue
                if interaction.outcome == "success":
                    mem.success_count += 1
                    mem.last_accessed  = self.current_time
                    mem.importance_score = min(1.0, mem.importance_score + 0.02)
                    # 🔥 FIX 2: Boost confidence on success
                    mem.confidence_score = min(1.0, mem.confidence_score + self.CONFIDENCE_BOOST)
                else:
                    mem.failure_count += 1
                    mem.importance_score = max(0.05, mem.importance_score - 0.05)
                    # 🔥 FIX 2: Penalize confidence on failure
                    mem.confidence_score = max(0.0, mem.confidence_score - self.CONFIDENCE_PENALTY)
                mem.access_count += 1

        if interaction.outcome == "failure":
            # 🔥 FIX 1: Validate correction before storing
            if doc_context_embeddings is not None:
                valid = self.validate_correction(interaction.expected_output, doc_context_embeddings)
                if not valid:
                    return  # Reject hallucinated correction

            concept = self._extract_concept(interaction.user_input)
            note_content = f"{concept}: {interaction.expected_output}"

            if query_key in self._failure_note_ids:
                mid = self._failure_note_ids[query_key]
                if mid in self.memories and not self.memories[mid].suppressed:
                    mem = self.memories[mid]
                    mem.content          = note_content
                    mem.embedding        = self._embed(note_content)
                    mem.failure_count   += 1
                    mem.last_accessed    = self.current_time
                    mem.importance_score = min(1.0, mem.importance_score + 0.05)
                    # Start confidence at 0.5 for updated note — must earn trust
                    mem.confidence_score = 0.5
                    return
            # New correction note starts with low confidence — must be validated by future use
            new_id = self.store(note_content, importance=0.80, category="failure_note", confidence=0.4)
            self._failure_note_ids[query_key] = new_id

    def prune(self) -> int:
        """
        🔥 FIX 5: Real pruning with aggressive thresholds.
        Pruning now works because:
          1. Decay rates are 2-5x higher than before
          2. Pruning thresholds are higher (0.40 vs 0.20)
          3. Confidence score incorporated in worth
        """
        pruned = 0
        now = self.current_time
        for mem in list(self.memories.values()):
            if mem.suppressed:
                continue
            strength = mem.current_strength(now)

            # Prune by strength (Ebbinghaus decay)
            if strength < self.STRENGTH_PRUNE_THRESHOLD:
                mem.suppressed = True
                pruned += 1
                continue

            # Prune by worth (failure history + confidence)
            if (mem.category != "user_preference"
                    and mem.failure_count >= self.MIN_FAILURES_FOR_WORTH
                    and mem.memory_worth() < self.WORTH_PRUNE_THRESHOLD):
                mem.suppressed = True
                pruned += 1
                continue

            # 🔥 FIX 5: Also prune memory notes with near-zero confidence
            if mem.category == "failure_note" and mem.confidence_score < 0.15:
                mem.suppressed = True
                pruned += 1

        return pruned

    @staticmethod
    def _extract_concept(question: str) -> str:
        """Extract a generalizable concept key from a question."""
        q = question.lower().strip().rstrip("?.,!")
        for prefix in ["what is ", "what are ", "who is ", "who are ",
                        "which ", "how many ", "why is ", "why does ",
                        "when did ", "when was ", "can ", "is ",
                        "difference between ", "what type of "]:
            if q.startswith(prefix):
                q = q[len(prefix):]
                break
        for w in ["the ", "a ", "an ", "of the ", "in india", "known as ",
                   "called ", "considered "]:
            q = q.replace(w, " ")
        return " ".join(q.split()).strip()
