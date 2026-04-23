"""
graph_rag.py — Entity-Based Knowledge Graph for GraphRAG
=========================================================
Standard vector search pulls isolated chunks ranked purely by embedding
similarity to the query. This works well for direct lookups but misses
*relational* context: if Article 32 is the answer but the most relevant
chunk only mentions "writs" without naming the article, dense search may
rank it low.

GraphRAG fixes this by building a knowledge graph over the corpus:

  Nodes  = document chunks
  Edges  = shared constitutional entities (articles, schedules, amendments,
            key legal concepts)
  Weight = number of shared entities between two chunks

During retrieval, after the dense + cross-encoder pipeline returns the
top-2 chunks ("seed nodes"), the graph expands context by walking to
the 1–2 most strongly connected neighbours — pulling in related chunks
that may contain complementary information the seeds lack.

Entity Extraction Strategy:
  - Regex patterns tuned for legal/constitutional documents
  - Frequency cap: entities appearing in > MAX_ENTITY_FREQ_RATIO of all
    chunks are considered "stop entities" (too common to be discriminative)
    and are excluded from edge-building
  - No external NLP library required (no spaCy, no NLTK)

Implementation note:
  We use an inverted index (entity → [chunk_ids]) rather than a full
  networkx adjacency matrix. This avoids O(N²) memory for 15k+ chunks
  while providing identical query-time expansion via set intersection.
"""

import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional


class KnowledgeGraph:
    """
    Entity-based knowledge graph for GraphRAG context expansion.

    Build once at ingestion time, then call expand_context() at query time
    to augment the top-k dense-retrieval results with related chunks.
    """

    # ── Entity extraction patterns (constitutional/legal domain) ──────
    ENTITY_PATTERNS = [
        # Article references: "Article 32", "Article 226A", "Article 13(1)"
        r'\bArticle\s+\d+[A-Za-z]?(?:\(\d+\))?\b',
        # Schedule references: "Tenth Schedule", "Seventh Schedule"
        r'\b(?:First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth|Eleventh|Twelfth)\s+Schedule\b',
        # Parts of the Constitution: "Part III", "Part IV-A"
        r'\bPart\s+(?:I{1,3}V?|VI{0,3}|IX|X{1,3}|XIV|XV|XVI{0,3}|XIX|XX{0,3}I?)\b',
        # Amendment references: "Forty-second Amendment Act", "44th Amendment"
        r'\b(?:\w+(?:ieth|teenth|eth|th|rd|nd|st))\s+(?:Constitutional\s+)?Amendment(?:\s+Act(?:,\s*\d{4})?)?\b',
        # Core constitutional concepts
        r'\b(?:Fundamental\s+Rights?|Directive\s+Principles?|Preamble|'
        r'Emergency(?:\s+Provisions?)?|Writ(?:s)?|Habeas\s+Corpus|'
        r'Judicial\s+Review|Basic\s+Structure|Single\s+Citizenship|'
        r'Money\s+Bill|Constitutional\s+Amendment|Anti-Defection|'
        r'Right\s+to\s+(?:Equality|Freedom|Education|Information))\b',
        # Key institutions
        r'\b(?:Supreme\s+Court|High\s+Court|Parliament|Rajya\s+Sabha|Lok\s+Sabha|'
        r'President\s+of\s+India|Vice[\s-]President|Prime\s+Minister|'
        r'Election\s+Commission|Attorney\s+General|Comptroller)\b',
    ]

    # Entities appearing in more than this fraction of all chunks are
    # too ubiquitous to be useful graph edges (e.g., "Parliament").
    MAX_ENTITY_FREQ_RATIO = 0.08

    def __init__(self):
        # chunk_id → set of extracted entity strings
        self._chunk_entities: Dict[str, Set[str]] = {}
        # entity string → list of chunk_ids containing it
        self._entity_to_chunks: Dict[str, List[str]] = defaultdict(list)
        self._total_chunks: int = 0
        self._built: bool = False

    # ──────────────────────────────────────────────────────────────────
    # Build
    # ──────────────────────────────────────────────────────────────────

    def build(self, chunk_ids: List[str], chunk_contents: List[str]) -> None:
        """
        Build the entity index from chunk IDs and their text.

        Args:
            chunk_ids:      Memory IDs returned by store_batch() — used as node keys.
            chunk_contents: Corresponding raw text of each chunk.
        """
        self._total_chunks = len(chunk_ids)
        raw_entity_to_chunks: Dict[str, List[str]] = defaultdict(list)

        # Pass 1: extract entities per chunk
        for chunk_id, content in zip(chunk_ids, chunk_contents):
            entities = self._extract_entities(content)
            self._chunk_entities[chunk_id] = entities
            for ent in entities:
                raw_entity_to_chunks[ent].append(chunk_id)

        # Pass 2: discard stop-entities (too frequent to be discriminative)
        freq_cap = max(2, int(self._total_chunks * self.MAX_ENTITY_FREQ_RATIO))
        for ent, cids in raw_entity_to_chunks.items():
            if len(cids) <= freq_cap:
                self._entity_to_chunks[ent] = cids

        self._built = True
        print(f"  [GraphRAG] Built: {self._total_chunks} nodes | "
              f"{len(self._entity_to_chunks)} discriminative entities")

    # ──────────────────────────────────────────────────────────────────
    # Query-time expansion
    # ──────────────────────────────────────────────────────────────────

    def expand_context(
        self,
        seed_ids: List[str],
        top_k: int = 2,
        exclude_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Walk the entity graph from seed chunk nodes and return the
        most strongly connected neighbours.

        Score of a candidate chunk C given seeds S:
            score(C) = |entities(C) ∩ ⋃entities(S)| / |⋃entities(S)|

        This is a normalised entity-overlap score in [0, 1].

        Args:
            seed_ids:    Chunk IDs already in the prompt (dense + reranker results).
            top_k:       Max number of additional chunks to return.
            exclude_ids: Extra IDs to skip (e.g., already-failed chunks).

        Returns:
            List of (chunk_id, score) sorted descending, length ≤ top_k.
        """
        if not self._built or not seed_ids:
            return []

        exclude = set(seed_ids) | (exclude_ids or set())

        # Collect the union of all entities in seed chunks
        seed_entities: Set[str] = set()
        for sid in seed_ids:
            seed_entities.update(self._chunk_entities.get(sid, set()))

        if not seed_entities:
            return []

        # Score candidates by entity overlap with seeds
        candidate_scores: Dict[str, int] = defaultdict(int)
        for ent in seed_entities:
            if ent not in self._entity_to_chunks:
                continue
            for cid in self._entity_to_chunks[ent]:
                if cid not in exclude:
                    candidate_scores[cid] += 1

        if not candidate_scores:
            return []

        # Normalise
        n_seed_ents = len(seed_entities)
        scored = [
            (cid, count / n_seed_ents)
            for cid, count in candidate_scores.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ──────────────────────────────────────────────────────────────────
    # Stats
    # ──────────────────────────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        return len(self._chunk_entities)

    @property
    def edge_count(self) -> int:
        """Approximate: sum of (entity frequency choose 2) for all non-stop entities."""
        return sum(
            len(cids) * (len(cids) - 1) // 2
            for cids in self._entity_to_chunks.values()
        )

    def stats(self) -> dict:
        return {
            "nodes":               self.node_count,
            "approx_edges":        self.edge_count,
            "discriminative_ents": len(self._entity_to_chunks),
            "built":               self._built,
        }

    # ──────────────────────────────────────────────────────────────────
    # Entity extraction
    # ──────────────────────────────────────────────────────────────────

    def _extract_entities(self, text: str) -> Set[str]:
        """Extract and normalise constitutional entities from a chunk."""
        entities: Set[str] = set()
        for pattern in self.ENTITY_PATTERNS:
            for match in re.findall(pattern, text, re.IGNORECASE):
                # Normalise: strip extra whitespace, lowercase for deduplication
                entities.add(re.sub(r'\s+', ' ', match).strip().lower())
        return entities
