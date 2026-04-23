# Memory System: Concrete Schema Design (Phase 1 Blueprint)

## Overview

This document provides a **working blueprint** for the memory storage layer that bridges the theoretical framework in the paper with implementable code. Target: **SQLite for Prototype A** (simplicity), **PostgreSQL + pgvector for Prototype B** (scalability).

---

## Part 1: SQLite Schema (Prototype A)

### Core Tables

```sql
-- Memories: the central record
CREATE TABLE memories (
  id TEXT PRIMARY KEY,  -- UUID
  
  -- Content & embedding
  content TEXT NOT NULL,
  embedding BLOB NOT NULL,  -- numpy.float32 array, ~1.5KB for 384-dim
  
  -- Temporal tracking
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  access_count INTEGER DEFAULT 0,
  
  -- Ebbinghaus decay model
  importance_score REAL NOT NULL,  -- 0–1, set at ingestion
  decay_rate REAL NOT NULL,  -- λ, category-dependent (see below)
  
  -- Success tracking (for FDL)
  success_count INTEGER DEFAULT 0,
  failure_count INTEGER DEFAULT 0,
  success_rate REAL DEFAULT 0.5,  -- success_count / (success_count + failure_count)
  
  -- Memory worth (derived, cached)
  memory_worth REAL,  -- MW < 0.17 → suppress (see calculation below)
  
  -- Storage & pruning
  storage_layer TEXT CHECK(storage_layer IN ('LML', 'SML')),  -- Long/Short-term
  suppressed BOOLEAN DEFAULT 0,
  pruned_at TIMESTAMP,
  
  -- Metadata
  task_type TEXT,  -- 'customer_service', 'coding', 'qa', etc.
  category TEXT,  -- 'preference', 'learned_fact', 'failure', 'procedure'
  
  INDEX idx_created_at (created_at),
  INDEX idx_last_accessed (last_accessed),
  INDEX idx_task_type (task_type),
  INDEX idx_suppressed (suppressed)
);

-- Interactions: log every turn
CREATE TABLE interactions (
  id TEXT PRIMARY KEY,
  memory_id TEXT REFERENCES memories(id),
  
  user_input TEXT NOT NULL,
  agent_output TEXT NOT NULL,
  outcome TEXT NOT NULL,  -- 'success', 'failure', 'partial', 'unknown'
  
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  task_type TEXT,
  
  -- Which memories were used?
  retrieved_memory_ids TEXT,  -- JSON array of IDs
  
  -- Did the memory help or hurt?
  memory_impact TEXT,  -- 'positive', 'negative', 'neutral'
  
  INDEX idx_timestamp (timestamp),
  INDEX idx_memory_id (memory_id)
);

-- Causal links: which memory caused which outcome?
CREATE TABLE causal_links (
  source_memory_id TEXT REFERENCES memories(id),
  target_memory_id TEXT REFERENCES memories(id),
  interaction_id TEXT REFERENCES interactions(id),
  
  -- Strength of link: how much did source contribute to target?
  contribution_score REAL,  -- 0–1
  relationship_type TEXT,  -- 'prerequisite', 'contradicts', 'refines', etc.
  
  PRIMARY KEY (source_memory_id, target_memory_id, interaction_id)
);

-- Consolidation log: what happened during sleep-time?
CREATE TABLE consolidation_log (
  id TEXT PRIMARY KEY,
  run_timestamp TIMESTAMP,
  
  -- What was done?
  memories_deduplicated INTEGER,
  fact_clusters_formed INTEGER,
  wiki_pages_generated INTEGER,
  
  -- Before/after memory stats
  memory_size_before INTEGER,  -- bytes
  memory_size_after INTEGER,
  
  -- Quality metrics
  redundancy_reduction REAL,  -- % of deduplicated content
  
  INDEX idx_run_timestamp (run_timestamp)
);
```

### Decay Rate Lookup Table

```sql
CREATE TABLE decay_rates (
  category TEXT PRIMARY KEY,
  decay_rate REAL NOT NULL,  -- λ in: Strength = Importance × e^(-λt)
  
  -- Typical values (tuned per domain):
  -- 'preference': 0.0001 (slow decay, important)
  -- 'failure': 0.00005 (very slow, critical learning signal)
  -- 'learned_fact': 0.0002 (medium decay)
  -- 'syntax_question': 0.001 (fast decay, low value)
  
  reason TEXT
);

INSERT INTO decay_rates VALUES
  ('user_preference', 0.0001, 'Preferences are stable, should persist'),
  ('learned_failure', 0.00005, 'Learning from failure is critical'),
  ('domain_fact', 0.0002, 'Facts decay gradually, rarely used facts fade'),
  ('interaction_detail', 0.001, 'Specific conversational details are ephemeral'),
  ('syntax_question', 0.002, 'Low-value questions fade quickly');
```

---

## Part 2: Calculated Fields & Queries

### Current Strength (Ebbinghaus Decay)

```sql
-- SQLite doesn't have computed columns easily, so use a VIEW:
CREATE VIEW memory_strength AS
SELECT 
  id,
  content,
  importance_score,
  decay_rate,
  (julianday('now') - julianday(last_accessed)) as days_since_access,
  importance_score * EXP(-decay_rate * (julianday('now') - julianday(last_accessed))) as current_strength
FROM memories
WHERE suppressed = 0 AND pruned_at IS NULL;

-- Check: memories still above suppression threshold
SELECT id, content, current_strength
FROM memory_strength
WHERE current_strength > 0.17
ORDER BY current_strength DESC;
```

### Memory Worth (MW) Calculation

Memory worth = how often does this memory co-occur with success vs. failure?

```sql
-- For each memory, calculate MW over last 30 days:
WITH recent_interactions AS (
  SELECT * FROM interactions
  WHERE timestamp > datetime('now', '-30 days')
    AND outcome IN ('success', 'partial')
),
memory_outcomes AS (
  SELECT 
    m.id,
    m.content,
    COUNT(CASE WHEN i.outcome IN ('success', 'partial') THEN 1 END) as success_interactions,
    COUNT(CASE WHEN i.outcome IN ('failure') THEN 1 END) as failure_interactions,
    COUNT(*) as total_interactions
  FROM memories m
  LEFT JOIN interactions i ON m.id IN (
    -- Parse JSON array of retrieved_memory_ids
    SELECT value FROM json_each(i.retrieved_memory_ids)
    WHERE value = m.id
  )
  WHERE i.timestamp > datetime('now', '-30 days')
  GROUP BY m.id
)
SELECT 
  id,
  content,
  CAST(success_interactions AS REAL) / NULLIF(total_interactions, 0) as success_rate,
  CASE 
    WHEN success_rate < 0.17 THEN 'SUPPRESS'
    WHEN success_rate < 0.30 THEN 'MONITOR'
    ELSE 'KEEP'
  END as action
FROM memory_outcomes
ORDER BY success_rate ASC;
```

### Retrieval (Hybrid BM25 + Semantic)

For SQLite, use **FTS5 for BM25** and **FAISS for semantic search**, then merge results.

```sql
-- Full-text search (BM25)
CREATE VIRTUAL TABLE memory_fts USING fts5(
  content,
  task_type,
  category,
  content=memories,
  content_rowid=id
);

-- Populate FTS index
INSERT INTO memory_fts(rowid, content, task_type, category)
SELECT id, content, task_type, category FROM memories
WHERE suppressed = 0 AND pruned_at IS NULL;

-- Query: BM25 search for "API deprecation"
SELECT id, content, bm25(memory_fts) as bm25_score
FROM memory_fts
WHERE content MATCH 'API deprecation'
ORDER BY bm25_score DESC
LIMIT 10;
```

For semantic search (Python + FAISS):

```python
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Build FAISS index from active memories
active_memories = db.execute("""
  SELECT id, content FROM memories
  WHERE suppressed = 0 AND pruned_at IS NULL
""").fetchall()

embeddings = np.array([m.embedding for m in active_memories], dtype=np.float32)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Query
query = "What was the API deprecation we discussed?"
query_embedding = model.encode([query])[0].astype(np.float32)

distances, indices = index.search(np.array([query_embedding]), k=10)

# Results
for idx, dist in zip(indices[0], distances[0]):
    memory = active_memories[idx]
    print(f"{memory['content']}: similarity={1/(1+dist):.2f}")
```

**Hybrid Ranking** (merge BM25 + semantic):

```python
def hybrid_retrieval(query_text, bm25_k=10, semantic_k=10, alpha=0.5):
    # Get BM25 results
    bm25_results = db.execute("""
        SELECT id, content, bm25(memory_fts) as score
        FROM memory_fts
        WHERE content MATCH ?
        ORDER BY score DESC
        LIMIT ?
    """, (query_text, bm25_k)).fetchall()
    
    # Get semantic results
    query_emb = model.encode([query_text])[0].astype(np.float32)
    semantic_results = semantic_search(query_emb, k=semantic_k)
    
    # Normalize scores to [0, 1] and merge
    merged = {}
    for i, (mem_id, score) in enumerate(bm25_results):
        merged[mem_id] = alpha * (1 - i / bm25_k)  # BM25 score (higher rank = higher score)
    
    for i, (mem_id, dist) in enumerate(semantic_results):
        semantic_score = 1 / (1 + dist)
        merged[mem_id] = merged.get(mem_id, 0) + (1 - alpha) * semantic_score
    
    # Return top-20 by merged score
    return sorted(merged.items(), key=lambda x: x[1], reverse=True)[:20]
```

---

## Part 3: Failure-Driven Learning (FDL) Workflow

### Failure Detection & Logging

```python
def log_failure(query, response, expected_output, reason):
    """
    Detect and log a failure.
    
    Args:
        query: user input
        response: agent's output
        expected_output: ground truth (if known)
        reason: human-readable reason (e.g., "test failure", "format error")
    """
    failure_id = str(uuid.uuid4())
    
    db.execute("""
        INSERT INTO memories (id, content, embedding, importance_score, decay_rate, category, task_type)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        failure_id,
        json.dumps({
            "type": "failure",
            "query": query,
            "response": response,
            "expected": expected_output,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }),
        model.encode([f"{query} {reason}"])[0],  # embedding
        0.9,  # high importance
        0.00005,  # very slow decay
        "learned_failure",
        "coding"  # or "customer_service", etc.
    ))
    
    return failure_id

def explain_failure(failure_id):
    """
    Use an LLM to explain why the failure occurred.
    """
    failure = db.execute("SELECT content FROM memories WHERE id = ?", (failure_id,)).fetchone()
    failure_data = json.loads(failure['content'])
    
    explanation_prompt = f"""
    The agent made an error:
    - Query: {failure_data['query']}
    - Response: {failure_data['response']}
    - Expected: {failure_data['expected']}
    - Reason: {failure_data['reason']}
    
    Explain concisely why this error occurred and how to avoid it.
    """
    
    explanation = llm(explanation_prompt)
    
    # Store explanation as a linked memory
    explanation_id = str(uuid.uuid4())
    db.execute("""
        INSERT INTO memories (id, content, embedding, importance_score, decay_rate, category, task_type)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        explanation_id,
        explanation,
        model.encode([explanation])[0],
        0.85,  # high importance
        0.00005,
        "learned_lesson",
        "coding"
    ))
    
    # Link failure → explanation
    db.execute("""
        INSERT INTO causal_links (source_memory_id, target_memory_id, relationship_type, contribution_score)
        VALUES (?, ?, 'explains', 0.9)
    """, (failure_id, explanation_id))
    
    return explanation_id, explanation
```

### Failure Avoidance (Negative RAG)

```python
def retrieve_past_failures(query, k=3):
    """
    Retrieve past failures similar to the current query.
    """
    # Search for memories with category='learned_failure'
    query_emb = model.encode([query])[0].astype(np.float32)
    
    failures = db.execute("""
        SELECT id, content, success_rate FROM memories
        WHERE category = 'learned_failure' AND suppressed = 0
        ORDER BY importance_score DESC
        LIMIT ?
    """, (k * 5,)).fetchall()  # get more, then re-rank
    
    # Re-rank by semantic similarity
    failure_embeddings = [json.loads(f['content']).get('embedding', []) for f in failures]
    similarities = [cosine_similarity([query_emb], [emb])[0][0] for emb in failure_embeddings]
    
    ranked = sorted(zip(failures, similarities), key=lambda x: x[1], reverse=True)
    
    return [(f[0]['id'], f[0]['content'], f[1]) for f in ranked[:k]]

def generate_with_negative_rag(query, retrieved_failures):
    """
    Generate a response, but prompt the LLM to avoid past failures.
    """
    failure_context = "\n\n".join([
        f"Past failure: {json.loads(f)[reason]}\nAvoid: {json.loads(f)['how_to_avoid']}"
        for f in retrieved_failures
    ])
    
    prompt = f"""
    User query: {query}
    
    Past failures to avoid:
    {failure_context}
    
    Generate a response that:
    1. Answers the query correctly
    2. Explicitly avoids the pitfalls above
    
    Response:
    """
    
    response = llm(prompt)
    
    # Validate: did the model follow the negative RAG advice?
    validation_prompt = f"""
    Did the response below successfully avoid the past failures listed?
    
    Failures to avoid: {failure_context}
    Response: {response}
    
    Verdict: avoid_success or avoid_failure
    Confidence: 0–1
    """
    
    validation = llm(validation_prompt)
    confidence = extract_confidence_from_validation(validation)
    
    if confidence < 0.6:
        # Fallback: regenerate without negative RAG
        response = llm(f"Answer this query without past context: {query}")
    
    return response
```

---

## Part 4: Nightly Consolidation (Sleep-Time Computation)

```python
def consolidate_memories():
    """
    Nightly batch job to:
    1. Extract facts from interaction logs
    2. Deduplicate via clustering
    3. Generate wiki summaries
    4. Re-embed and re-index
    """
    
    # Get recent interactions
    recent_interactions = db.execute("""
        SELECT * FROM interactions
        WHERE timestamp > datetime('now', '-7 days')
        ORDER BY timestamp DESC
    """).fetchall()
    
    # Step 1: Extract facts via LLM
    facts = []
    for interaction in recent_interactions:
        fact_prompt = f"""
        From this interaction, extract the key facts:
        User: {interaction['user_input']}
        Agent: {interaction['agent_output']}
        Outcome: {interaction['outcome']}
        
        Format: JSON array of {{subject, predicate, object, confidence}}
        """
        
        extracted = json.loads(llm(fact_prompt))
        facts.extend(extracted)
    
    # Step 2: Deduplicate by semantic similarity
    # Cluster facts with cosine_similarity > 0.85
    fact_embeddings = [model.encode([f"{f['subject']} {f['predicate']} {f['object']}"])[0] for f in facts]
    
    clusters = cluster_by_similarity(fact_embeddings, threshold=0.85)
    
    dedup_facts = []
    for cluster in clusters:
        representative = facts[cluster[0]]  # pick first fact in cluster
        representative['merged_count'] = len(cluster)
        dedup_facts.append(representative)
    
    # Step 3: Generate wiki-style summaries
    for topic in group_facts_by_subject(dedup_facts):
        summary_prompt = f"""
        Summarize these facts into a concise wiki page:
        {json.dumps(topic['facts'])}
        
        Format: Markdown, 2–3 paragraphs, bullet list of key points
        """
        
        wiki_content = llm(summary_prompt)
        
        # Store as a new memory
        wiki_id = str(uuid.uuid4())
        db.execute("""
            INSERT INTO memories (id, content, embedding, importance_score, decay_rate, category, task_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            wiki_id,
            wiki_content,
            model.encode([wiki_content])[0],
            0.8,  # consolidated facts are important
            0.0001,  # slow decay
            "consolidated_fact",
            "general"
        ))
    
    # Step 4: Log consolidation
    db.execute("""
        INSERT INTO consolidation_log (id, run_timestamp, memories_deduplicated, fact_clusters_formed)
        VALUES (?, datetime('now'), ?, ?)
    """, (str(uuid.uuid4()), len(facts), len(dedup_facts)))
    
    print(f"✓ Consolidation complete: {len(facts)} facts → {len(dedup_facts)} clusters")

# Schedule nightly
import schedule
schedule.every().day.at("02:00").do(consolidate_memories)
```

---

## Part 5: Pruning & Decay

```python
def prune_memories():
    """
    Remove memories that have decayed below threshold.
    Runs hourly or after significant memory growth.
    """
    
    # Candidates for pruning: decayed + low success rate
    to_prune = db.execute("""
        SELECT id, content, current_strength, memory_worth, category
        FROM memory_strength
        WHERE (current_strength < 0.1)
           OR (memory_worth < 0.17 AND category = 'learned_failure')
           OR (success_rate < 0.2 AND access_count < 5)
        ORDER BY current_strength ASC
    """).fetchall()
    
    pruned_count = 0
    for memory in to_prune:
        db.execute("""
            UPDATE memories
            SET suppressed = 1, pruned_at = datetime('now')
            WHERE id = ?
        """, (memory['id'],))
        
        pruned_count += 1
        
        # Log pruning reason
        reason = "decayed"
        if memory['memory_worth'] < 0.17:
            reason = "low_worth"
        elif memory['success_rate'] < 0.2:
            reason = "ineffective"
        
        print(f"  Pruned: {memory['content'][:50]}... ({reason})")
    
    print(f"✓ Pruned {pruned_count} memories. Storage before: {get_storage_size_bytes()} bytes")
```

---

## Part 6: Prototype A Success Metrics

```python
def evaluate_prototype_a():
    """
    Benchmark the memory system on customer service dialogues.
    """
    
    test_dialogues = load_test_set("customer_service_100.json")
    
    # Baseline 1: No memory (pure LLM)
    baseline_no_memory = run_agent(test_dialogues, use_memory=False)
    
    # Baseline 2: Simple RAG (retrieve all, no decay)
    baseline_rag = run_agent(test_dialogues, use_memory=True, decay_enabled=False)
    
    # System: Full memory + Ebbinghaus + FDL
    full_system = run_agent(test_dialogues, use_memory=True, decay_enabled=True, fdl_enabled=True)
    
    # Metrics
    print("=" * 60)
    print("PROTOTYPE A EVALUATION: Customer Service (100 dialogues)")
    print("=" * 60)
    
    print("\n1. ACCURACY (% of preferences correctly remembered)")
    print(f"   No memory: {baseline_no_memory['accuracy']:.1%}")
    print(f"   RAG only:  {baseline_rag['accuracy']:.1%}")
    print(f"   Full system: {full_system['accuracy']:.1%}")
    print(f"   Target: ≥85%")
    
    print("\n2. ERROR REPETITION (% of errors repeated after 1st occurrence)")
    print(f"   No memory: {baseline_no_memory['error_repetition']:.1%}")
    print(f"   RAG only:  {baseline_rag['error_repetition']:.1%}")
    print(f"   Full system: {full_system['error_repetition']:.1%}")
    print(f"   Target: ≤20%")
    
    print("\n3. MEMORY EFFICIENCY (KB of memory store at end)")
    print(f"   RAG only:  {baseline_rag['memory_size_kb']:.0f} KB (no decay)")
    print(f"   Full system: {full_system['memory_size_kb']:.0f} KB (decay enabled)")
    print(f"   Reduction: {(1 - full_system['memory_size_kb']/baseline_rag['memory_size_kb']):.1%}")
    print(f"   Target: ≥30% reduction")
    
    # If all metrics met, Prototype A is successful
    if (full_system['accuracy'] >= 0.85 and 
        full_system['error_repetition'] <= 0.2 and 
        full_system['memory_size_kb'] < baseline_rag['memory_size_kb'] * 0.7):
        print("\n✅ PROTOTYPE A SUCCESSFUL - Proceed to Prototype B")
    else:
        print("\n⚠️ PROTOTYPE A INCOMPLETE - Iterate before proceeding")
```

---

## Part 7: Migration Path (SQLite → PostgreSQL)

Once Prototype A is successful, migrate to PostgreSQL for Prototype B:

```sql
-- PostgreSQL schema (simplified)
CREATE TABLE memories (
  id UUID PRIMARY KEY,
  content TEXT NOT NULL,
  embedding vector(384),  -- pgvector extension
  
  created_at TIMESTAMPTZ DEFAULT NOW(),
  last_accessed TIMESTAMPTZ DEFAULT NOW(),
  access_count INTEGER DEFAULT 0,
  
  importance_score REAL NOT NULL,
  decay_rate REAL NOT NULL,
  current_strength REAL GENERATED ALWAYS AS (
    importance_score * EXP(-decay_rate * EXTRACT(EPOCH FROM (NOW() - last_accessed)) / 86400.0)
  ) STORED,
  
  success_count INTEGER DEFAULT 0,
  failure_count INTEGER DEFAULT 0,
  success_rate REAL GENERATED ALWAYS AS (
    CASE WHEN (success_count + failure_count) = 0 THEN 0.5
         ELSE success_count::REAL / (success_count + failure_count)
    END
  ) STORED,
  
  suppressed BOOLEAN DEFAULT FALSE,
  pruned_at TIMESTAMPTZ,
  
  task_type TEXT,
  category TEXT,
  
  INDEX idx_embedding ON memories USING ivfflat (embedding vector_cosine_ops),
  INDEX idx_strength ON memories (current_strength DESC) WHERE suppressed = FALSE,
  INDEX idx_task ON memories (task_type, success_rate DESC)
);

-- Automatic pruning trigger
CREATE OR REPLACE FUNCTION prune_memories()
RETURNS void AS $$
BEGIN
  UPDATE memories
  SET suppressed = TRUE, pruned_at = NOW()
  WHERE (current_strength < 0.1 OR success_rate < 0.17)
    AND pruned_at IS NULL;
END;
$$ LANGUAGE plpgsql;

SELECT cron.schedule('prune-memories', '0 * * * *', 'SELECT prune_memories()');  -- hourly
```

---

## Summary

This schema and workflow provide the foundation for Prototype A. The key design decisions:

1. **Ebbinghaus decay** is computed on-read (view) for flexibility
2. **Hybrid retrieval** combines BM25 (precision) + semantic search (recall)
3. **FDL** is simplified: detect → explain → retain (no causal blame assignment)
4. **Consolidation** is offline (nightly), avoiding latency during interactions
5. **Pruning** is automated, based on decay + success rate thresholds

Next step: Implement and evaluate on 100-dialogue customer service test set.
