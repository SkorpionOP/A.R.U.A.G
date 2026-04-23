# RAG + FDL Memory System — Technical Pipeline Reference

> A complete reference document covering every component, formula, algorithm,
> and design decision in the autonomous RAG + Failure-Driven Learning pipeline.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Ingestion Layer](#2-ingestion-layer)
3. [Embedding Engine](#3-embedding-engine)
4. [Memory Architecture](#4-memory-architecture)
5. [Retrieval Engine](#5-retrieval-engine)
6. [Agent Layer](#6-agent-layer)
7. [LLM-as-a-Judge Evaluator](#7-llm-as-a-judge-evaluator)
8. [FDL Engine — Self-Correction Loop](#8-fdl-engine--self-correction-loop)
9. [Memory Maintenance — Decay & Pruning](#9-memory-maintenance--decay--pruning)
10. [Benchmark Pipeline](#10-benchmark-pipeline)
11. [Streamlit Frontend](#11-streamlit-frontend)
12. [Design Decisions & Known Limitations](#12-design-decisions--known-limitations)

---

## 1. System Overview

This system compares two RAG agents operating on the same document corpus:

| Agent | Strategy |
|---|---|
| **SimpleRAGMemory** | Store all chunks. Retrieve by cosine similarity. No learning. |
| **EnhancedRAGMemory** | All of the above + Ebbinghaus decay + confidence scores + Failure-Driven Learning (FDL) that self-corrects without ground truth. |

The core hypothesis from the *"AI Memory and Failure Learning"* paper:

> **Memory should behave like a living substrate — not a static filing cabinet.**
> Facts that are recalled successfully get stronger. Facts that lead to wrong
> answers get weaker and are eventually pruned. New corrections must be
> validated against the document before being stored.

---

## 2. Ingestion Layer

**Module:** `pdf_processor.py`

### 2.1 PDF Extraction

```
raw_text = PdfReader(file).pages[i].extract_text()  for all pages
```

### 2.2 Chunking

Sliding-window word-level chunking:

```
Parameters:
  chunk_size = 250 characters
  overlap    = 30 words

Algorithm:
  for word in words:
    append word to current_chunk
    current_length += len(word) + 1

    if current_length >= chunk_size:
      chunks.append(join(current_chunk))
      current_chunk = last_overlap_words   <- preserve context across boundary
      reset current_length
```

**Why overlap?** Prevents answers from being split across chunk boundaries.
**Filter:** Chunks with `len < 10` characters are discarded (headers, page numbers).

---

## 3. Embedding Engine

**Module:** `embedder.py`

### 3.1 Dense Semantic Embedder (all-MiniLM-L6-v2)

```
Model:  all-MiniLM-L6-v2 (sentence-transformers)
Output: 384-dimensional L2-normalised dense vector
Method: Single forward pass through the transformer
```

**Why dense over TF-IDF?**
- Understands synonyms and paraphrases natively ("FR" ≈ "Fundamental Rights")
- Fixed 384-d output regardless of corpus vocabulary — zero dimension-mismatch crashes
- Eliminates the need for LLM-based Query Rewriting on every query

### 3.2 No Corpus Fitting Required

Unlike TF-IDF, `all-MiniLM-L6-v2` is a pre-trained model. `fit()` is kept
as a no-op for API compatibility. New PDFs can be ingested without rebuilding
any index, and query embeddings always match stored embeddings dimensionally.

---

## 4. Memory Architecture

**Module:** `memory.py`

### 4.1 Memory Object Schema

| Field | Type | Purpose |
|---|---|---|
| `id` | str (UUID4) | Unique identifier |
| `content` | str | Raw text content |
| `embedding` | np.ndarray | TF-IDF vector |
| `created_at` | datetime | Creation timestamp |
| `last_accessed` | datetime | Last retrieval timestamp |
| `access_count` | int | Total number of retrievals |
| `importance_score` | float [0,1] | Ebbinghaus strength anchor |
| `decay_rate` | float | lambda in decay formula |
| `success_count` | int | Successful retrievals |
| `failure_count` | int | Failed retrievals |
| `confidence_score` | float [0,1] | Proven reliability (FIX 2) |
| `category` | str | learned_fact / failure_note / user_preference |
| `suppressed` | bool | Pruned flag (soft delete) |

### 4.2 Memory Categories

| Category | Source | Decay Rate lambda | Default Importance |
|---|---|---|---|
| `learned_fact` | PDF chunks | 0.20 | 0.80 |
| `failure_note` | FDL corrections | **1.00** (fast!) | 0.80 |
| `user_preference` | Manual feedback | 0.10 | 0.95 |

**Why fast decay for failure notes?** A correction that was needed once may not be relevant again. If it IS retrieved and leads to a successful answer, its confidence_score rises. If never reused, it decays and is pruned within ~17 hours of simulated time.

---

## 5. Retrieval Engine

**Module:** `memory.py` -> `EnhancedMemory.retrieve()`

### 5.1 Priority Scoring

Each candidate memory gets a composite priority score:

```
priority(mem, query, t) =
    cosine_sim(query, mem)          <- semantic relevance
  x current_strength(mem, t)        <- Ebbinghaus decay
  x recency_boost(mem, t)           <- recent = more useful
  x confidence_score(mem)           <- proven reliability

Where:
  current_strength(mem, t) = importance x exp(-lambda x days_since_access)

  recency_boost(mem, t)    = 1.0 + 0.3 x exp(-0.1 x days_since_access)
                           -> ranges from 1.30 (fresh) to 1.00 (old)

  confidence_score(mem)    = starts at 0.50
                             floor at 0.30 for untested memories
```

### 5.2 Confidence Gate (FIX 2)

Memory notes are NOT considered unless:
```
mem.confidence_score >= 0.35
```
A new correction starts at `0.40` — just above the gate — and must earn more trust through successful reuse.

### 5.3 Hard Memory Slot Cap (FIX 3)

```
MAX_MEMORY_SLOTS = 1

doc_slots = k - 1   (e.g., k=3 -> 2 doc slots)
mem_slots = 1       (always exactly 1 memory slot maximum)
```

### 5.4 Document vs Memory Arbitration (FIX 4)

Before including a memory note in results:
```
doc_embeddings = [memories[mid].embedding for mid in selected_docs]
avg_doc_sim = mean(cosine(mem.embedding, de) for de in doc_embeddings)

if avg_doc_sim >= 0.02:
    include memory    <- topically compatible with document context
else:
    discard memory    <- contradicts document, document wins
```

### 5.5 Two-Pool Retrieval Summary

```
Pool 1 (doc_scores):  all learned_fact memories, sorted by priority
Pool 2 (mem_scores):  failure_note/user_pref, filtered by confidence >= 0.35

Final result = top-(k-1) from Pool 1 + top-1 from Pool 2 (if passes arbitration)
             -> re-sorted by score -> top-k returned
```

---

## 6. Agent Layer

**Module:** `agent.py` -> `OllamaRAGAgent`

### 6.1 Query Rewriting — REMOVED ✅

Previously, an LLM was invoked to expand abbreviations before every retrieval
call because TF-IDF only matches exact vocabulary. With `all-MiniLM-L6-v2`,
the embedder understands synonyms natively — no rewriting needed.

```
OLD: query → LLM(rewrite) → expanded_query → TF-IDF retrieval
NEW: query → semantic_embed → cosine retrieval   (one step removed)
```

### 6.2 Top-5 Retrieval + Cross-Encoder Reranking ✅

```
Step 1: retrieve(query, k=5)              <- dense cosine similarity
Step 2: CrossEncoder reranker:
  Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  Input: (query, chunk) pairs for all 5 candidates
  Output: relevance scores → sort descending → pick top 2
Step 3: Build context from best-2 chunks

Why Cross-Encoder vs. LLM Reranker?
  - Trained specifically for passage relevance — not a generative task
  - Runs in milliseconds vs. seconds per LLM call
  - Deterministic output; no fragile text parsing required
```

### 6.3 Answer Generation

```
Prompt:
  "Use ONLY the context below.
   Answer in ONE SHORT FACTUAL LINE. No explanation.
   If context lacks answer: say NOT_FOUND.
   Context: {best-2 chunks}
   Question: {query}"
```

### 6.4 System-2 Single-Prompt Synthesis ✅

```
OLD Map-Reduce:
  retrieve(q, k=15) → for each chunk: separate LLM call → up to 15 calls

NEW Single-Prompt:
  retrieve(q, k=5, doc_only) → inject all 5 chunks into ONE prompt
  LLM synthesises answer from combined context in 1 inference call

Prompt:
  "[Chunk 1]\n{content}\n\n[Chunk 2]\n{content}\n...
   Question: {q}
   Answer ONE SHORT FACTUAL LINE, or NOT_FOUND"
```

---

## 7. LLM-as-a-Judge Evaluator

**Module:** `evaluator.py` -> `LLMJudge`

### 7.1 Faithfulness Check — Conditional Invocation ✅

```
Condition: only invoked when System-1 retrieval confidence < 0.75
  - If the cross-encoder already returned a high-confidence top chunk,
    skip the judge entirely (one less LLM call per query)

Prompt:
  "Strict fact-checker. Use ONLY this context:
   {doc_only_context}     <- NEVER includes memory notes (FIX 4)
   Q: {question}
   Answer: {answer}
   
   Reply EXACTLY:
   VERDICT: YES or NO
   CONFIDENCE: 0.0 to 1.0
   REASON: one sentence"

Acceptance gate: faithful == True AND confidence >= 0.6
```

### 7.2 Consistency Check (Fallback)

```
1. rephrase_question(q)             -> alternate phrasing
2. generate_response(rephrased_q)   -> second answer
3. check_consistency(q, a1, a2):
   "Same factual meaning? CONSISTENT: YES/NO"
   
If CONSISTENT -> accept s1_answer, confidence override = 0.65
```

---

## 8. FDL Engine — Self-Correction Loop

**Module:** `fdl_engine.py` -> `FDLEngine.ask()`

### 8.1 Full Decision Flow

```
ask(question)
|
+-- STEP 1: System-1 Fast Answer
|   agent.generate_response(q, decay=True) -> (s1_answer, s1_mids)
|
+-- Build doc-only context (FIX 4)
|   EXCLUDE failure_note memories
|   -> (doc_context, doc_embeddings)
|
+-- STEP 2: Faithfulness Check
|   judge.check_faithfulness(q, s1, doc_context)
|   |
|   +-- faithful AND conf >= 0.6?
|   |   YES -> log_success() -> return s1_answer
|   |
|   +-- NO -> STEP 3
|
+-- STEP 3: System-2 Map-Reduce
|   extract_correction(q) -> s2_answer
|   |
|   +-- VALIDATION (FIX 1)
|   |   avg cosine(s2_answer_emb, doc_embeddings) >= 0.02?
|   |   NO  -> REJECT (hallucination) -> STEP 4
|   |   YES -> re-verify:
|   |          check_faithfulness(q, s2, deep_doc_context)
|   |          faithful? YES -> log_failure_and_learn() -> return s2
|   |          faithful? NO  -> STEP 4
|   |
|   +-- STEP 4: Consistency Check (Last Resort)
|       rephrase -> re-ask -> compare
|       consistent? YES -> log_success() -> return s1 (conf=0.65)
|       consistent? NO  -> log_failure() -> return best_available
```

### 8.2 Correction Validation Formula (FIX 1)

```
corr_emb = embed(correction_text)
avg_sim  = mean(cosine(corr_emb, de) for de in doc_embeddings)

if avg_sim >= 0.02:  ACCEPT -> store as failure_note
else:                REJECT -> do not store (block hallucination poisoning)
```

**Threshold 0.02:** A hallucination from parametric training data (e.g., "Mountbatten was president") has near-zero similarity to actual Constitution chunks, well below 0.02.

### 8.3 Concept Extraction for Structured Memory (FIX 3)

```
"How many Fundamental Rights are currently recognized in India?"
  -> strip prefix "how many "
  -> strip fillers "the", "in india"
  -> "fundamental rights currently recognized"

Stored as: "fundamental rights currently recognized: 6 Fundamental Rights"
```

Benefits generalization across differently-phrased questions on the same topic.

### 8.4 Confidence Score Updates

```
On SUCCESS:
  confidence_score = min(1.0, confidence + 0.10)
  importance_score = min(1.0, importance + 0.02)

On FAILURE:
  confidence_score = max(0.0, confidence - 0.15)    <- penalises harder
  importance_score = max(0.05, importance - 0.05)

Example trajectory:
  New note (conf=0.40) -> 1 success (0.50) -> 1 failure (0.35)
  -> 3 successes (0.65) -> fully trusted
```

---

## 9. Memory Maintenance — Decay & Pruning

### 9.1 Ebbinghaus Decay Formula

```
S(t) = I x exp(-lambda x delta_t)

Where:
  S(t)    = current memory strength at time t
  I       = importance_score (0.0 to 1.0)
  lambda  = decay_rate (category-dependent, per day)
  delta_t = days since last_accessed
```

### 9.2 Decay Rate Table

| Category | lambda | Half-life | Strength at 3 days (I=0.80) |
|---|---|---|---|
| `user_preference` | 0.10 | 6.9 days | 0.59 |
| `learned_fact` | 0.20 | 3.5 days | 0.44 |
| `failure_note` | **1.00** | **17 hours** | **0.04** |

### 9.3 Why Old System Pruned 0 Memories

```
Old: lambda=0.08, threshold=0.20
After 3 days: S = 0.80 x exp(-0.08 x 3) = 0.63  <- above 0.20, not pruned

New: lambda=0.20, threshold=0.40
After 3 days: S = 0.80 x exp(-0.20 x 3) = 0.44  <- barely above (pruned at ~5 days)

Failure notes: lambda=1.00, threshold=0.40
After 3 days: S = 0.80 x exp(-1.00 x 3) = 0.04  <- far below, pruned immediately
```

### 9.4 Pruning Conditions (FIX 5)

A memory is suppressed if ANY of:
```
Condition 1: current_strength(t) < 0.40

Condition 2: failure_count >= 2
             AND memory_worth() < 0.30
             memory_worth = 0.6 x success_rate + 0.4 x confidence_score

Condition 3: category == "failure_note"
             AND confidence_score < 0.15
```

---

## 10. Benchmark Pipeline

**Module:** `test_custom_qa.py`

### 10.1 Two-Pass Design

```
PASS 1: 20 questions
  Both agents answer each question
  FDL stores validated corrections
  Per-question: faithful, confidence, corrected, time

advance_time(3 days) + prune()

PASS 2: Same 20 questions
  Enhanced agent now uses Pass 1 corrections
  (decay ensures stale/bad ones are gone)
```

### 10.2 Metrics (No Ground Truth)

| Metric | Formula |
|---|---|
| Faithfulness Rate | count(faith=True) / N |
| Avg Confidence | sum(confidence) / N |
| Self-Correction Rate | count(corrected) / N |
| Avg Time (s/q) | total_time / N |

---

## 11. Streamlit Frontend

**Module:** `app.py`

Session state resets completely on new PDF upload to avoid TF-IDF dimension mismatch.
Each answer shows: text, faithful badge, confidence score, whether it self-corrected.

Memory sidebar shows:
- Simple memory size (KB)
- Enhanced memory size (KB)
- Reduction % = (1 - enhanced_kb / simple_kb) x 100

---

## 12. Design Decisions & Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| LLM judge is also 1.5b | Judge can misjudge | Confidence gate at 0.6 + conditional invocation |
| Semantic embedder may over-generalise | Slight topic drift in retrieval | Cross-encoder reranker corrects ranking |
| Single-prompt System-2 may dilute context | LLM attention spread across chunks | Top-5 limit keeps context focused |
| 0.02 correction validation may be loose | Marginal hallucinations slip through | Confidence gate filters on next use |
| No persistence across restarts | Memory wiped | In-memory design; SQLite is next step |

### 12.1 Hyperparameter Reference

| Parameter | Value | File | Effect |
|---|---|---|---|
| chunk_size | 250 chars | pdf_processor.py | Concise chunks for synthesis prompt |
| chunk_overlap | 30 words | pdf_processor.py | Boundary context |
| embed_model | all-MiniLM-L6-v2 | embedder.py | 384-d semantic embeddings |
| reranker_model | ms-marco-MiniLM-L-6-v2 | agent.py | Cross-encoder passage scoring |
| k (System-1) | 5 → 2 | agent.py | Wide retrieval, tight answer |
| k (System-2) | 5 | agent.py | Single-prompt synthesis window |
| judge_threshold | 0.75 | fdl_engine.py | Skip judge above this retrieval score |
| faith_threshold | 0.6 | fdl_engine.py | Accept System-1 answer |
| confidence_boost | +0.10 | memory.py | Per successful retrieval |
| confidence_penalty | -0.15 | memory.py | Per failed retrieval |
| min_confidence | 0.35 | memory.py | Memory retrieval gate |
| new_note_confidence | 0.40 | memory.py | Above gate, must earn trust |
| MAX_MEMORY_SLOTS | 1 | memory.py | Hard cap on memory influence |
| correction_min_sim | 0.02 | memory.py / fdl_engine.py | Validation threshold |
| STRENGTH_PRUNE | 0.40 | memory.py | Ebbinghaus prune gate |
| WORTH_PRUNE | 0.30 | memory.py | Track-record prune gate |
| MIN_FAILURES | 2 | memory.py | Before worth pruning applies |
| lambda (learned_fact) | 0.20 /day | memory.py | Half-life 3.5 days |
| lambda (failure_note) | 1.00 /day | memory.py | Half-life 17 hours |
| lambda (user_pref) | 0.10 /day | memory.py | Half-life 7 days |
| recency_boost_max | +30% | memory.py | Score bonus for fresh access |
