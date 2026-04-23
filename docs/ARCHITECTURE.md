# Architecture Deep Dive

This document provides a highly technical, end-to-end breakdown of the Enhanced RAG Memory Extension. It details exactly how data is stored under the hood, how the system autonomously learns from its mistakes, and the mathematics governing memory metabolism.

---

## 1. Data Storage & Schema

Unlike standard RAG which blindly dumps embeddings into a rigid vector database, this system utilizes a dynamic "Cognitive Stack" where memories are deeply characterized and mutable.

### The Memory Node Schema
Every piece of information (whether it's a chunk from an ingested PDF or a dynamically generated "Failure Note") is stored as a `Memory` dataclass in RAM.

```python
@dataclass
class Memory:
    id: str                  # Unique UUID
    text: str                # The actual content
    vector: np.ndarray       # The all-MiniLM-L6-v2 embedding
    timestamp: datetime      # Creation time
    last_accessed: datetime  # Time of last successful retrieval
    strength: float          # Current importance score (0.0 to 1.0)
    category: str            # "learned_fact", "failure_note", or "document_chunk"
    suppressed: bool         # If True, it is excluded from semantic searches
```

### Persistence Layer
During execution, the `EnhancedMemory` class holds these nodes in a standard Python dictionary (`Dict[str, Memory]`). When `.save("brain.pkl")` is called, the system bypasses complex external database requirements by serializing this exact dictionary structure—alongside the Semantic Cache LRU list and the GraphRAG NetworkX edges—into a highly compressed byte-stream on the disk using Python's `pickle`. 

---

## 2. How the System "Learns" (Failure-Driven Learning)

The core innovation of this system is the **FDL Engine** (`src/memory/fdl_engine.py`). It does not require human-labeled ground truth to improve. Instead, it relies on a strict "System-1 / System-2" correction loop.

### The Learning Sequence Flow
1. **System-1 Attempt (Fast Retrieval)**: 
   When a user asks a query, the agent performs a standard vector cosine-similarity search to find the top-k memories. It quickly generates a provisional answer.
   
2. **The Adjudication (`LLMJudge`)**: 
   Before returning the answer to the user, an independent umpire (the `LLMJudge`) is invoked in the background. It is given the raw retrieved memory texts and the agent's provisional answer. It strictly evaluates if the answer is faithful to the context or if it is a hallucination.

3. **System-2 Correction (Deep Search)**: 
   If the judge outputs `"False"` (indicating a failure), the engine intercepts the response. It triggers a deep search (lowering similarity thresholds, increasing `k`, and querying the GraphRAG relationships) to find the absolute truth. The agent then generates a new, corrected answer.

4. **Failure Note Generation**: 
   To ensure the agent *never makes the same mistake again*, the engine dynamically generates a targeted "Failure Note" summarizing the error and the correction. 
   *Example: "[CORRECTION]: The agent previously claimed Lencho asked for 50 pesos, but the document explicitly states he asked for 100 pesos."*
   This string is passed back to the Embedder, assigned a vector, given a high initial `strength` of `1.0`, and permanently saved into the `EnhancedMemory` dictionary under `category="failure_note"`.

5. **Future Queries**: 
   Because the Failure Note is semantically dense, the next time the user asks "How much did Lencho ask for?", the vector search will return the exact Failure Note as the #1 result, guaranteeing a perfect, fast response.

---

## 3. Memory Metabolism (Ebbinghaus Decay)

A system that continuously generates new Failure Notes and user preference notes will eventually suffer from "Context Rot" (database bloat causing slow, noisy retrievals). 

This system mathematically models human forgetfulness using the **Ebbinghaus Forgetting Curve** to solve this.

### The Mathematics of Forgetting
Every time time passes (simulated via `simulate_decay()`), the system iterates over the memory dictionary and applies the following decay formula to the `strength` of each node:

$$ S(t) = S_0 \times e^{-\lambda t} $$
- **$S(t)$**: Current memory strength
- **$S_0$**: Initial or previous strength
- **$\lambda$**: Decay rate (e.g., `0.05` for standard facts, `0.01` for critical failure notes)
- **$t$**: Days since `last_accessed`

### Pruning and Reinforcement
- **Reinforcement**: If a memory is successfully retrieved and the `LLMJudge` verifies it was useful in answering the user, its `last_accessed` timestamp is updated to `now()`, and its strength is boosted back up. 
- **Pruning**: If a memory is never used and its strength drops below the `suppression_threshold` (e.g., `0.17`), `suppressed` is set to `True`. It physically remains on disk but is completely excluded from all future vector cosine-similarity searches, instantly speeding up retrieval times for active memories.

---

## 4. Sub-Millisecond Semantic Caching

To further optimize performance, `src/rag/semantic_cache.py` sits at the very front of the architecture.

1. **Storage**: It maintains an LRU (Least Recently Used) list of dictionary entries: `{query_vector, answer_text, timestamp}`.
2. **Lookup**: When a new query arrives, it is instantly embedded. The system calculates the dot-product cosine similarity against all previously cached queries.
3. **Thresholding**: If `similarity > 0.92`, the system mathematically assumes the queries are paraphrased versions of each other (e.g., "What did Lencho want?" vs "What did Lencho ask for?"). 
4. **Bypass**: The system instantly returns the cached `answer_text` and completely skips the LLM generation and the `LLMJudge` verification, dropping latency from ~3000ms down to ~2ms.
