"""
test_custom_qa.py — Autonomous RAG vs Enhanced RAG Benchmark
=============================================================
Tests 20 Constitution of India questions across TWO passes.

Pass 1: Both agents answer. Enhanced agent self-corrects via FDL pipeline.
Pass 2: Same questions again. Enhanced agent should adapt from Pass 1 lessons.

Metrics are computed by the LOCAL LLM (LLM-as-a-Judge) — NO ground truth needed.
The document itself is the source of truth.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import json

from src.utils.pdf_processor import extract_text_from_pdf, chunk_text
from src.rag.embedder import Embedder
from src.memory.memory import SimpleRAGMemory, EnhancedMemory
from src.rag.agent import OllamaRAGAgent
from src.eval.evaluator import LLMJudge
from src.memory.fdl_engine import FDLEngine
from src.rag.graph_rag import KnowledgeGraph
from src.rag.semantic_cache import SemanticCache

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

QUESTIONS = [
    "What did Lencho ask for in his letter to God?",
    "From which tree did the crow shake down the dust of snow?",
    "According to the poem 'Fire and Ice', what are the two ways the world could end?",
    "Where did the inauguration ceremonies for Nelson Mandela take place?",
    "What are the 'twin obligations' that Nelson Mandela mentions every man has in life?",
    "In the poem 'A Tiger in the Zoo', what does the tiger stare at with his brilliant eyes?",
    "Why was the young seagull afraid to make his first flight?",
    "What was the call sign of the old Dakota aeroplane in 'The Black Aeroplane'?",
    "According to 'How to Tell Wild Animals', which beast's hide is peppered with spots?",
    "What does the poet say the boy is learning in 'The Ball Poem'?",
    "What name did Anne Frank give to her long-awaited friend, her diary?",
    "What was the title of the first essay Mr. Keesing assigned to Anne Frank as a punishment?",
    "In the poem 'Amanda!', what does the girl imagine herself to be in the languid, emerald sea?",
    "What was the peculiar, single-piece long frock worn by the traditional Goan baker known as?",
    "Which river obtains its water from the hills and forests of Coorg?",
    "According to the Indian legend in 'Tea from Assam', what grew out of the Buddhist ascetic Bodhidharma's severed eyelids?",
    "What name did the zoologists give to the previously unknown race of otter discovered by Gavin Maxwell?",
    "What was the one-way bus fare from Valli's village to the town in 'Madam Rides the Bus'?",
    "What was the original name of Gautama Buddha before he attained enlightenment?",
    "In Anton Chekhov's play 'The Proposal', what is the name of the disputed land that Lomov and Natalya argue over?"
]

MODEL = "qwen2.5:1.5b"
PDF_PATH = "test_document/10_English_Textbook_2024-25.pdf"


def setup():
    """Load PDF -> chunk -> embed -> create memory stores and agents."""
    t0 = time.time()
    print("Loading PDF...")
    with open(PDF_PATH, "rb") as f:
        text = extract_text_from_pdf(f)
    print(f"  PDF loaded in {time.time()-t0:.1f}s")

    t1 = time.time()
    print("Chunking...")
    chunks = chunk_text(text, chunk_size=250, overlap=30)
    print(f"  {len(chunks)} chunks in {time.time()-t1:.1f}s")

    t2 = time.time()
    print("Batch-embedding chunks with all-MiniLM-L6-v2...")
    embedder = Embedder()
    embedder.fit(chunks)  # no-op for dense model, kept for API compat
    embeddings = embedder.embed_batch(chunks)
    print(f"  Embeddings computed in {time.time()-t2:.1f}s")

    t3 = time.time()
    simple_mem   = SimpleRAGMemory(embed_func=embedder.embed)
    enhanced_mem = EnhancedMemory(embed_func=embedder.embed)
    print("Storing chunks in both memory stores (batch insert)...")
    simple_ids   = simple_mem.store_batch(chunks, embeddings, importance=0.8, category="learned_fact")
    enhanced_ids = enhanced_mem.store_batch(chunks, embeddings, importance=0.8, category="learned_fact")
    print(f"  Stored {len(chunks)} chunks in {time.time()-t3:.1f}s")

    # ── GraphRAG: build knowledge graph on enhanced memory ────────────
    t4 = time.time()
    print("Building knowledge graph (GraphRAG)...")
    graph = KnowledgeGraph()
    graph.build(enhanced_ids, chunks)
    g_stats = graph.stats()
    print(f"  Graph built in {time.time()-t4:.1f}s "
          f"| nodes={g_stats['nodes']} approx_edges={g_stats['approx_edges']} "
          f"entities={g_stats['discriminative_ents']}")

    # ── Semantic Cache ────────────────────────────────────────────────
    cache = SemanticCache(
        embed_func=embedder.embed,
        similarity_threshold=0.92,
        ttl_seconds=7200.0,   # 2-hour TTL for a benchmark run
    )
    print(f"  Semantic cache initialised (threshold={cache.threshold})")

    simple_agent   = OllamaRAGAgent(simple_mem,   "SimpleRAG",   model=MODEL)
    enhanced_agent = OllamaRAGAgent(enhanced_mem, "EnhancedRAG", model=MODEL, graph=graph)
    judge          = LLMJudge(model=MODEL)
    fdl_engine     = FDLEngine(enhanced_agent, judge, cache=cache)

    print(f"Setup complete. Total: {time.time()-t0:.1f}s\n")
    return simple_agent, enhanced_agent, judge, fdl_engine, enhanced_mem, cache


def run_pass(pass_num: int, simple_agent, fdl_engine, judge, cache=None):
    """Run all questions. Metrics by LLM-as-a-Judge (no ground truth)."""
    print(f"\n{'='*70}")
    print(f"  PASS {pass_num}")
    print(f"{'='*70}")

    pass_start = time.time()
    results = []
    simple_faithful_count   = 0
    enhanced_faithful_count = 0
    enhanced_corrected      = 0
    simple_total_conf       = 0.0
    enhanced_total_conf     = 0.0
    simple_total_time       = 0.0
    enhanced_total_time     = 0.0

    for i, q in enumerate(QUESTIONS):
        print(f"\n--- Q{i+1}/{len(QUESTIONS)}: {q}")
        entry = {"question": q}

        # ── Simple RAG ───────────────────────────────────────────────
        t_s = time.time()
        s_answer, s_mids = simple_agent.generate_response(q, decay_enabled=False)
        s_context = "\n".join(
            simple_agent.memory.memories[mid].content
            for mid in s_mids
            if mid in simple_agent.memory.memories
        )
        s_faith = judge.check_faithfulness(q, s_answer, s_context)
        s_elapsed = time.time() - t_s
        simple_total_time += s_elapsed

        entry["simple_answer"]     = s_answer
        entry["simple_faithful"]   = s_faith["faithful"]
        entry["simple_confidence"] = s_faith["confidence"]
        entry["simple_reason"]     = s_faith["reason"]
        entry["simple_time_s"]     = round(s_elapsed, 2)

        if s_faith["faithful"]:
            simple_faithful_count += 1
        simple_total_conf += s_faith["confidence"]

        print(f"  [Simple]   Faithful={s_faith['faithful']}  Conf={s_faith['confidence']:.2f}  Time={s_elapsed:.1f}s")

        # ── Enhanced RAG (FDL pipeline) ──────────────────────────────
        t_e = time.time()
        fdl_result = fdl_engine.ask(q)
        e_elapsed = time.time() - t_e
        enhanced_total_time += e_elapsed

        entry["enhanced_s1_answer"]  = fdl_result["system1_answer"]
        entry["enhanced_s2_answer"]  = fdl_result["system2_answer"]
        entry["enhanced_final"]      = fdl_result["final_answer"]
        entry["enhanced_faithful"]   = fdl_result["faithful"]
        entry["enhanced_confidence"] = fdl_result["confidence"]
        entry["enhanced_corrected"]  = fdl_result["self_corrected"]
        entry["enhanced_reason"]     = fdl_result["reason"]
        entry["enhanced_time_s"]     = round(e_elapsed, 2)

        if fdl_result["faithful"]:
            enhanced_faithful_count += 1
        if fdl_result["self_corrected"]:
            enhanced_corrected += 1
        enhanced_total_conf += fdl_result["confidence"]

        print(f"  [Enhanced] Faithful={fdl_result['faithful']}  Conf={fdl_result['confidence']:.2f}  Corrected={fdl_result['self_corrected']}  Time={e_elapsed:.1f}s")

        results.append(entry)
        time.sleep(0.3)

    # ── Metrics ──────────────────────────────────────────────────────
    pass_elapsed = time.time() - pass_start
    n = len(QUESTIONS)
    metrics = {
        "pass":                       pass_num,
        "simple_faithfulness_rate":   round(simple_faithful_count / n, 3),
        "enhanced_faithfulness_rate": round(enhanced_faithful_count / n, 3),
        "simple_avg_confidence":      round(simple_total_conf / n, 3),
        "enhanced_avg_confidence":    round(enhanced_total_conf / n, 3),
        "enhanced_self_corrections":  enhanced_corrected,
        "enhanced_correction_rate":   round(enhanced_corrected / n, 3),
        "simple_total_time_s":        round(simple_total_time, 1),
        "enhanced_total_time_s":      round(enhanced_total_time, 1),
        "simple_avg_time_s":          round(simple_total_time / n, 2),
        "enhanced_avg_time_s":        round(enhanced_total_time / n, 2),
        "pass_total_time_s":          round(pass_elapsed, 1),
    }

    print(f"\n{'='*70}")
    print(f"  PASS {pass_num} METRICS  ({pass_elapsed:.0f}s total)")
    print(f"{'='*70}")
    print(f"  Simple RAG:")
    print(f"    Faithfulness Rate:  {metrics['simple_faithfulness_rate']:.1%}")
    print(f"    Avg Confidence:     {metrics['simple_avg_confidence']:.2f}")
    print(f"    Total Time:         {metrics['simple_total_time_s']:.1f}s  (avg {metrics['simple_avg_time_s']:.1f}s/q)")
    print(f"  Enhanced RAG (FDL):")
    print(f"    Faithfulness Rate:  {metrics['enhanced_faithfulness_rate']:.1%}")
    print(f"    Avg Confidence:     {metrics['enhanced_avg_confidence']:.2f}")
    print(f"    Self-Corrections:   {metrics['enhanced_self_corrections']}/{n}")
    print(f"    Total Time:         {metrics['enhanced_total_time_s']:.1f}s  (avg {metrics['enhanced_avg_time_s']:.1f}s/q)")

    fname = f"results_pass_{pass_num}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {fname}")

    # Print cache stats if available
    if cache is not None:
        cs = cache.stats()
        print(f"  Cache: {cs['hits']} hits / {cs['hits']+cs['misses']} queries "
              f"({cs['hit_rate']:.1%} hit rate) | {cs['size']} entries stored")

    return metrics


def main():
    total_start = time.time()
    simple_agent, enhanced_agent, judge, fdl_engine, enhanced_mem, cache = setup()

    # ── Pass 1 ────────────────────────────────────────────────────────
    m1 = run_pass(1, simple_agent, fdl_engine, judge, cache=cache)

    # ── Decay + Prune ─────────────────────────────────────────────────
    print("\n" + "*"*70)
    print("  Simulating 3 days of memory decay + pruning...")
    enhanced_mem.advance_time(days=3.0)
    pruned = enhanced_mem.prune()
    print(f"  Pruned {pruned} stale/low-worth memories.")
    print(f"  Enhanced Memory: {enhanced_mem.get_memory_size_kb():.1f} KB")
    print("*"*70)

    # ── Pass 2 ────────────────────────────────────────────────────────
    # Cache carries over from Pass 1 — Pass 2 questions that are semantically
    # similar to Pass 1 answers will hit the cache instantly.
    m2 = run_pass(2, simple_agent, fdl_engine, judge, cache=cache)

    # ── Final Comparison ──────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON: Pass 1 vs Pass 2  (total: {total_elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"  {'Metric':<35} {'Pass 1':>10} {'Pass 2':>10} {'Delta':>10}")
    print(f"  {'-'*65}")

    for key in ["simple_faithfulness_rate", "enhanced_faithfulness_rate",
                 "simple_avg_confidence", "enhanced_avg_confidence",
                 "enhanced_correction_rate",
                 "simple_avg_time_s", "enhanced_avg_time_s"]:
        v1 = m1[key]
        v2 = m2[key]
        delta = v2 - v1
        sign = "+" if delta >= 0 else ""
        print(f"  {key:<35} {v1:>10.3f} {v2:>10.3f} {sign}{delta:>9.3f}")

    print(f"\n  Total benchmark time: {total_elapsed:.0f}s")
    print("Done.")


if __name__ == "__main__":
    main()
