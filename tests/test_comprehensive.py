"""
test_comprehensive.py — Full-Coverage 3-Pass Benchmark
=======================================================
Real-world query diversity across 5 categories:

  DIRECT_FACTUAL  — Simple who/what/when recall
  INFERENCE       — Reasoning required from context
  MULTI_HOP       — Answer needs multiple chunks (GraphRAG advantage)
  NEAR_PARAPHRASE — Same meaning, different wording (cache stress-test)
  OUT_OF_SCOPE    — Not in document (NOT_FOUND robustness test)

3-Pass Design:
  Pass 1 (Cold)       : DIRECT_FACTUAL + INFERENCE  — seeds cache + FDL
  Pass 2 (Cache Test) : NEAR_PARAPHRASE + DIRECT_FACTUAL subset — hits cache
  Pass 3 (Post-Decay) : All categories after 3-day time skip + prune

Each pass saves results_comprehensive_pass_N.json
Final report shows per-category breakdown + 3-way comparison table.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import json
from collections import defaultdict

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

MODEL    = "qwen2.5:1.5b"
PDF_PATH = "test_document/10_English_Textbook_2024-25.pdf"

# ─────────────────────────────────────────────────────────────────────────────
# Question Suites  (category, question_text)
# ─────────────────────────────────────────────────────────────────────────────

DIRECT_FACTUAL = [
    # A Letter to God
    ("A Letter to God",           "What did Lencho ask for in his letter to God?"),
    ("A Letter to God",           "How much money did Lencho originally ask God to send him?"),
    ("A Letter to God",           "Who collected and replied to Lencho's letter on behalf of the post office?"),
    # Nelson Mandela
    ("Nelson Mandela",            "Where did the inauguration ceremonies for Nelson Mandela take place?"),
    ("Nelson Mandela",            "What are the twin obligations Nelson Mandela says every man has in life?"),
    ("Nelson Mandela",            "Who were the two Deputy Presidents sworn in at Mandela's inauguration?"),
    # Two Stories About Flying
    ("His First Flight",          "Why was the young seagull afraid to make his first flight?"),
    ("His First Flight",          "What did the young seagull's mother use to lure him into flying?"),
    ("The Black Aeroplane",       "What did the pilot find when he searched for the mysterious black aeroplane after landing?"),
    ("The Black Aeroplane",       "What was wrong with the pilot's compass and fuel gauge during the storm?"),
    # Poetry
    ("Dust of Snow",              "From which tree did the crow shake down the dust of snow?"),
    ("Fire and Ice",              "What are the two ways the world could end according to the poem Fire and Ice?"),
    ("A Tiger in the Zoo",        "What does the tiger stare at with his brilliant eyes in the poem?"),
    ("How to Tell Wild Animals",  "Which beast's hide is peppered with spots according to the poem?"),
    ("The Ball Poem",             "What does the poet say the boy is learning in The Ball Poem?"),
    ("Amanda!",                   "What does Amanda imagine herself to be in the languid emerald sea?"),
    # Anne Frank
    ("Anne Frank",                "What name did Anne Frank give to her diary as her long-awaited friend?"),
    ("Anne Frank",                "What was the title of the first essay Mr. Keesing assigned Anne Frank as punishment?"),
    ("Anne Frank",                "How long had Anne Frank been at the new school before writing the diary entry?"),
    # Glimpses of India
    ("A Baker from Goa",         "What was the single-piece long frock worn by the traditional Goan baker called?"),
    ("Coorg",                    "Which river obtains its water from the hills and forests of Coorg?"),
    ("Tea from Assam",           "What grew out of Bodhidharma's severed eyelids according to Indian legend?"),
    # Mijbil the Otter
    ("Mijbil the Otter",         "What name did zoologists give to the previously unknown race of otter found by Gavin Maxwell?"),
    # Madam Rides the Bus
    ("Madam Rides the Bus",      "What was the one-way bus fare from Valli's village to the town?"),
    # The Proposal
    ("The Proposal",             "What is the name of the disputed land that Lomov and Natalya argue over in The Proposal?"),
    # Footprints Without Feet
    ("A Triumph of Surgery",     "What was wrong with Tricki, Mrs Pumphrey's dog, at the start of the story?"),
    ("The Thief's Story",        "What was the name of the young thief in The Thief's Story?"),
    ("Footprints Without Feet",  "How did Griffin become invisible in the story Footprints Without Feet?"),
    ("The Necklace",             "How long did Matilda and her husband work to repay the debt for the lost necklace?"),
    ("Bholi",                    "Why was Bholi considered a problem child by her parents?"),
]

INFERENCE = [
    ("A Letter to God",     "Why did Lencho feel angrier towards the post office employees than towards the hailstorm?"),
    ("Nelson Mandela",      "What does Mandela mean when he says the oppressor is also a prisoner?"),
    ("His First Flight",    "What does the young seagull's fear of flying symbolise about growing up?"),
    ("The Black Aeroplane", "What is the most likely explanation for the mysterious black aeroplane the pilot encountered?"),
    ("Anne Frank",          "Why did Anne Frank feel she could confide more in her diary than in real people?"),
    ("A Baker from Goa",    "What do the bread-baking traditions described tell us about Goa's colonial history?"),
    ("Madam Rides the Bus", "What does Valli's careful planning for the bus ride reveal about her character?"),
    ("Amanda!",             "What is the central conflict the poem Amanda! explores between the child and the adult?"),
    ("A Tiger in the Zoo",  "What contrast does the poem draw between the tiger's natural habitat and the zoo?"),
    ("The Ball Poem",       "What larger lesson about life and loss does the ball symbolise for the boy?"),
    ("Dust of Snow",        "How does a small moment of joy from nature change the poet's entire mood in Dust of Snow?"),
    ("The Thief's Story",   "Why did Hari Singh decide not to rob Anil even though he had the money in his hand?"),
    ("The Necklace",        "How does the ending of The Necklace change our understanding of Matilda's sacrifice?"),
    ("Footprints Without Feet", "What does Griffin's invisibility allow him to do, and what moral problems does this create?"),
    ("Bholi",               "How does education change Bholi's self-confidence by the end of the story?"),
]

MULTI_HOP = [
    ("A Letter to God",     "What sequence of events led Lencho to conclude that the post office employees had stolen part of God's money?"),
    ("Nelson Mandela",      "What historical injustices does Mandela specifically refer to that led to the new South African democracy?"),
    ("His First Flight",    "Trace the complete emotional journey of the young seagull from paralysis to his first successful flight."),
    ("Anne Frank",          "What events led to Mr. Keesing finally giving up his punishments of Anne Frank?"),
    ("Madam Rides the Bus", "What did Valli plan and observe on her complete round-trip bus journey?"),
    ("Mijbil the Otter",    "Describe all the unusual and playful behaviours Mijbil displayed after Gavin Maxwell got him."),
    ("The Proposal",        "What are all the different things that Lomov and Natalya argue about during the play?"),
    ("A Triumph of Surgery","What was the complete treatment Dr Herriot gave Tricki and what was the outcome?"),
    ("Footprints Without Feet", "Describe all the actions Griffin took while invisible that eventually led to his arrest at Iping."),
    ("The Necklace",        "Describe the chain of events from losing the necklace to discovering the truth about it."),
]

NEAR_PARAPHRASE = [
    # Pairs semantically equivalent to DIRECT_FACTUAL for cache stress-test
    ("A Letter to God",    "In the story A Letter to God, what was Lencho's request in the letter he sent?"),
    ("A Letter to God",    "How much did Lencho pray for God to send him after the hailstorm destroyed his crops?"),
    ("Nelson Mandela",     "In which location did Nelson Mandela's presidential inauguration ceremony take place?"),
    ("Nelson Mandela",     "What two responsibilities does Mandela say every man owes — one to family and one broader?"),
    ("Dust of Snow",       "Which kind of tree does the crow sit in when it shakes dust of snow on the poet?"),
    ("Fire and Ice",       "In Robert Frost's poem, what two forces could bring about the destruction of the world?"),
    ("A Tiger in the Zoo", "What is the tiger gazing at through the bars with his shining eyes in the poem?"),
    ("His First Flight",   "What made the young seagull hesitant and afraid to attempt his very first flight?"),
    ("A Baker from Goa",   "What was the name of the traditional outfit the Goan bread-seller wore?"),
    ("Coorg",              "What is the water source for the river flowing through the forests of Coorg?"),
    ("Anne Frank",         "What did Anne Frank name her diary, treating it as her dearest friend?"),
    ("Amanda!",            "What sea creature does Amanda daydream about being while floating in the emerald sea?"),
    ("Mijbil the Otter",   "What scientific name did researchers assign to the new species of otter Maxwell discovered?"),
    ("Madam Rides the Bus","How much did a single ticket cost on the bus Valli took to the town?"),
    ("The Proposal",       "What piece of land is at the centre of the argument between Lomov and Natalya in Chekhov's play?"),
    ("The Thief's Story",  "What is the real name of the young thief who befriends Anil in The Thief's Story?"),
    ("Footprints Without Feet", "How did Griffin achieve invisibility in the story Footprints Without Feet?"),
    ("The Necklace",       "How many years did Matilda and her husband spend repaying the debt from losing the necklace?"),
    ("Bholi",              "Why did Bholi's parents see her as a burden at the beginning of the story?"),
    ("A Triumph of Surgery","What illness or condition made Tricki need medical attention in A Triumph of Surgery?"),
]

OUT_OF_SCOPE = [
    ("Out of scope", "What is the capital of France?"),
    ("Out of scope", "How does photosynthesis work in plants?"),
    ("Out of scope", "Who wrote the Mahabharata?"),
    ("Out of scope", "What is the speed of sound in air at sea level?"),
    ("Out of scope", "Who invented the telephone?"),
    ("Out of scope", "What is the chemical formula for table salt?"),
    ("Out of scope", "Who was the first man to walk on the moon?"),
    ("Out of scope", "What is the Pythagorean theorem?"),
    ("Out of scope", "What is the largest planet in our solar system?"),
    ("Out of scope", "How many bones are there in the adult human body?"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Pass Configurations
# ─────────────────────────────────────────────────────────────────────────────

PASS_1_QUESTIONS = DIRECT_FACTUAL + INFERENCE          # 45 q — cold start, seeds cache
PASS_2_QUESTIONS = NEAR_PARAPHRASE + DIRECT_FACTUAL[:10]  # 30 q — cache stress-test
PASS_3_QUESTIONS = DIRECT_FACTUAL[10:] + MULTI_HOP + OUT_OF_SCOPE  # 40 q — post-decay

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

def setup():
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
    print("Batch-embedding with all-MiniLM-L6-v2...")
    embedder = Embedder()
    embedder.fit(chunks)
    embeddings = embedder.embed_batch(chunks)
    print(f"  Embeddings done in {time.time()-t2:.1f}s")

    t3 = time.time()
    simple_mem   = SimpleRAGMemory(embed_func=embedder.embed)
    enhanced_mem = EnhancedMemory(embed_func=embedder.embed)
    print("Storing chunks (batch insert)...")
    simple_ids   = simple_mem.store_batch(chunks, embeddings, importance=0.8, category="learned_fact")
    enhanced_ids = enhanced_mem.store_batch(chunks, embeddings, importance=0.8, category="learned_fact")
    print(f"  Stored {len(chunks)} chunks in {time.time()-t3:.1f}s")

    t4 = time.time()
    print("Building knowledge graph (GraphRAG)...")
    graph = KnowledgeGraph()
    graph.build(enhanced_ids, chunks)
    g = graph.stats()
    print(f"  Graph: nodes={g['nodes']} approx_edges={g['approx_edges']} entities={g['discriminative_ents']} ({time.time()-t4:.1f}s)")

    cache = SemanticCache(embed_func=embedder.embed, similarity_threshold=0.92, ttl_seconds=10800.0)
    print(f"  Semantic cache ready (threshold={cache.threshold})")

    simple_agent   = OllamaRAGAgent(simple_mem,   "SimpleRAG",   model=MODEL)
    enhanced_agent = OllamaRAGAgent(enhanced_mem, "EnhancedRAG", model=MODEL, graph=graph)
    judge          = LLMJudge(model=MODEL)
    fdl_engine     = FDLEngine(enhanced_agent, judge, cache=cache)

    print(f"Setup complete in {time.time()-t0:.1f}s\n")
    return simple_agent, enhanced_agent, judge, fdl_engine, enhanced_mem, cache

# ─────────────────────────────────────────────────────────────────────────────
# Run a pass
# ─────────────────────────────────────────────────────────────────────────────

def run_pass(pass_num: int, questions: list, simple_agent, fdl_engine, judge, cache=None):
    """
    Run a list of (category, question) tuples.
    Returns (metrics_dict, results_list).
    """
    print(f"\n{'='*72}")
    print(f"  PASS {pass_num}  ({len(questions)} questions)")
    print(f"{'='*72}")

    pass_start = time.time()
    results = []

    # Per-category accumulators
    cat_simple_faith   = defaultdict(list)
    cat_enhanced_faith = defaultdict(list)
    cat_cache_hits     = defaultdict(int)
    cat_corrections    = defaultdict(int)

    s_faith_total = 0
    e_faith_total = 0
    e_correct_total = 0
    s_conf_total = 0.0
    e_conf_total = 0.0
    s_time_total = 0.0
    e_time_total = 0.0

    for i, (category, q) in enumerate(questions):
        print(f"\n  Q{i+1:02d}/{len(questions)} [{category}]")
        print(f"  {q}")

        entry = {"pass": pass_num, "category": category, "question": q}

        # ── Simple RAG ───────────────────────────────────────────────
        t_s = time.time()
        s_ans, s_mids = simple_agent.generate_response(q, decay_enabled=False)
        s_ctx = "\n".join(
            simple_agent.memory.memories[m].content
            for m in s_mids if m in simple_agent.memory.memories
        )
        s_faith = judge.check_faithfulness(q, s_ans, s_ctx)
        s_elapsed = time.time() - t_s
        s_time_total += s_elapsed

        entry.update({
            "simple_answer":     s_ans,
            "simple_faithful":   s_faith["faithful"],
            "simple_confidence": s_faith["confidence"],
            "simple_reason":     s_faith["reason"],
            "simple_time_s":     round(s_elapsed, 2),
        })

        if s_faith["faithful"]:
            s_faith_total += 1
        s_conf_total += s_faith["confidence"]
        cat_simple_faith[category].append(s_faith["faithful"])

        print(f"    [Simple]   Faith={s_faith['faithful']}  Conf={s_faith['confidence']:.2f}  ({s_elapsed:.1f}s)")

        # ── Enhanced RAG (FDL) ───────────────────────────────────────
        t_e = time.time()
        fdl = fdl_engine.ask(q)
        e_elapsed = time.time() - t_e
        e_time_total += e_elapsed

        cache_tag = " [CACHE HIT]" if fdl.get("cache_hit") else ""
        entry.update({
            "enhanced_s1_answer":  fdl["system1_answer"],
            "enhanced_s2_answer":  fdl["system2_answer"],
            "enhanced_final":      fdl["final_answer"],
            "enhanced_faithful":   fdl["faithful"],
            "enhanced_confidence": fdl["confidence"],
            "enhanced_corrected":  fdl["self_corrected"],
            "enhanced_reason":     fdl["reason"],
            "enhanced_time_s":     round(e_elapsed, 2),
            "cache_hit":           fdl.get("cache_hit", False),
            "cache_similarity":    fdl.get("cache_similarity"),
        })

        if fdl["faithful"]:
            e_faith_total += 1
        if fdl["self_corrected"]:
            e_correct_total += 1
            cat_corrections[category] += 1
        if fdl.get("cache_hit"):
            cat_cache_hits[category] += 1

        e_conf_total += fdl["confidence"]
        cat_enhanced_faith[category].append(fdl["faithful"])

        print(f"    [Enhanced] Faith={fdl['faithful']}  Conf={fdl['confidence']:.2f}  "
              f"Corrected={fdl['self_corrected']}  ({e_elapsed:.1f}s){cache_tag}")

        results.append(entry)
        time.sleep(0.2)

    # ── Aggregate metrics ─────────────────────────────────────────────
    n = len(questions)
    pass_elapsed = time.time() - pass_start
    metrics = {
        "pass":                       pass_num,
        "n_questions":                n,
        "simple_faithfulness_rate":   round(s_faith_total / n, 3),
        "enhanced_faithfulness_rate": round(e_faith_total / n, 3),
        "simple_avg_confidence":      round(s_conf_total / n, 3),
        "enhanced_avg_confidence":    round(e_conf_total / n, 3),
        "enhanced_self_corrections":  e_correct_total,
        "enhanced_correction_rate":   round(e_correct_total / n, 3),
        "simple_avg_time_s":          round(s_time_total / n, 2),
        "enhanced_avg_time_s":        round(e_time_total / n, 2),
        "pass_total_time_s":          round(pass_elapsed, 1),
    }

    if cache is not None:
        cs = cache.stats()
        metrics["cache_hits"]     = cs["hits"]
        metrics["cache_misses"]   = cs["misses"]
        metrics["cache_hit_rate"] = cs["hit_rate"]
        metrics["cache_size"]     = cs["size"]

    # ── Print pass summary ────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  PASS {pass_num} METRICS  ({pass_elapsed:.0f}s total for {n} questions)")
    print(f"{'='*72}")
    print(f"  Simple RAG   — Faith: {metrics['simple_faithfulness_rate']:.1%}  "
          f"AvgConf: {metrics['simple_avg_confidence']:.2f}  "
          f"AvgTime: {metrics['simple_avg_time_s']:.1f}s")
    print(f"  Enhanced RAG — Faith: {metrics['enhanced_faithfulness_rate']:.1%}  "
          f"AvgConf: {metrics['enhanced_avg_confidence']:.2f}  "
          f"AvgTime: {metrics['enhanced_avg_time_s']:.1f}s  "
          f"Corrections: {e_correct_total}/{n}")
    if cache is not None:
        print(f"  Cache        — Hits: {cs['hits']}/{cs['hits']+cs['misses']}  "
              f"HitRate: {cs['hit_rate']:.1%}  Stored: {cs['size']}")

    # ── Per-category breakdown ────────────────────────────────────────
    all_cats = sorted(set(cat_simple_faith) | set(cat_enhanced_faith))
    if all_cats:
        print(f"\n  {'Category':<28} {'#Q':>3}  {'Simple':>8}  {'Enhanced':>9}  {'Cache':>6}  {'Fixes':>5}")
        print(f"  {'-'*68}")
        for cat in all_cats:
            s_lst = cat_simple_faith.get(cat, [])
            e_lst = cat_enhanced_faith.get(cat, [])
            nq    = max(len(s_lst), len(e_lst))
            sf    = f"{sum(s_lst)/len(s_lst):.0%}" if s_lst else "—"
            ef    = f"{sum(e_lst)/len(e_lst):.0%}" if e_lst else "—"
            ch    = cat_cache_hits.get(cat, 0)
            cr    = cat_corrections.get(cat, 0)
            print(f"  {cat:<28} {nq:>3}  {sf:>8}  {ef:>9}  {ch:>6}  {cr:>5}")

    # ── Save ─────────────────────────────────────────────────────────
    fname = f"results_comprehensive_pass_{pass_num}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved → {fname}")

    return metrics, results

# ─────────────────────────────────────────────────────────────────────────────
# Final comparison table
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(m1, m2, m3, total_elapsed):
    keys = [
        "simple_faithfulness_rate",
        "enhanced_faithfulness_rate",
        "simple_avg_confidence",
        "enhanced_avg_confidence",
        "enhanced_correction_rate",
        "simple_avg_time_s",
        "enhanced_avg_time_s",
        "cache_hit_rate",
    ]
    print(f"\n{'='*72}")
    print(f"  3-PASS FINAL COMPARISON  (total benchmark: {total_elapsed:.0f}s)")
    print(f"{'='*72}")
    print(f"  {'Metric':<34} {'Pass 1':>9} {'Pass 2':>9} {'Pass 3':>9} {'P1→P3':>8}")
    print(f"  {'-'*72}")
    for key in keys:
        v1 = m1.get(key, 0)
        v2 = m2.get(key, 0)
        v3 = m3.get(key, 0)
        delta = v3 - v1
        sign = "+" if delta >= 0 else ""
        print(f"  {key:<34} {v1:>9.3f} {v2:>9.3f} {v3:>9.3f} {sign}{delta:>7.3f}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    total_start = time.time()
    simple_agent, enhanced_agent, judge, fdl_engine, enhanced_mem, cache = setup()

    # ── Pass 1: Cold start — factual + inference ──────────────────────
    m1, _ = run_pass(1, PASS_1_QUESTIONS, simple_agent, fdl_engine, judge, cache=cache)

    # ── 1-day decay (light) ───────────────────────────────────────────
    print("\n" + "*"*72)
    print("  Simulating 1 day of memory decay (light pruning)...")
    enhanced_mem.advance_time(days=1.0)
    pruned = enhanced_mem.prune()
    print(f"  Pruned {pruned} memories. Enhanced memory: {enhanced_mem.get_memory_size_kb():.1f} KB")
    print("*"*72)

    # ── Pass 2: Near-paraphrase — cache stress-test ───────────────────
    m2, _ = run_pass(2, PASS_2_QUESTIONS, simple_agent, fdl_engine, judge, cache=cache)

    # ── 3-day decay (aggressive) ──────────────────────────────────────
    print("\n" + "*"*72)
    print("  Simulating 3 days of memory decay (aggressive pruning)...")
    enhanced_mem.advance_time(days=3.0)
    pruned = enhanced_mem.prune()
    print(f"  Pruned {pruned} memories. Enhanced memory: {enhanced_mem.get_memory_size_kb():.1f} KB")
    print("*"*72)

    # ── Pass 3: Post-decay — multi-hop + out-of-scope + remaining ─────
    m3, _ = run_pass(3, PASS_3_QUESTIONS, simple_agent, fdl_engine, judge, cache=cache)

    # ── Final report ─────────────────────────────────────────────────
    print_comparison(m1, m2, m3, time.time() - total_start)

    summary = {"pass_1": m1, "pass_2": m2, "pass_3": m3}
    with open("results_comprehensive_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("  Full summary saved → results_comprehensive_summary.json")
    print("Done.")


if __name__ == "__main__":
    main()
