import streamlit as st
from src.rag.embedder import Embedder
from src.utils.pdf_processor import extract_text_from_pdf, chunk_text
from src.memory.memory import SimpleRAGMemory, EnhancedMemory, Interaction
from src.rag.agent import OllamaRAGAgent
from src.eval.evaluator import LLMJudge
from src.memory.fdl_engine import FDLEngine
from src.rag.graph_rag import KnowledgeGraph
from src.rag.semantic_cache import SemanticCache

st.set_page_config(page_title="RAG vs Enhanced RAG", layout="wide")

st.markdown("""
<style>
.metric-card {
    padding: 12px 16px;
    border-radius: 8px;
    border: 1px solid #333;
    background: #1a1a2e;
    margin-bottom: 8px;
}
.metric-card h4 { margin: 0 0 4px 0; color: #e0e0e0; font-size: 14px; }
.metric-card .value { font-size: 22px; font-weight: bold; }
.pass-tag { font-size: 11px; padding: 2px 6px; border-radius: 4px; }
.faithful { color: #4ecca3; }
.unfaithful { color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

# ── Session State Init ─────────────────────────────────────────────
if "embedder" not in st.session_state:
    st.session_state.embedder = Embedder()
if "simple_memory" not in st.session_state:
    st.session_state.simple_memory = SimpleRAGMemory(embed_func=st.session_state.embedder.embed)
if "enhanced_memory" not in st.session_state:
    st.session_state.enhanced_memory = EnhancedMemory(embed_func=st.session_state.embedder.embed)
if "graph" not in st.session_state:
    st.session_state.graph = KnowledgeGraph()
if "sem_cache" not in st.session_state:
    st.session_state.sem_cache = SemanticCache(
        embed_func=st.session_state.embedder.embed,
        similarity_threshold=0.92,
        ttl_seconds=3600.0,
    )
if "simple_agent" not in st.session_state:
    st.session_state.simple_agent = OllamaRAGAgent(st.session_state.simple_memory, "SimpleRAG", model="qwen2.5:1.5b")
if "enhanced_agent" not in st.session_state:
    st.session_state.enhanced_agent = OllamaRAGAgent(
        st.session_state.enhanced_memory, "EnhancedRAG", model="qwen2.5:1.5b",
        graph=st.session_state.graph)
if "judge" not in st.session_state:
    st.session_state.judge = LLMJudge(model="qwen2.5:1.5b")
if "fdl_engine" not in st.session_state:
    st.session_state.fdl_engine = FDLEngine(
        st.session_state.enhanced_agent, st.session_state.judge,
        cache=st.session_state.sem_cache)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed" not in st.session_state:
    st.session_state.processed = False

st.title("RAG vs Enhanced RAG (FDL + Ebbinghaus Decay)")
st.markdown("Compare standard RAG against an autonomous memory system that **self-corrects without ground truth** using LLM-as-a-Judge, consistency checks, and failure-driven learning.")

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    ollama_model = st.text_input("Ollama Model", value="qwen2.5:1.5b")
    if ollama_model:
        st.session_state.simple_agent.model = ollama_model
        st.session_state.enhanced_agent.model = ollama_model
        st.session_state.judge.model = ollama_model

    st.markdown("---")
    st.header("1. Document Ingestion")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Process PDFs", type="primary"):
        if uploaded_files:
            with st.spinner("Processing..."):
                all_chunks = []
                for file in uploaded_files:
                    text = extract_text_from_pdf(file)
                    chunks = chunk_text(text, chunk_size=250, overlap=30)
                    all_chunks.extend(chunks)

                st.session_state.embedder = Embedder()
                st.session_state.simple_memory = SimpleRAGMemory(embed_func=st.session_state.embedder.embed)
                st.session_state.enhanced_memory = EnhancedMemory(embed_func=st.session_state.embedder.embed)
                st.session_state.graph = KnowledgeGraph()
                st.session_state.sem_cache = SemanticCache(
                    embed_func=st.session_state.embedder.embed,
                    similarity_threshold=0.92, ttl_seconds=3600.0)
                st.session_state.simple_agent = OllamaRAGAgent(
                    st.session_state.simple_memory, "SimpleRAG", model=ollama_model)
                st.session_state.enhanced_agent = OllamaRAGAgent(
                    st.session_state.enhanced_memory, "EnhancedRAG", model=ollama_model,
                    graph=st.session_state.graph)
                st.session_state.judge = LLMJudge(model=ollama_model)
                st.session_state.fdl_engine = FDLEngine(
                    st.session_state.enhanced_agent, st.session_state.judge,
                    cache=st.session_state.sem_cache)

                st.session_state.embedder.fit(all_chunks)  # no-op for dense model
                embeddings = st.session_state.embedder.embed_batch(all_chunks)
                simple_ids   = st.session_state.simple_memory.store_batch(all_chunks, embeddings, importance=0.8, category="learned_fact")
                enhanced_ids = st.session_state.enhanced_memory.store_batch(all_chunks, embeddings, importance=0.8, category="learned_fact")
                st.session_state.graph.build(enhanced_ids, all_chunks)

                st.session_state.processed = True
                st.session_state.messages = []
                st.success(f"Stored {len(all_chunks)} chunks into both memory systems!")
        else:
            st.warning("Upload a PDF first.")

    st.markdown("---")
    st.header("2. Memory Management")
    if st.button("Advance Time + Prune (3 Days)"):
        st.session_state.enhanced_memory.advance_time(days=3.0)
        pruned = st.session_state.enhanced_memory.prune()
        st.success(f"Pruned {pruned} memories.")
        st.rerun()

    st.markdown("### Memory Stats")
    simple_kb = st.session_state.simple_memory.get_memory_size_kb()
    enh_kb = st.session_state.enhanced_memory.get_memory_size_kb()
    st.metric("Simple RAG", f"{simple_kb:.1f} KB")
    st.metric("Enhanced RAG", f"{enh_kb:.1f} KB")
    if simple_kb > 0:
        reduction = (1 - enh_kb / simple_kb) * 100
        st.metric("Memory Reduction", f"{reduction:.1f}%")

    st.markdown("### Semantic Cache")
    cs = st.session_state.sem_cache.stats()
    st.metric("Cache Entries", cs["size"])
    st.metric("Hit Rate", f"{cs['hit_rate']:.1%}")
    col_h, col_m = st.columns(2)
    col_h.metric("Hits", cs["hits"])
    col_m.metric("Misses", cs["misses"])

    if st.session_state.graph._built:
        g = st.session_state.graph.stats()
        st.markdown("### Knowledge Graph")
        st.metric("Graph Nodes", g["nodes"])
        st.metric("Approx Edges", g["approx_edges"])
        st.metric("Entities Indexed", g["discriminative_ents"])


# ── Main Chat Area ─────────────────────────────────────────────────
if not st.session_state.processed:
    st.info("Upload PDF documents in the sidebar to begin.")
else:
    st.markdown("### Interactive QA")

    # Render chat history
    for val in st.session_state.messages:
        if val["role"] == "user":
            with st.chat_message("user"):
                st.write(val["content"])
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Simple RAG** (no self-correction)")
                st.write(val["simple_answer"])
                faith = val.get("simple_faithful", False)
                conf = val.get("simple_confidence", 0)
                color = "faithful" if faith else "unfaithful"
                st.markdown(f"<span class='{color}'>Faithful: {faith} | Confidence: {conf:.2f}</span>", unsafe_allow_html=True)

            with col2:
                st.markdown("**Enhanced RAG** (FDL Self-Correction)")
                if val.get("self_corrected"):
                    st.caption("Self-corrected via System-2 Deep Search")
                st.write(val["enhanced_final"])
                faith = val.get("enhanced_faithful", False)
                conf = val.get("enhanced_confidence", 0)
                color = "faithful" if faith else "unfaithful"
                st.markdown(f"<span class='{color}'>Faithful: {faith} | Confidence: {conf:.2f}</span>", unsafe_allow_html=True)

    # Chat input
    if query := st.chat_input("Ask a question about the document..."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("Simple RAG answering..."):
            s_ans, s_mids = st.session_state.simple_agent.generate_response(query, decay_enabled=False)
            s_ctx = "\n".join(
                st.session_state.simple_memory.memories[mid].content
                for mid in s_mids
                if mid in st.session_state.simple_memory.memories
            )
            s_faith = st.session_state.judge.check_faithfulness(query, s_ans, s_ctx)

        with st.spinner("Enhanced RAG (FDL pipeline)..."):
            fdl_result = st.session_state.fdl_engine.ask(query)

        st.session_state.messages.append({
            "role": "assistant",
            "query": query,
            "simple_answer": s_ans,
            "simple_faithful": s_faith["faithful"],
            "simple_confidence": s_faith["confidence"],
            "enhanced_final": fdl_result["final_answer"],
            "enhanced_faithful": fdl_result["faithful"],
            "enhanced_confidence": fdl_result["confidence"],
            "self_corrected": fdl_result["self_corrected"],
            "enhanced_reason": fdl_result["reason"],
        })
        st.rerun()
