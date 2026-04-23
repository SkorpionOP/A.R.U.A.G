import os
import pickle
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Core Imports
from src.utils.pdf_processor import extract_text_from_pdf, chunk_text
from src.rag.embedder import Embedder
from src.memory.memory import EnhancedMemory
from src.rag.agent import OllamaRAGAgent
from src.eval.evaluator import LLMJudge
from src.memory.fdl_engine import FDLEngine
from src.rag.graph_rag import KnowledgeGraph
from src.rag.semantic_cache import SemanticCache

@dataclass
class ExtensionConfig:
    """Configuration settings for the RAG Memory Extension."""
    llm_model: str = "qwen2.5:1.5b"
    judge_model: str = "qwen2.5:1.5b"
    chunk_size: int = 250
    chunk_overlap: int = 30
    cache_similarity_threshold: float = 0.92
    cache_ttl_seconds: float = 3600.0
    memory_importance: float = 0.8
    memory_category: str = "learned_fact"

class RAGMemoryExtension:
    """
    A drop-in extension wrapper that makes the Enhanced RAG system accessible everywhere.
    It encapsulates the Embedder, Memory, GraphRAG, Semantic Cache, Agent, and FDL Engine.
    """
    def __init__(self, config: Optional[ExtensionConfig] = None):
        self.config = config or ExtensionConfig()
        
        # 1. Initialize Sub-systems
        self.embedder = Embedder()
        self.memory = EnhancedMemory(embed_func=self.embedder.embed)
        self.graph = KnowledgeGraph()
        
        self.cache = SemanticCache(
            embed_func=self.embedder.embed,
            similarity_threshold=self.config.cache_similarity_threshold,
            ttl_seconds=self.config.cache_ttl_seconds
        )
        
        # 2. Initialize Agents
        self.agent = OllamaRAGAgent(
            memory_store=self.memory, 
            name="EnhancedRAG", 
            model=self.config.llm_model, 
            graph=self.graph
        )
        
        self.judge = LLMJudge(model=self.config.judge_model)
        
        # 3. Initialize Failure-Driven Learning Pipeline
        self.fdl_engine = FDLEngine(
            agent=self.agent, 
            judge=self.judge, 
            cache=self.cache
        )

    def ingest_pdfs(self, pdf_paths: List[str]) -> int:
        """Process and ingest multiple PDFs into the memory and graph systems."""
        all_chunks = []
        for path in pdf_paths:
            if not os.path.exists(path):
                print(f"Warning: File not found -> {path}")
                continue
                
            with open(path, "rb") as f:
                text = extract_text_from_pdf(f)
            chunks = chunk_text(text, chunk_size=self.config.chunk_size, overlap=self.config.chunk_overlap)
            all_chunks.extend(chunks)
            
        if not all_chunks:
            return 0
            
        # Fit embedder (no-op for pre-trained dense models, but good practice)
        self.embedder.fit(all_chunks)
        embeddings = self.embedder.embed_batch(all_chunks)
        
        # Store in Memory and Graph
        ids = self.memory.store_batch(
            all_chunks, 
            embeddings, 
            importance=self.config.memory_importance, 
            category=self.config.memory_category
        )
        self.graph.build(ids, all_chunks)
        
        return len(all_chunks)

    def ask(self, query: str) -> Dict[str, Any]:
        """Ask a question and route it through the FDL-enabled RAG pipeline."""
        return self.fdl_engine.ask(query)

    def simulate_decay(self, days: float) -> int:
        """Simulates time passing to trigger Ebbinghaus decay and prune weak memories."""
        self.memory.advance_time(days=days)
        return self.memory.prune()

    def get_stats(self) -> Dict[str, Any]:
        """Retrieve memory, cache, and graph statistics."""
        return {
            "memory_kb": self.memory.get_memory_size_kb(),
            "cache_stats": self.cache.stats(),
            "graph_stats": self.graph.stats() if self.graph._built else None
        }

    def save(self, filepath: str = "brain.pkl"):
        """Saves the entire brain (memory, graph, and cache) to your hard drive so it remembers after a restart."""
        state = {
            "memories": self.memory.memories,
            "query_outcomes": self.memory._query_outcomes,
            "failure_note_ids": self.memory._failure_note_ids,
            "cache_entries": self.cache._entries,
            "graph_nodes": self.graph.nodes if self.graph._built else None,
            "graph_edges": self.graph.edges if self.graph._built else None
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        print(f"Brain successfully saved to {filepath}")

    def load(self, filepath: str = "brain.pkl") -> bool:
        """Loads a previously saved brain from your hard drive."""
        if not os.path.exists(filepath):
            print(f"No saved brain found at {filepath}. Starting fresh.")
            return False
            
        with open(filepath, "rb") as f:
            state = pickle.load(f)
            
        self.memory.memories = state["memories"]
        self.memory._query_outcomes = state["query_outcomes"]
        self.memory._failure_note_ids = state["failure_note_ids"]
        self.cache._entries = state["cache_entries"]
        
        if state["graph_nodes"] is not None:
            self.graph.nodes = state["graph_nodes"]
            self.graph.edges = state["graph_edges"]
            self.graph._built = True
            
        print(f"Brain successfully loaded from {filepath}")
        return True

# Example usage string if someone runs this directly
if __name__ == "__main__":
    print("RAGMemoryExtension is ready to be imported and used.")
    print("Example usage:")
    print("  from src.extension import RAGMemoryExtension, ExtensionConfig")
    print("  ext = RAGMemoryExtension(ExtensionConfig(llm_model='llama3'))")
    print("  ext.ingest_pdfs(['my_document.pdf'])")
    print("  response = ext.ask('What is the document about?')")
