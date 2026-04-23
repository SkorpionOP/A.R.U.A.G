# Enhanced RAG with Failure-Driven Learning (FDL)

This repository contains an autonomous memory system that extends standard Retrieval-Augmented Generation (RAG) by giving it a "cognitive stack". It features Ebbinghaus decay, Semantic Caching, GraphRAG, and Failure-Driven Learning (FDL) to continuously improve from its own interactions without needing human-labeled ground truth.

## 🚀 Features

- **Failure-Driven Learning (FDL):** Uses an LLM-as-a-judge to detect failures and correct itself, updating memory so it doesn't repeat mistakes.
- **Ebbinghaus Memory Decay:** Memories fade naturally if not used, and reinforce themselves if proven useful. This prevents memory bloat.
- **Semantic Caching:** Sub-millisecond response times for near-paraphrased queries.
- **GraphRAG + Vector RAG:** A unified memory system that performs standard semantic similarity alongside graph-based reasoning.
- **Extension Wrapper API:** The entire pipeline is bundled into an easy-to-use drop-in extension.

## 📂 Project Structure

```text
.
├── app.py                # Streamlit Playground App
├── src/                  # Core Library
│   ├── eval/             # LLM as a Judge Evaluation System
│   ├── memory/           # Memory Models & FDL Engine
│   ├── rag/              # Embedder, Agent, Semantic Cache, GraphRAG
│   └── utils/            # Document parsing tools
├── tests/                # Benchmarking and QA scripts
├── docs/                 # Research papers, schemas, executive summaries
├── results/              # JSON output files from benchmark runs
└── setup.py              # Packaging configuration
```

## 🛠 Prerequisites

Before running the project, your friend will need:
1. **Python 3.8+** installed on their system.
2. **[Ollama](https://ollama.com/)** installed and running locally. The system uses local LLMs to ensure privacy and avoid API costs.
   - Once Ollama is installed, they need to pull the default model by running this in their terminal:
     ```bash
     ollama run qwen2.5:1.5b
     ```

## 🛠 Installation

Once the prerequisites are met, they can clone the repository and install it as a Python package.

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
pip install -e .
```

## 📖 Quickstart

Using the `RAGMemoryExtension` wrapper makes it incredibly easy to embed this pipeline in any Python application.

```python
from src.extension import RAGMemoryExtension, ExtensionConfig

# 1. Initialize the pipeline
config = ExtensionConfig(llm_model="qwen2.5:1.5b")
memory_ext = RAGMemoryExtension(config)

# 2. Ingest documents
memory_ext.ingest_pdfs(["test_document/10_English_Textbook_2024-25.pdf"])

# 3. Ask a question!
response = memory_ext.ask("What did Lencho ask for in his letter to God?")

# The response contains the answer and evaluation metrics
print(response["final_answer"])
```

### Advanced Memory Management

The extension includes built-in features to easily save, load, monitor, and clear the AI's memory without touching any databases.

You can set the system to **automatically save and load** its brain by simply turning on `auto_save` in the config:

```python
config = ExtensionConfig(
    llm_model="qwen2.5:1.5b",
    auto_save=True,               # <--- Turns on automatic saving/loading
    save_filepath="my_brain.pkl"  # <--- Where to save the brain
)
memory_ext = RAGMemoryExtension(config)

# Now, every time you ingest a PDF or ask a question, it automatically updates the file!
memory_ext.ingest_pdfs(["test_document/biology.pdf"])
```

You can also trigger these actions manually:

```python
# Print a beautiful summary of ingested PDFs, cache size, and memory usage
memory_ext.summary()

# Completely wipe the memory, cache, graph, and document history
memory_ext.clear()
```

### Try the Playground

We have a Streamlit app to interactively test the basic RAG vs Enhanced RAG side by side:

```bash
streamlit run app.py
```

## 📝 Documentation

Check the `docs/` folder for a technical deep dive into how the system achieves Ebbinghaus decay, FDL, and Semantic Caching:
- `docs/ARCHITECTURE.md`
