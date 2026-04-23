from src.extension import RAGMemoryExtension, ExtensionConfig

print("--- Simple RAG Memory Example ---")

# 1. Initialize with auto_save enabled
config = ExtensionConfig(
    llm_model="qwen2.5:1.5b", 
    auto_save=True,
    save_filepath="my_first_brain.pkl"
)
ext = RAGMemoryExtension(config)

# 2. Ingest a document
pdf_path = "test_document/10_English_Textbook_2024-25.pdf"
print(f"Ingesting {pdf_path}...")
ext.ingest_pdfs([pdf_path])

# 3. Ask a question
question = "What did Lencho ask for in his letter to God?"
print(f"\nQuestion: {question}")
response = ext.ask(question)
print(f"Answer: {response['final_answer']}")

# 4. View Stats
ext.summary()
