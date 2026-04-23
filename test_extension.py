import os
from src.extension import RAGMemoryExtension, ExtensionConfig

def main():
    print("=== Testing Full RAG Memory Extension API ===\n")
    
    # 1. Start fresh by deleting old brain if it exists
    if os.path.exists("test_brain.pkl"):
        os.remove("test_brain.pkl")

    # 2. Initialize
    config = ExtensionConfig(
        llm_model="qwen2.5:1.5b",
        auto_save=True,
        save_filepath="test_brain.pkl"
    )
    ext = RAGMemoryExtension(config)
    
    # 3. Ingest
    print("\n[1] Ingesting Document...")
    ext.ingest_pdfs(["test_document/10_English_Textbook_2024-25.pdf"])
    
    # 4. Ask a question (System 1 + System 2 potentially)
    print("\n[2] Asking Initial Question (Testing Retrieval & FDL)...")
    q1 = "How many pesos did Lencho ask for?"
    res1 = ext.ask(q1)
    print(f"Answer: {res1['final_answer']}")
    print(f"FDL Self-Corrected: {res1.get('self_corrected', False)}")
    print(f"Reason: {res1.get('reason', '')}")
        
    # 5. Ask similar question to trigger Semantic Cache
    print("\n[3] Asking Paraphrased Question (Testing Semantic Cache)...")
    q2 = "What amount of pesos did Lencho want?"
    res2 = ext.ask(q2)
    print(f"Answer: {res2.get('final_answer', res2.get('answer', ''))}")
    print(f"Cache Hit: {res2.get('cache_hit', False)}")
    
    # 6. Simulate Decay
    print("\n[4] Simulating 10 Days of Time Passing...")
    pruned = ext.simulate_decay(days=10.0)
    print(f"Memories pruned due to decay: {pruned}")
    
    # 7. Print Status
    ext.summary()
    
    # 8. Test Clear
    print("\n[5] Wiping Brain...")
    ext.clear()
    ext.summary()
    print("Test Complete!")

if __name__ == "__main__":
    main()
