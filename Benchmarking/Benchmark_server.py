"""
benchmark_server.py — WebSocket server for RAG Benchmark UI
============================================================
Bridges the HTML dashboard with the actual Python benchmark,
sending real-time progress updates over WebSocket.

Run: python benchmark_server.py
Then open: http://localhost:8765
"""

import asyncio
import json
import time
import sys
from pathlib import Path
from datetime import datetime

# Assuming these are your actual modules:
# from pdf_processor import extract_text_from_pdf, chunk_text
# from embedder import Embedder
# from memory import SimpleRAGMemory, EnhancedMemory
# from agent import OllamaRAGAgent
# from evaluator import LLMJudge
# from fdl_engine import FDLEngine

try:
    import websockets
    from websockets.server import serve
except ImportError:
    print("Installing websockets...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets"])
    import websockets
    from websockets.server import serve

QUESTIONS = [
    "When did the Constitution of India come into effect?",
    "Who is known as the Father of the Indian Constitution?",
    "How many Fundamental Rights are currently recognized in India?",
    "Which part of the Constitution deals with Fundamental Rights?",
    "What is the minimum voting age in India?",
    "Which article of the Constitution abolishes untouchability?",
    "What is the maximum strength of the Lok Sabha?",
    "Which body is known as the guardian of the Constitution?",
    "What type of federal system does India follow?",
    "Which amendment added Socialist and Secular to the Preamble?",
    "Which article allows the President to declare a National Emergency?",
    "What is the difference between Article 32 and Article 226?",
    "Which schedule deals with anti-defection laws?",
    "Can Fundamental Rights be suspended during Emergency?",
    "Who has the power to amend the Constitution?",
    "Why is Article 32 called heart and soul?",
    "What is the Basic Structure Doctrine?",
    "Difference between Fundamental Rights and Directive Principles?",
    "What is a Money Bill?",
    "Why does India have a single citizenship system?",
]

# Global state
benchmark_state = {
    "running": False,
    "pass": 0,
    "question": 0,
    "results": {"pass1": None, "pass2": None},
    "start_time": None,
}

clients = set()


async def broadcast(message):
    """Send message to all connected clients."""
    if clients:
        await asyncio.gather(*[client.send(json.dumps(message)) for client in clients], return_exceptions=True)


async def setup_benchmark(config):
    """
    Initialize RAG components.
    In production, this would load actual PDF and create embeddings.
    """
    print("[Setup] Initializing benchmark...")
    await broadcast({"type": "status", "message": "Loading PDF and creating embeddings...", "status": "info"})
    
    # Simulate setup delay
    await asyncio.sleep(1.5)
    
    await broadcast({"type": "status", "message": "Setup complete!", "status": "success"})
    print("[Setup] Complete")
    
    # In production:
    # with open(config['pdf_path'], 'rb') as f:
    #     text = extract_text_from_pdf(f)
    # chunks = chunk_text(text, config['chunk_size'], 30)
    # embedder = Embedder()
    # embedder.fit(chunks)
    # embeddings = embedder.embed_batch(chunks)
    # 
    # simple_mem = SimpleRAGMemory(embed_func=embedder.embed)
    # enhanced_mem = EnhancedMemory(embed_func=embedder.embed)
    # simple_mem.store_batch(chunks, embeddings, 0.8, "learned_fact")
    # enhanced_mem.store_batch(chunks, embeddings, 0.8, "learned_fact")
    # 
    # simple_agent = OllamaRAGAgent(simple_mem, "SimpleRAG", model=config['model'])
    # enhanced_agent = OllamaRAGAgent(enhanced_mem, "EnhancedRAG", model=config['model'])
    # judge = LLMJudge(model=config['model'])
    # fdl_engine = FDLEngine(enhanced_agent, judge)
    
    return None  # Return actual agents in production


async def simulate_question(q_num, question):
    """Simulate a single question evaluation."""
    import random
    
    # Add realistic delay
    await asyncio.sleep(0.5 + random.random() * 1.5)
    
    simple_faithful = random.random() > 0.35
    enhanced_faithful = random.random() > 0.15
    
    result = {
        "q_num": q_num,
        "question": question,
        "simple_answer": f"Sample answer for '{question}'",
        "simple_faithful": simple_faithful,
        "simple_confidence": round(0.65 + random.random() * 0.3, 3),
        "simple_time_s": round(1.2 + random.random() * 2, 2),
        "enhanced_s1_answer": f"Quick response for '{question}'",
        "enhanced_s2_answer": f"Deliberative response for '{question}'",
        "enhanced_final": f"Final answer for '{question}'",
        "enhanced_faithful": enhanced_faithful,
        "enhanced_confidence": round(0.75 + random.random() * 0.2, 3),
        "enhanced_corrected": enhanced_faithful and not simple_faithful,
        "enhanced_time_s": round(2.5 + random.random() * 3, 2),
    }
    
    return result


async def run_pass(pass_num, agents=None):
    """Run a single pass through all questions."""
    print(f"\n[Pass {pass_num}] Starting...")
    benchmark_state["pass"] = pass_num
    benchmark_state["question"] = 0
    
    results = []
    metrics = {
        "pass": pass_num,
        "simple_faithfulness_rate": 0.0,
        "enhanced_faithfulness_rate": 0.0,
        "simple_avg_confidence": 0.0,
        "enhanced_avg_confidence": 0.0,
        "enhanced_self_corrections": 0,
        "enhanced_correction_rate": 0.0,
        "simple_total_time_s": 0.0,
        "enhanced_total_time_s": 0.0,
        "simple_avg_time_s": 0.0,
        "enhanced_avg_time_s": 0.0,
    }
    
    for i, question in enumerate(QUESTIONS):
        if not benchmark_state["running"]:
            break
            
        benchmark_state["question"] = i + 1
        
        # Get question result
        result = await simulate_question(i + 1, question)
        results.append(result)
        
        # Update metrics
        if result["simple_faithful"]:
            metrics["simple_faithfulness_rate"] += 1
        if result["enhanced_faithful"]:
            metrics["enhanced_faithfulness_rate"] += 1
        
        metrics["simple_avg_confidence"] += result["simple_confidence"]
        metrics["enhanced_avg_confidence"] += result["enhanced_confidence"]
        metrics["simple_total_time_s"] += result["simple_time_s"]
        metrics["enhanced_total_time_s"] += result["enhanced_time_s"]
        
        if result["enhanced_corrected"]:
            metrics["enhanced_self_corrections"] += 1
        
        # Send progress update
        await broadcast({
            "type": "progress",
            "pass": pass_num,
            "current": i + 1,
            "total": len(QUESTIONS),
            "current_question": question,
        })
        
        # Send question result
        await broadcast({
            "type": "question_result",
            "pass": pass_num,
            "result": result,
        })
        
        print(f"[Pass {pass_num}] Q{i+1}/{len(QUESTIONS)}: {question[:50]}...")
    
    # Finalize metrics
    n = len(QUESTIONS)
    if n > 0:
        metrics["simple_faithfulness_rate"] = round(metrics["simple_faithfulness_rate"] / n, 3)
        metrics["enhanced_faithfulness_rate"] = round(metrics["enhanced_faithfulness_rate"] / n, 3)
        metrics["simple_avg_confidence"] = round(metrics["simple_avg_confidence"] / n, 3)
        metrics["enhanced_avg_confidence"] = round(metrics["enhanced_avg_confidence"] / n, 3)
        metrics["enhanced_correction_rate"] = round(metrics["enhanced_self_corrections"] / n, 3)
        metrics["simple_avg_time_s"] = round(metrics["simple_total_time_s"] / n, 2)
        metrics["enhanced_avg_time_s"] = round(metrics["enhanced_total_time_s"] / n, 2)
    
    benchmark_state["results"][f"pass{pass_num}"] = {
        "metrics": metrics,
        "results": results
    }
    
    await broadcast({
        "type": "pass_complete",
        "pass": pass_num,
        "metrics": metrics,
    })
    
    print(f"[Pass {pass_num}] Complete\n")
    return metrics


async def run_benchmark(config):
    """Run the full benchmark: Pass 1, decay, Pass 2."""
    benchmark_state["running"] = True
    benchmark_state["start_time"] = time.time()
    
    try:
        # Setup
        agents = await setup_benchmark(config)
        
        # Pass 1
        await broadcast({"type": "status", "message": "Running Pass 1...", "status": "info"})
        m1 = await run_pass(1, agents)
        
        if not benchmark_state["running"]:
            return
        
        # Simulate memory decay
        decay_days = config.get("decay_days", 3)
        await broadcast({
            "type": "status",
            "message": f"Simulating {decay_days} days of memory decay and pruning...",
            "status": "info"
        })
        await asyncio.sleep(2.0)
        
        # Pass 2
        await broadcast({"type": "status", "message": "Running Pass 2...", "status": "info"})
        m2 = await run_pass(2, agents)
        
        if benchmark_state["running"]:
            # Send final comparison
            await broadcast({
                "type": "benchmark_complete",
                "metrics_pass1": m1,
                "metrics_pass2": m2,
            })
            await broadcast({
                "type": "status",
                "message": "Benchmark complete!",
                "status": "success"
            })
            print("[Benchmark] All passes complete!")
        
    except Exception as e:
        print(f"[Error] {str(e)}")
        await broadcast({
            "type": "error",
            "message": str(e),
        })
    finally:
        benchmark_state["running"] = False


async def handle_client(websocket, path):
    """Handle WebSocket client connection."""
    clients.add(websocket)
    print(f"[Client] Connected: {websocket.remote_address}")
    
    try:
        async for message in websocket:
            data = json.loads(message)
            command = data.get("command")
            
            if command == "start":
                config = data.get("config", {})
                if not benchmark_state["running"]:
                    asyncio.create_task(run_benchmark(config))
                    print("[Benchmark] Started")
            
            elif command == "stop":
                benchmark_state["running"] = False
                print("[Benchmark] Stopped")
                await broadcast({
                    "type": "status",
                    "message": "Benchmark stopped by user",
                    "status": "info"
                })
            
            elif command == "get_state":
                await websocket.send(json.dumps({
                    "type": "state",
                    "state": benchmark_state,
                }))
    
    except websockets.exceptions.ConnectionClosed:
        print(f"[Client] Disconnected: {websocket.remote_address}")
    finally:
        clients.discard(websocket)


async def main():
    """Start the WebSocket server."""
    print("="*70)
    print("  RAG Benchmark Server")
    print("="*70)
    print("\nStarting WebSocket server on ws://localhost:8765")
    print("Serve the HTML file via HTTP and connect to this server.\n")
    print("Python modules to implement:")
    print("  - pdf_processor (extract_text_from_pdf, chunk_text)")
    print("  - embedder (Embedder class)")
    print("  - memory (SimpleRAGMemory, EnhancedMemory)")
    print("  - agent (OllamaRAGAgent)")
    print("  - evaluator (LLMJudge)")
    print("  - fdl_engine (FDLEngine)")
    print("\n" + "="*70 + "\n")
    
    async with serve(handle_client, "localhost", 8765):
        print("Server running. Press Ctrl+C to stop.")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped.")