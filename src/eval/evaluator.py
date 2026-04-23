"""
evaluator.py — LLM-as-a-Judge Self-Evaluation Module
=====================================================
Uses the local LLM to evaluate its OWN answers against retrieved context,
enabling fully autonomous improvement without ground truth.

Based on "AI Memory and Failure Learning" paper:
  - LLM-as-a-Judge layer to validate retrieval relevance
  - Consistency checking (ask the same question multiple ways)
  - Confidence scoring using the document itself as source of truth
"""

import requests
from typing import Optional


class LLMJudge:
    """
    Uses the local Ollama LLM to:
      1. Score whether an answer is actually supported by the retrieved context
      2. Detect hallucination vs. grounded answers
      3. Extract the correct answer from context when hallucination is detected
    
    This enables self-correction WITHOUT ground truth.
    """

    def __init__(self, model: str = "qwen2.5:1.5b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def _call_llm(self, prompt: str, timeout: int = 20) -> Optional[str]:
        try:
            res = requests.post(f"{self.base_url}/api/generate", json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }, timeout=timeout)
            if res.status_code == 200:
                return res.json().get("response", "").strip()
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # 1. Faithfulness Check: Is the answer grounded in the context?
    # ------------------------------------------------------------------
    def check_faithfulness(self, question: str, answer: str, context: str) -> dict:
        """
        Ask the LLM: 'Given ONLY this context, is the answer factually supported?'
        Returns {"faithful": bool, "confidence": float, "reason": str}
        """
        prompt = f"""You are a strict fact-checker. You must ONLY use the context below.

Context:
{context}

Question: {question}
Answer given: {answer}

Is the answer factually supported by the context above? 
Reply in this exact format (nothing else):
VERDICT: YES or NO
CONFIDENCE: a number from 0.0 to 1.0
REASON: one short sentence"""

        raw = self._call_llm(prompt)
        if not raw:
            return {"faithful": False, "confidence": 0.0, "reason": "LLM unreachable"}

        lines = raw.strip().split("\n")
        verdict = False
        confidence = 0.0
        reason = raw

        for line in lines:
            lu = line.upper().strip()
            if "VERDICT" in lu:
                verdict = "YES" in lu
            elif "CONFIDENCE" in lu:
                try:
                    confidence = float("".join(c for c in lu.split(":")[-1] if c in "0123456789."))
                except:
                    confidence = 0.5
            elif "REASON" in lu:
                reason = line.split(":", 1)[-1].strip()

        return {"faithful": verdict, "confidence": confidence, "reason": reason}

    # ------------------------------------------------------------------
    # 2. Consistency Check: Ask the question a different way
    # ------------------------------------------------------------------
    def rephrase_question(self, question: str) -> Optional[str]:
        """Generate an alternative phrasing of the question."""
        prompt = f"""Rephrase this question in a completely different way. 
Output ONLY the rephrased question, nothing else.

Original: {question}
Rephrased:"""
        return self._call_llm(prompt, timeout=10)

    def check_consistency(self, question: str, answer1: str, answer2: str) -> dict:
        """
        Compare two answers to semantically similar questions.
        If they contradict, flag as inconsistent → likely hallucination.
        """
        prompt = f"""Do these two answers convey the same factual meaning?

Question: {question}
Answer A: {answer1}
Answer B: {answer2}

Reply in this exact format:
CONSISTENT: YES or NO
EXPLANATION: one short sentence"""

        raw = self._call_llm(prompt)
        if not raw:
            return {"consistent": False, "explanation": "LLM unreachable"}

        consistent = "YES" in raw.upper().split("CONSISTENT")[-1][:20] if "CONSISTENT" in raw.upper() else False
        explanation = raw.split("EXPLANATION")[-1].strip(": \n") if "EXPLANATION" in raw.upper() else raw

        return {"consistent": consistent, "explanation": explanation}

    # ------------------------------------------------------------------
    # 3. Chunk-by-Chunk Extraction (Map-Reduce, no hallucination)
    # ------------------------------------------------------------------
    def extract_from_chunk(self, question: str, chunk: str) -> Optional[str]:
        """
        Ask the LLM to extract the answer from a SINGLE small chunk.
        Returns the answer if found, or None if the chunk doesn't contain it.
        Small context = small model can focus = less hallucination.
        """
        prompt = f"""You are a strict data extractor. Read ONLY the context below.
Does it contain the answer to the question?
If YES, output ONLY the factual answer in ONE SHORT LINE. No explanation.
If NO, output exactly: NOT_FOUND

Context:
{chunk}

Question: {question}
Answer:"""

        ans = self._call_llm(prompt, timeout=15)
        if ans and "NOT_FOUND" not in ans.upper() and len(ans.strip()) > 2:
            return ans.strip()
        return None
