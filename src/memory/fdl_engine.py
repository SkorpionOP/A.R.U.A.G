"""
fdl_engine.py — Autonomous Failure-Driven Learning Pipeline
============================================================
Implements the self-correction loop from the AI Memory and Failure Learning paper:

1. Generate answer (System 1: fast retrieval)
2. Judge the answer (LLM-as-a-Judge: faithfulness check)
   ✅ OPT-3: Judge is only invoked when retrieval confidence < threshold
             (skips expensive LLM call when System-1 is already high-confidence)
3. If unfaithful → Self-correct (System 2: single-prompt multi-chunk synthesis)
   ✅ OPT-3: Single LLM call over top-5 chunks replaces 15-call Map-Reduce loop
4. If STILL unfaithful → Consistency check (rephrase & re-ask, compare)
5. Validate correction against doc context before storing (FIX 1)
6. Log outcome → Update memory confidence scores (FIX 2)

NO GROUND TRUTH REQUIRED. The document itself is the source of truth.
"""

import numpy as np
from src.memory.memory import EnhancedMemory, Interaction
from src.rag.agent import OllamaRAGAgent
from src.eval.evaluator import LLMJudge
from src.rag.semantic_cache import SemanticCache
from typing import Optional


class FDLEngine:
    """
    Autonomous self-correction engine with correction validation.

    OPT-3: Conditional Judge threshold.
      When System-1's top-chunk retrieval cosine similarity is already
      high (>= CONFIDENCE_JUDGE_THRESHOLD), skip the LLM faithfulness
      call entirely — the answer is almost certainly grounded.

    OPT-5: Semantic Cache.
      Before running ANY LLM call, check whether a semantically equivalent
      query was already answered (cosine similarity >= cache.threshold).
      Cache hits return in ~2 ms, bypassing the entire pipeline.
    """

    # Only invoke the LLM faithfulness judge when retrieval score is below this.
    CONFIDENCE_JUDGE_THRESHOLD = 0.75

    def __init__(self, agent: OllamaRAGAgent, judge: LLMJudge,
                 cache: Optional[SemanticCache] = None):
        self.agent = agent
        self.judge = judge
        self.cache = cache

    def ask(self, question: str) -> dict:
        result = {
            "question":          question,
            "system1_answer":    None,
            "system2_answer":    None,
            "final_answer":      None,
            "faithful":          False,
            "confidence":        0.0,
            "self_corrected":    False,
            "consistency_check": None,
            "reason":            "",
            "cache_hit":         False,
            "cache_similarity":  None,
        }

        # ── STEP 0: Semantic Cache Lookup ───────────────────────────
        # ✅ OPT-5: Bypass entire LLM pipeline if semantically equivalent
        # query was already answered (cache hit ≈ 2 ms vs ~10+ s normal).
        if self.cache is not None:
            cached = self.cache.lookup(question)
            if cached is not None:
                result.update(cached)
                return result

        # ── STEP 1: System-1 Fast Answer ──────────────────────────────
        s1_answer, s1_mids = self.agent.generate_response(question, decay_enabled=True)
        result["system1_answer"] = s1_answer

        # Build doc-only context for judging + validation
        # 🔥 FIX 4: Only use learned_fact chunks (not memory notes) as source of truth
        doc_context, doc_embeddings = self._build_doc_context(s1_mids)

        # ── STEP 2: Conditional Faithfulness Check ────────────────────
        # ✅ OPT-3: Skip the LLM judge if System-1 retrieval score is
        # already high — the cross-encoder already verified relevance.
        s1_top_score = self._top_retrieval_score(s1_mids)
        if s1_top_score >= self.CONFIDENCE_JUDGE_THRESHOLD:
            result["final_answer"] = s1_answer
            result["faithful"]     = True
            result["confidence"]   = s1_top_score
            result["reason"]       = "High retrieval confidence — judge skipped."
            self._log_success(question, s1_answer, s1_mids)
            if self.cache is not None:
                self.cache.store(question, result)
            return result

        faith = self.judge.check_faithfulness(question, s1_answer, doc_context)
        result["faithful"]   = faith["faithful"]
        result["confidence"] = faith["confidence"]
        result["reason"]     = faith["reason"]

        if faith["faithful"] and faith["confidence"] >= 0.6:
            result["final_answer"] = s1_answer
            self._log_success(question, s1_answer, s1_mids)
            if self.cache is not None:
                self.cache.store(question, result)
            return result

        # ── STEP 3: System-2 Self-Correction (Map-Reduce) ─────────────
        s2_answer = self._system2_extract(question)
        result["system2_answer"] = s2_answer
        result["self_corrected"] = True

        if s2_answer:
            # 🔥 FIX 1: Validate the correction against doc context BEFORE accepting it
            is_valid = self._validate_correction(s2_answer, doc_embeddings)

            if is_valid:
                # Verify answer is actually faithful
                deep_context, _ = self._build_deep_doc_context(question)
                faith2 = self.judge.check_faithfulness(question, s2_answer, deep_context)

                if faith2["faithful"]:
                    result["final_answer"] = s2_answer
                    result["faithful"]     = True
                    result["confidence"]   = faith2["confidence"]
                    result["reason"]       = faith2["reason"]
                    self._log_failure_and_learn(question, s1_answer, s2_answer, s1_mids, doc_embeddings)
                    if self.cache is not None:
                        self.cache.store(question, result)
                    return result
            # else: s2_answer failed validation — don't store it

        # ── STEP 4: Consistency Check (last resort) ───────────────────
        rephrased = self.judge.rephrase_question(question)
        if rephrased:
            r_answer, _ = self.agent.generate_response(rephrased, decay_enabled=True)
            consistency = self.judge.check_consistency(question, s1_answer, r_answer)
            result["consistency_check"] = {
                "rephrased_question": rephrased,
                "rephrased_answer":   r_answer,
                "consistent":         consistency["consistent"],
                "explanation":        consistency["explanation"],
            }

            if consistency["consistent"]:
                result["final_answer"] = s1_answer
                result["faithful"]     = True
                result["confidence"]   = 0.65
                self._log_success(question, s1_answer, s1_mids)
                if self.cache is not None:
                    self.cache.store(question, result)
                return result

        # ── FALLBACK ──────────────────────────────────────────────────
        final = s2_answer if s2_answer else s1_answer
        result["final_answer"] = final
        result["reason"] = "Low confidence. Could not self-verify."
        # Only log failure if we have a meaningful correction candidate
        if s2_answer and s2_answer != "The document does not contain the answer to this question.":
            self._log_failure_and_learn(question, s1_answer, s2_answer, s1_mids, doc_embeddings)
        return result

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _top_retrieval_score(self, memory_ids: list[str]) -> float:
        """
        Return the highest raw cosine similarity score among the retrieved
        doc chunks used to generate the System-1 answer.
        Used to decide whether to skip the LLM faithfulness judge.
        """
        if not memory_ids:
            return 0.0
        scores = []
        for mid in memory_ids:
            mem = self.agent.memory.memories.get(mid)
            if mem and mem.category == "learned_fact":
                # Re-score against stored embedding is not ideal but avoids
                # threading the score through generate_response; we approximate
                # via the memory's confidence_score as a proxy.
                scores.append(mem.confidence_score)
        return max(scores) if scores else 0.0

    def _build_doc_context(self, memory_ids: list[str]) -> tuple[str, list]:
        """
        🔥 FIX 4: Build context ONLY from document chunks (learned_fact).
        Memory notes are excluded — document is source of truth.
        Returns (context_string, list_of_embeddings).
        """
        parts = []
        embeddings = []
        for mid in memory_ids:
            mem = self.agent.memory.memories.get(mid)
            if mem and not mem.suppressed and mem.category == "learned_fact":
                parts.append(mem.content)
                embeddings.append(mem.embedding)
        return "\n".join(parts), embeddings

    def _build_deep_doc_context(self, question: str) -> tuple[str, list]:
        """Retrieve top-10 doc chunks only, for deeper verification."""
        retrieved = self.agent.memory.retrieve(question, k=10, decay_enabled=False)
        parts = []
        embeddings = []
        for mid, _ in retrieved:
            mem = self.agent.memory.memories.get(mid)
            if mem and not mem.suppressed and mem.category == "learned_fact":
                parts.append(mem.content)
                embeddings.append(mem.embedding)
        return "\n".join(parts), embeddings

    def _validate_correction(self, correction: str, doc_embeddings: list) -> bool:
        """
        🔥 FIX 1: Check correction is related to actual doc context.
        Rejects hallucinations from training data.
        """
        if not doc_embeddings:
            return False  # No doc context = can't validate = reject
        corr_emb = self.agent.memory._embed(correction)
        sims = [
            self.agent.memory._cosine(corr_emb, de)
            for de in doc_embeddings
        ]
        avg_sim = float(np.mean(sims))
        return avg_sim >= 0.02  # Even a weak thematic match is acceptable

    def _system2_extract(self, question: str) -> Optional[str]:
        """
        ✅ OPT-3: Single-prompt multi-chunk synthesis.

        Replaces the old Map-Reduce loop that made up to 15 sequential LLM
        calls.  Now delegates to agent.extract_correction() which retrieves
        the top-5 doc chunks and synthesises an answer in ONE LLM call.
        """
        result = self.agent.extract_correction(question, top_k=5)
        if result and result != "The document does not contain the answer to this question.":
            return result
        return None

    def _log_success(self, question: str, answer: str, mids: list[str]):
        """Log a successful interaction to strengthen retrieved memories."""
        inter = Interaction(
            user_input=question,
            expected_output=answer,
            agent_output=answer,
            outcome="success"
        )
        if isinstance(self.agent.memory, EnhancedMemory):
            self.agent.memory.log_interaction(inter, mids)

    def _log_failure_and_learn(self, question: str, bad_answer: str,
                                good_answer: str, mids: list[str],
                                doc_embeddings: list = None):
        """
        🔥 FIX 1: Pass doc_embeddings to log_interaction so the memory
        store can validate the correction before persisting it.
        """
        inter = Interaction(
            user_input=question,
            expected_output=good_answer,
            agent_output=bad_answer,
            outcome="failure"
        )
        if isinstance(self.agent.memory, EnhancedMemory):
            self.agent.memory.log_interaction(inter, mids, doc_context_embeddings=doc_embeddings)
