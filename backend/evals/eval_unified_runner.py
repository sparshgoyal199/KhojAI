# backend/evals/eval_unified_runner.py
"""
Unified Quality & Operational Evaluation Runner.

Evaluates:
  - Operational Latencies (E2E, TTFT, Embed, Search, Decode) + Throughput + Cost
  - The 5 Core Quality Metrics (Contextual Precision, Recall, Faithfulness, Contextual Relevancy, Answer Relevancy)

Execution Policy:
  - Phase 1 (Operational): Timed production pipeline. Clock stops before Phase 2.
  - Phase 2 (Quality): Untimed. Evaluates 1 metric at a time, sequentially (async_mode=False),
    with breathing pauses between metrics. Single generator LLM, FallbackJudge for metrics.
"""
import argparse
import asyncio
import json
import os
import shutil
import time
from datetime import datetime, timezone
from typing import Dict, Any, List
import numpy as np

# DeepEval metrics
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
from langchain_core.messages import SystemMessage, HumanMessage

from evals.harness import load_goldens
from evals.fallback_judge import FallbackJudge
from evals.eval_unified_metrics import (
    LatencyBudget,
    QualitySLA,
    inspect_runtime_configuration,
    compute_configuration_diff,
    compute_noise_statistics,
    evaluate_delta_vs_noise,
    calculate_cost,
    segment_workload,
)
from pipeline.prompt import system_prompt
from services.embedding_service import generate_embeddings
from services.retrieval_service import (
    creating_user_prompt,
    retrieve_relevant_chunks,
    format_retrieved_chunks,
)
from core.llm import llm

GOLDEN_PATH = "goldens/retriever_goldens.json"
BASELINE_PATH = "goldens/baseline.json"
CANDIDATE_PATH = "goldens/candidate.json"
HISTORY_PATH = "goldens/history.json"
ARCHIVE_DIR = "goldens/baselines_archive"
DOC_ID = "0d7887ca44684c71b407bfd8842a720c"

# Rate-limiting safety pause between judge metrics
PAUSE_BETWEEN_METRICS = 2.0  # seconds

# Global Fallback Judge instance (absorbs Groq 429s during judging)
eval_judge = FallbackJudge()


async def evaluate_single_row(
    g: Dict[str, Any],
    latency_budget: LatencyBudget,
    quality_sla: QualitySLA,
    run_quality: bool = True,
) -> Dict[str, Any]:
    query = g["query"]
    ideal_answer = g.get("ideal_answer") or g.get("expected_output", "")

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 1: TIMED PRODUCTION EXECUTION (Operational Evals)
    # ═════════════════════════════════════════════════════════════════════════
    t_e2e_start = time.perf_counter()
    t_embed_sec = 0.0
    t_search_sec = 0.0
    t_prompt_sec = 0.0
    t_ttft_sec = None
    t_gen_sec = 0.0

    try:
        # 1. Embedding
        t_embed_start = time.perf_counter()
        query_embedding = await generate_embeddings(query=query)
        t_embed_sec = time.perf_counter() - t_embed_start

        # 2. Vector Search (Qdrant Hybrid)
        t_search_start = time.perf_counter()
        retrieved_chunks = await retrieve_relevant_chunks(DOC_ID, query_embedding, query)
        t_search_sec = time.perf_counter() - t_search_start

        total_retrieval_sec = t_embed_sec + t_search_sec

        # 3. Prompt Compilation & Footprints
        t_prompt_start = time.perf_counter()
        user_prompt = creating_user_prompt(retrieved_chunks, query, messages=None)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        t_prompt_sec = time.perf_counter() - t_prompt_start

        formatted_context = format_retrieved_chunks(retrieved_chunks)
        context_char_len = len(formatted_context)
        raw_content_char_len = len("".join(c.get("content", "") for c in retrieved_chunks))
        query_char_len = len(query)
        total_prompt_char_len = len(user_prompt)

        # 4. Generation & TTFT (Streaming via Production LLM)
        t_gen_start = time.perf_counter()
        first_token_time = None
        collected_tokens = []
        usage_metadata = {}

        async for chunk in llm.astream(messages):
            if first_token_time is None and chunk.content:
                first_token_time = time.perf_counter()
            collected_tokens.append(chunk.content)
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                usage_metadata = chunk.usage_metadata

        t_gen_sec = time.perf_counter() - t_gen_start
        t_ttft_sec = (first_token_time - t_gen_start) if first_token_time else t_gen_sec

        # ── CLOCK STOPS HERE! Operational E2E Latency is strictly locked ─────
        t_e2e_sec = time.perf_counter() - t_e2e_start
        actual_output = "".join(collected_tokens)

        # 5. Token & Cost Accounting
        prompt_tokens = usage_metadata.get("input_tokens") or max(1, len(user_prompt) // 4)
        completion_tokens = usage_metadata.get("output_tokens") or max(1, len(actual_output) // 4)
        model_name = getattr(llm, "model_name", "openai/gpt-oss-20b")
        cost_usd = calculate_cost(model_name, prompt_tokens, completion_tokens)

        inter_token_duration = max(t_gen_sec - t_ttft_sec, 0.001)
        tokens_per_sec = (completion_tokens - 1) / inter_token_duration if completion_tokens > 1 else 0.0

        # Operational Budget Checks
        oper_breaches = []
        if t_embed_sec > latency_budget.embedding_max_sec: oper_breaches.append("Embedding")
        if t_search_sec > latency_budget.retrieval_search_max_sec: oper_breaches.append("Qdrant Search")
        if total_retrieval_sec > latency_budget.total_retrieval_max_sec: oper_breaches.append("Total Retrieval")
        if t_ttft_sec > latency_budget.ttft_max_sec: oper_breaches.append("TTFT")
        if t_gen_sec > latency_budget.generation_max_sec: oper_breaches.append("Generation")
        if t_e2e_sec > latency_budget.e2e_max_sec: oper_breaches.append("E2E Latency")
        if cost_usd > latency_budget.max_cost_usd: oper_breaches.append("Cost")

        workload = segment_workload(query, context_char_len)

    except Exception as e:
        return {
            "status": "FAILED",
            "query": query,
            "duration_until_fail_sec": round(time.perf_counter() - t_e2e_start, 3),
            "error_type": type(e).__name__,
            "error": str(e),
        }

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 2: UNTIMED QUALITY EVALUATION (One Metric at a Time, Strictly Serial)
    # ═════════════════════════════════════════════════════════════════════════
    quality_scores = {}
    quality_breaches = []

    if run_quality:
        # Build test case reusing Phase 1 artifacts
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            expected_output=ideal_answer,
            retrieval_context=[c.get("content", "") for c in retrieved_chunks],
        )

        # ── 1. Contextual Precision ──────────────────────────────────────────
        try:
            m_prec = ContextualPrecisionMetric(
                threshold=quality_sla.min_precision,
                model=eval_judge,
                async_mode=False,  # Serial execution
                include_reason=True,
            )
            m_prec.measure(test_case)
            quality_scores["retriever_precision"] = round(float(m_prec.score), 4)
            if not m_prec.is_successful():
                quality_breaches.append(f"Precision ({m_prec.score:.2f} < {quality_sla.min_precision:.2f})")
        except Exception as e:
            quality_scores["retriever_precision"] = 0.0
            quality_breaches.append(f"Precision Error: {str(e)[:40]}")

        await asyncio.sleep(PAUSE_BETWEEN_METRICS)

        # ── 2. Contextual Recall ─────────────────────────────────────────────
        try:
            m_rec = ContextualRecallMetric(
                threshold=quality_sla.min_recall,
                model=eval_judge,
                async_mode=False,
                include_reason=True,
            )
            m_rec.measure(test_case)
            quality_scores["retriever_recall"] = round(float(m_rec.score), 4)
            if not m_rec.is_successful():
                quality_breaches.append(f"Recall ({m_rec.score:.2f} < {quality_sla.min_recall:.2f})")
        except Exception as e:
            quality_scores["retriever_recall"] = 0.0
            quality_breaches.append(f"Recall Error: {str(e)[:40]}")

        await asyncio.sleep(PAUSE_BETWEEN_METRICS)

        # ── 3. Faithfulness (Hallucination Check) ────────────────────────────
        try:
            m_faith = FaithfulnessMetric(
                threshold=quality_sla.min_faithfulness,
                model=eval_judge,
                async_mode=False,
                include_reason=True,
            )
            m_faith.measure(test_case)
            quality_scores["rag_faithfulness"] = round(float(m_faith.score), 4)
            if not m_faith.is_successful():
                quality_breaches.append(f"Faithfulness ({m_faith.score:.2f} < {quality_sla.min_faithfulness:.2f})")
        except Exception as e:
            quality_scores["rag_faithfulness"] = 0.0
            quality_breaches.append(f"Faithfulness Error: {str(e)[:40]}")

        await asyncio.sleep(PAUSE_BETWEEN_METRICS)

        # ── 4. Contextual Relevancy (Noise in context) ────────────────────────
        try:
            m_crel = ContextualRelevancyMetric(
                threshold=quality_sla.min_contextual_relevancy,
                model=eval_judge,
                async_mode=False,
                include_reason=True,
            )
            m_crel.measure(test_case)
            quality_scores["rag_contextual_relevancy"] = round(float(m_crel.score), 4)
            if not m_crel.is_successful():
                quality_breaches.append(f"Context Relevancy ({m_crel.score:.2f} < {quality_sla.min_contextual_relevancy:.2f})")
        except Exception as e:
            quality_scores["rag_contextual_relevancy"] = 0.0
            quality_breaches.append(f"Context Relevancy Error: {str(e)[:40]}")

        await asyncio.sleep(PAUSE_BETWEEN_METRICS)

        # ── 5. Answer Relevancy ──────────────────────────────────────────────
        try:
            m_arel = AnswerRelevancyMetric(
                threshold=quality_sla.min_answer_relevancy,
                model=eval_judge,
                async_mode=False,
                include_reason=True,
            )
            m_arel.measure(test_case)
            quality_scores["answer_relevancy"] = round(float(m_arel.score), 4)
            if not m_arel.is_successful():
                quality_breaches.append(f"Answer Relevancy ({m_arel.score:.2f} < {quality_sla.min_answer_relevancy:.2f})")
        except Exception as e:
            quality_scores["answer_relevancy"] = 0.0
            quality_breaches.append(f"Answer Relevancy Error: {str(e)[:40]}")

    return {
        "status": "SUCCESS",
        "query": query,
        "actual_output": actual_output,
        "workload": workload,
        "context_metrics": {
            "chunk_count": len(retrieved_chunks),
            "context_char_len": context_char_len,
            "raw_content_char_len": raw_content_char_len,
            "query_char_len": query_char_len,
            "total_prompt_char_len": total_prompt_char_len,
        },
        "token_metrics": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "tokens_per_sec": round(tokens_per_sec, 1),
        },
        "component_latencies_sec": {
            "embedding": round(t_embed_sec, 3),
            "retrieval_search": round(t_search_sec, 3),
            "total_retrieval": round(total_retrieval_sec, 3),
            "prompt_compilation": round(t_prompt_sec, 4),
            "ttft": round(t_ttft_sec, 3),
            "generation": round(t_gen_sec, 3),
            "e2e_total": round(t_e2e_sec, 3),
        },
        "cost_usd": round(cost_usd, 6),
        "operational_budget": {
            "passed": len(oper_breaches) == 0,
            "breaches": oper_breaches,
        },
        "quality_scores": quality_scores,
        "quality_budget": {
            "passed": len(quality_breaches) == 0,
            "breaches": quality_breaches,
        },
        "error": None,
    }


async def run_eval_pass(
    goldens: List[dict],
    latency_budget: LatencyBudget,
    quality_sla: QualitySLA,
    run_quality: bool = True,
) -> Dict[str, Any]:
    rows = []
    for idx, g in enumerate(goldens, 1):
        print(f"    -> Query [{idx}/{len(goldens)}]: {g['query'][:45]!r}")
        res = await evaluate_single_row(g, latency_budget, quality_sla, run_quality=run_quality)
        rows.append(res)
        await asyncio.sleep(2.0)  # breathing pause before next row

    successes = [r for r in rows if r["status"] == "SUCCESS"]
    if not successes:
        raise RuntimeError("All rows failed in this pass!")

    # 1. Operational Distributions
    e2e = [r["component_latencies_sec"]["e2e_total"] for r in successes]
    ttft = [r["component_latencies_sec"]["ttft"] for r in successes]
    embed = [r["component_latencies_sec"]["embedding"] for r in successes]
    search = [r["component_latencies_sec"]["retrieval_search"] for r in successes]
    gen = [r["component_latencies_sec"]["generation"] for r in successes]
    tps = [r["token_metrics"]["tokens_per_sec"] for r in successes]
    cost = [r["cost_usd"] for r in successes]
    oper_passes = sum(1 for r in successes if r["operational_budget"]["passed"])

    summary = {
        "e2e_p50": float(np.percentile(e2e, 50)),
        "e2e_p95": float(np.percentile(e2e, 95)),
        "e2e_mean": float(np.mean(e2e)),
        "ttft_p50": float(np.percentile(ttft, 50)),
        "ttft_p95": float(np.percentile(ttft, 95)),
        "embedding_p50": float(np.percentile(embed, 50)),
        "embedding_p95": float(np.percentile(embed, 95)),
        "search_p50": float(np.percentile(search, 50)),
        "search_p95": float(np.percentile(search, 95)),
        "generation_p50": float(np.percentile(gen, 50)),
        "generation_p95": float(np.percentile(gen, 95)),
        "tokens_per_sec_mean": float(np.mean(tps)),
        "cost_usd_mean": float(np.mean(cost)),
        "sla_pass_rate": float(oper_passes / len(successes) * 100.0),
        "error_rate": float((len(rows) - len(successes)) / len(rows) * 100.0),
    }

    # 2. Quality Distributions (Mean scores across the dataset)
    if run_quality:
        prec = [r["quality_scores"].get("retriever_precision", 0.0) for r in successes]
        rec = [r["quality_scores"].get("retriever_recall", 0.0) for r in successes]
        faith = [r["quality_scores"].get("rag_faithfulness", 0.0) for r in successes]
        crel = [r["quality_scores"].get("rag_contextual_relevancy", 0.0) for r in successes]
        arel = [r["quality_scores"].get("answer_relevancy", 0.0) for r in successes]
        qual_passes = sum(1 for r in successes if r["quality_budget"]["passed"])

        summary.update({
            "retriever_precision": float(np.mean(prec)),
            "retriever_recall": float(np.mean(rec)),
            "rag_faithfulness": float(np.mean(faith)),
            "rag_contextual_relevancy": float(np.mean(crel)),
            "answer_relevancy": float(np.mean(arel)),
            "quality_pass_rate": float(qual_passes / len(successes) * 100.0),
        })

    return {"summary": summary, "rows": rows}


def promote_candidate_to_baseline():
    """Archives old baseline and promotes candidate.json as new baseline.json."""
    if not os.path.exists(CANDIDATE_PATH):
        raise FileNotFoundError(f"Cannot promote: {CANDIDATE_PATH} does not exist.")

    with open(CANDIDATE_PATH, "r", encoding="utf-8") as f:
        cand = json.load(f)

    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    if os.path.exists(BASELINE_PATH):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        archive_path = os.path.join(ARCHIVE_DIR, f"baseline_{ts}.json")
        shutil.copyfile(BASELINE_PATH, archive_path)
        print(f"✓ Archived previous baseline to: {archive_path}")

    metrics = cand.get("calibrated_metrics") or {
        k: {
            "mean": v["candidate"],
            "std_dev": 0.0,
            "cv_percent": 0.0,
            "noise_threshold": v["noise_threshold"],
            "noise_band_low": v["candidate"],
            "noise_band_high": v["candidate"],
            "raw_runs": [v["candidate"]],
        }
        for k, v in cand["comparison"].items()
    }

    new_baseline = {
        "metadata": cand["metadata"],
        "calibration_runs": cand.get("runs_evaluated", 1),
        "golden_count": len(cand.get("detailed_rows", [])),
        "metrics": metrics,
    }

    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(new_baseline, f, indent=2)

    print("=" * 78)
    print(f"🎉 PROMOTION COMPLETE! '{cand['metadata']['config_name']}' is now the active baseline.")
    print(f"Saved to: {BASELINE_PATH}")
    print("=" * 78)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true", help="Calibrate as a fresh baseline")
    parser.add_argument("--promote", action="store_true", help="Promote candidate.json to baseline.json")
    parser.add_argument("--runs", type=int, default=5, help="Number of calibration/eval passes (default: 5)")
    parser.add_argument("--config-name", type=str, default=None, help="Configuration label")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of golden items")
    parser.add_argument("--skip-quality", action="store_true", help="Run operational evals only")
    args = parser.parse_args()

    if args.promote:
        promote_candidate_to_baseline()
        return

    config_name = args.config_name or ("Baseline " if args.calibrate else "Candidate ") + datetime.now().strftime("%b %d %H:%M")
    current_sys_config = inspect_runtime_configuration()
    latency_budget = LatencyBudget()
    quality_sla = QualitySLA()
    run_quality = not args.skip_quality

    goldens = load_goldens(GOLDEN_PATH)
    if args.limit:
        goldens = goldens[:args.limit]

    print("=" * 78)
    mode_title = "CALIBRATING UNIFIED BASELINE" if args.calibrate else "EVALUATING UNIFIED CANDIDATE"
    print(f"{mode_title}: '{config_name}' ({args.runs} RUNS, {len(goldens)} GOLDEN ITEMS)")
    print(f"Quality Metrics: {'ENABLED (Strict Serial)' if run_quality else 'DISABLED'}")
    print("=" * 78)

    runs_data = []
    last_run_rows = []
    for r in range(1, args.runs + 1):
        print(f"\n[Pass {r}/{args.runs}] Executing full pass...")
        t0 = time.perf_counter()
        pass_res = await run_eval_pass(goldens, latency_budget, quality_sla, run_quality=run_quality)
        runs_data.append(pass_res["summary"])
        last_run_rows = pass_res["rows"]
        summ = pass_res["summary"]
        print(f"  ✓ Pass {r} finished in {time.perf_counter() - t0:.1f}s | "
              f"E2E P95: {summ['e2e_p95']:.2f}s | TTFT P95: {summ['ttft_p95']:.2f}s | "
              f"Faithfulness: {summ.get('rag_faithfulness', 'N/A')}")
        if r < args.runs:
            await asyncio.sleep(3.0)

    # 1. Baseline Calibration Path
    if args.calibrate:
        stats = compute_noise_statistics(runs_data)
        baseline_payload = {
            "metadata": {
                "config_name": config_name,
                "calibrated_at": datetime.now(timezone.utc).isoformat(),
                "components": current_sys_config,
            },
            "calibration_runs": args.runs,
            "golden_count": len(goldens),
            "metrics": stats,
        }
        os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)
        with open(BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump(baseline_payload, f, indent=2)

        print("\n" + "=" * 78)
        print("BASELINE CALIBRATED AND SAVED (Operational + Quality)!")
        print(f"File: {BASELINE_PATH}")
        print("=" * 78)
        return

    # 2. Candidate Evaluation Path
    if not os.path.exists(BASELINE_PATH):
        raise FileNotFoundError(f"Missing {BASELINE_PATH}! Run with '--calibrate --runs 5' first.")

    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        baseline_file = json.load(f)

    cand_avg_metrics = {m: float(np.mean([run[m] for run in runs_data])) for m in runs_data[0].keys()}
    cand_noise_stats = compute_noise_statistics(runs_data) if len(runs_data) > 1 else None
    config_diff = compute_configuration_diff(baseline_file["metadata"]["components"], current_sys_config)

    comparison = {}
    for m, val in cand_avg_metrics.items():
        if m in baseline_file["metrics"]:
            comparison[m] = evaluate_delta_vs_noise(m, val, baseline_file["metrics"][m])

    candidate_payload = {
        "metadata": {
            "config_name": config_name,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "components": current_sys_config,
        },
        "runs_evaluated": args.runs,
        "config_diff": config_diff,
        "baseline_referenced": {
            "config_name": baseline_file["metadata"]["config_name"],
            "calibrated_at": baseline_file["metadata"]["calibrated_at"],
            "components": baseline_file["metadata"]["components"],
        },
        "comparison": comparison,
        "calibrated_metrics": cand_noise_stats,
        "detailed_rows": last_run_rows,
    }

    with open(CANDIDATE_PATH, "w", encoding="utf-8") as f:
        json.dump(candidate_payload, f, indent=2)

    # Rolling history
    history = []
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f: history = json.load(f)
        except Exception: history = []
    history.append({
        "config_name": config_name,
        "timestamp": candidate_payload["metadata"]["evaluated_at"],
        "e2e_p95": cand_avg_metrics["e2e_p95"],
        "ttft_p95": cand_avg_metrics["ttft_p95"],
        "faithfulness": cand_avg_metrics.get("rag_faithfulness"),
        "precision": cand_avg_metrics.get("retriever_precision"),
        "e2e_verdict": comparison["e2e_p95"]["verdict"],
    })
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Terminal Report
    print("\n" + "=" * 78)
    print(f"UNIFIED COMPARISON REPORT: '{config_name}' vs BASELINE")
    print("=" * 78)

    print("\n[1] OPERATIONAL HEALTH (Latency & Cost)")
    for m in ["e2e_p95", "ttft_p95", "tokens_per_sec_mean", "cost_usd_mean", "sla_pass_rate"]:
        if m in comparison:
            c = comparison[m]
            v_icon = {"IMPROVED": "🟢 ▲", "REGRESSED": "🔴 ▼", "NOISE": "⚪ ≈"}.get(c["verdict"], "")
            print(f"  {m:<22} Cand: {c['candidate']:7.3f} | Base: {c['baseline_mean']:7.3f} | "
                  f"Δ: {c['pct_delta']:+6.1f}% | {v_icon} {c['verdict']} (±{c['noise_threshold']:.3f})")

    if run_quality:
        print("\n[2] GROUNDED QUALITY & RAG TRIAD")
        for m in ["retriever_precision", "retriever_recall", "rag_faithfulness", "rag_contextual_relevancy", "answer_relevancy", "quality_pass_rate"]:
            if m in comparison:
                c = comparison[m]
                v_icon = {"IMPROVED": "🟢 ▲", "REGRESSED": "🔴 ▼", "NOISE": "⚪ ≈"}.get(c["verdict"], "")
                print(f"  {m:<24} Cand: {c['candidate']:7.3f} | Base: {c['baseline_mean']:7.3f} | "
                      f"Δ: {c['pct_delta']:+6.1f}% | {v_icon} {c['verdict']} (±{c['noise_threshold']:.3f})")

    print("\n" + "=" * 78)
    print(f"Results saved to: {CANDIDATE_PATH}")
    print("View visual dashboard: Open backend/evals/dashboard.html")
    print("To promote this candidate to be the new official baseline, run:")
    print("  python -m evals.eval_unified_runner --promote")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())