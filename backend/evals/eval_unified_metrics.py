# backend/evals/eval_unified_metrics.py
"""
Unified Metrics Engine: Directionalities, Configuration Diffing, and Noise Math.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class QualitySLA:
    """Quality thresholds (scale: 0.0 to 1.0)."""
    min_precision: float = 0.70
    min_recall: float = 0.70
    min_faithfulness: float = 0.70
    min_contextual_relevancy: float = 0.70
    min_answer_relevancy: float = 0.70


@dataclass
class LatencyBudget:
    """Latency SLA budgets in seconds."""
    embedding_max_sec: float = 0.800
    retrieval_search_max_sec: float = 0.400
    total_retrieval_max_sec: float = 1.200
    ttft_max_sec: float = 1.000
    generation_max_sec: float = 2.500
    e2e_max_sec: float = 3.500
    max_cost_usd: float = 0.005


MODEL_PRICING = {
    "openai/gpt-oss-20b": {"input_per_m": 0.10, "output_per_m": 0.30},
    "openai/gpt-oss-120b": {"input_per_m": 0.30, "output_per_m": 1.20},
    "mistral-large-latest": {"input_per_m": 2.00, "output_per_m": 6.00},
    "qwen/qwen3.8-27b": {"input_per_m": 0.20, "output_per_m": 0.60},
}

# Metric Directionality:
# True  = Higher is Better (▲) [All Quality Metrics, Throughput, SLA Pass Rates]
# False = Lower is Better  (▼) [All Latencies, Cost, Error Rate]
METRIC_HIGHER_IS_BETTER = {
    # ── Quality Metrics (0.0 to 1.0) ──────────────────────────────────────────
    "retriever_precision": True,
    "retriever_recall": True,
    "rag_faithfulness": True,
    "rag_contextual_relevancy": True,
    "answer_relevancy": True,
    # ── Operational Throughput & Reliability ──────────────────────────────────
    "tokens_per_sec_mean": True,
    "tokens_per_sec_p50": True,
    "sla_pass_rate": True,
    "quality_pass_rate": True,
    # ── Operational Latencies & Cost ──────────────────────────────────────────
    "e2e_p50": False,
    "e2e_p95": False,
    "e2e_mean": False,
    "ttft_p50": False,
    "ttft_p95": False,
    "embedding_p50": False,
    "embedding_p95": False,
    "search_p50": False,
    "search_p95": False,
    "generation_p50": False,
    "generation_p95": False,
    "cost_usd_mean": False,
    "error_rate": False,
}


def inspect_runtime_configuration() -> Dict[str, Any]:
    """Dynamically extracts system settings directly from active code objects."""
    from core.llm import llm
    from core.embedding_models import MAX_TOKENS, tokenizer

    llm_model = getattr(llm, "model_name", None) or getattr(llm, "model", "openai/gpt-oss-20b")
    llm_temp = getattr(llm, "temperature", 0.0)

    tok_name = getattr(tokenizer, "name_or_path", "BAAI/bge-base-en-v1.5")
    if hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "name_or_path"):
        tok_name = tokenizer.tokenizer.name_or_path

    return {
        "embedding_model": str(tok_name),
        "embedding_dim": 768,
        "chunk_max_tokens": int(MAX_TOKENS),
        "chunker_serializer": "MDTableSerializerProvider (Markdown)",
        "llm_model": str(llm_model),
        "llm_temperature": float(llm_temp),
        "vector_db": "Qdrant Cloud",
        "vector_search": "Hybrid (Dense DOT 768 + Sparse BM25 via RRF [1.5, 1.0])",
        "quantization": "Scalar INT8 (quantile 0.99)",
        "top_k": 5,
        "eval_judge": "FallbackJudge (Multi-model Groq)",
    }


def compute_configuration_diff(base_cfg: Dict[str, Any], cand_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Detects differences between baseline and candidate configurations."""
    diffs = {}
    all_keys = set(base_cfg.keys()).union(set(cand_cfg.keys()))
    for k in sorted(all_keys):
        b_val = base_cfg.get(k)
        c_val = cand_cfg.get(k)
        diffs[k] = {
            "baseline": b_val,
            "candidate": c_val,
            "changed": (b_val != c_val),
        }
    return diffs


def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model_name, {"input_per_m": 0.15, "output_per_m": 0.40})
    return (prompt_tokens / 1_000_000) * pricing["input_per_m"] + \
           (completion_tokens / 1_000_000) * pricing["output_per_m"]


def calculate_percentiles(values: List[float]) -> Dict[str, float]:
    """Computes Mean, P50, P90, P95, Min, and Max over an array of floats."""
    if not values:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(values)
    return {
        "n": len(values),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def compute_noise_statistics(runs_data: List[Dict[str, float]]) -> Dict[str, Any]:
    """
    Computes Mean (μ), StdDev (σ), CV (%), and 2-sigma Noise Threshold across runs.
    """
    metrics = runs_data[0].keys()
    stats = {}
    for m in metrics:
        values = [run[m] for run in runs_data]
        mean_val = float(np.mean(values))
        std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        cv_val = float((std_val / mean_val * 100.0) if mean_val != 0 else 0.0)
        noise_threshold = float(2.0 * std_val)  # 2-sigma (95% CI)

        stats[m] = {
            "mean": round(mean_val, 4),
            "std_dev": round(std_val, 4),
            "cv_percent": round(cv_val, 2),
            "noise_threshold": round(noise_threshold, 4),
            "noise_band_low": round(max(0.0, mean_val - noise_threshold), 4),
            "noise_band_high": round(mean_val + noise_threshold, 4),
            "raw_runs": [round(v, 4) for v in values],
        }
    return stats


def evaluate_delta_vs_noise(metric_name: str, candidate_val: float, baseline_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates candidate metric against baseline using directionality rules
    and the 2-sigma noise threshold.
    """
    b_mean = baseline_data["mean"]
    noise_thresh = baseline_data["noise_threshold"]
    higher_is_better = METRIC_HIGHER_IS_BETTER.get(metric_name, False)

    delta = candidate_val - b_mean
    pct_delta = (delta / b_mean * 100.0) if b_mean != 0 else 0.0

    if abs(delta) <= noise_thresh:
        verdict = "NOISE"
        verdict_icon = "≈"
    elif higher_is_better:
        if delta > noise_thresh:
            verdict = "IMPROVED"
            verdict_icon = "▲"
        else:
            verdict = "REGRESSED"
            verdict_icon = "▼"
    else:  # Lower is better (latency, cost, error_rate)
        if delta < -noise_thresh:
            verdict = "IMPROVED"
            verdict_icon = "▼"
        else:
            verdict = "REGRESSED"
            verdict_icon = "▲"

    return {
        "candidate": round(candidate_val, 4),
        "baseline_mean": b_mean,
        "delta": round(delta, 4),
        "pct_delta": round(pct_delta, 2),
        "noise_threshold": noise_thresh,
        "noise_band": [baseline_data["noise_band_low"], baseline_data["noise_band_high"]],
        "higher_is_better": higher_is_better,
        "verdict": verdict,
        "verdict_icon": verdict_icon,
    }


def segment_workload(query: str, context_char_len: int) -> Dict[str, str]:
    """Classifies query into representative operational workload segments."""
    if context_char_len < 1500:
        size_tier = "light (<1.5k chars)"
    elif context_char_len <= 3000:
        size_tier = "medium (1.5k-3k chars)"
    else:
        size_tier = "heavy (>3k chars)"

    q_lower = query.lower()
    if any(k in q_lower for k in ["what is", "define", "who is", "state"]):
        intent = "factoid_definition"
    elif any(k in q_lower for k in ["how", "explain", "describe", "why"]):
        intent = "concept_explanation"
    elif any(k in q_lower for k in ["compare", "difference", "versus", "vs", "summarize"]):
        intent = "synthesis_comparison"
    else:
        intent = "general_qa"

    return {"size_tier": size_tier, "intent": intent}