# backend/evals/eval_operational_latency.py
import argparse
import asyncio
import json
import time
import traceback
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage

from evals.harness import load_goldens
from evals.eval_operational_metrics import (
    LatencyBudget,
    calculate_cost,
    calculate_percentiles,
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

GOLDEN_PATH = "goldens/faithfulness_dataset.json"
RESULTS_PATH = "goldens/operational_latency_results.json"
DOC_ID = "0d7887ca44684c71b407bfd8842a720c"


async def evaluate_single_row(g: Dict[str, Any], budget: LatencyBudget) -> Dict[str, Any]:
    query = g["query"]
    t_e2e_start = time.perf_counter()

    # Track component timings
    t_embed_sec = 0.0
    t_search_sec = 0.0
    t_prompt_sec = 0.0
    t_ttft_sec = None
    t_gen_sec = 0.0

    try:
        # ── 1. Embedding Component ───────────────────────────────────────────
        t_embed_start = time.perf_counter()
        query_embedding = await generate_embeddings(query=query)
        t_embed_sec = time.perf_counter() - t_embed_start

        # ── 2. Vector Search Component (Qdrant Hybrid) ───────────────────────
        t_search_start = time.perf_counter()
        retrieved_chunks = await retrieve_relevant_chunks(DOC_ID, query_embedding, query)
        t_search_sec = time.perf_counter() - t_search_start

        total_retrieval_sec = t_embed_sec + t_search_sec

        # ── 3. Prompt Compilation & Footprint Measurement ────────────────────
        t_prompt_start = time.perf_counter()
        user_prompt = creating_user_prompt(retrieved_chunks, query, messages=None)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        t_prompt_sec = time.perf_counter() - t_prompt_start

        # Distinct footprint accounting:
        # 1. Full formatted context (headings, page numbers, filenames, content, [chunk X])
        formatted_context = format_retrieved_chunks(retrieved_chunks)
        context_char_len = len(formatted_context)
        # 2. Raw content text only
        raw_content_char_len = len("".join(c.get("content", "") for c in retrieved_chunks))
        # 3. User query length
        query_char_len = len(query)
        # 4. Total user prompt (context + history + query)
        total_prompt_char_len = len(user_prompt)

        # ── 4. Generation Component (TTFT & Streaming) ───────────────────────
        t_gen_start = time.perf_counter()
        first_token_time = None
        collected_tokens = []
        usage_metadata = {}

        # Consume token-by-token to capture real TTFT
        async for chunk in llm.astream(messages):
            if first_token_time is None and chunk.content:
                first_token_time = time.perf_counter()
            collected_tokens.append(chunk.content)
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                usage_metadata = chunk.usage_metadata

        t_gen_sec = time.perf_counter() - t_gen_start
        t_ttft_sec = (first_token_time - t_gen_start) if first_token_time else t_gen_sec
        t_e2e_sec = time.perf_counter() - t_e2e_start

        full_answer = "".join(collected_tokens)

        # ── 5. Token & Cost Accounting ────────────────────────────────────────
        # Fallback approximation: 1 token ~ 4 chars if provider doesn't stream usage
        prompt_tokens = usage_metadata.get("input_tokens") or max(1, len(user_prompt) // 4)
        completion_tokens = usage_metadata.get("output_tokens") or max(1, len(full_answer) // 4)
        model_name = getattr(llm, "model_name", "openai/gpt-oss-20b")
        cost_usd = calculate_cost(model_name, prompt_tokens, completion_tokens)

        # Inter-token generation throughput
        inter_token_duration = max(t_gen_sec - t_ttft_sec, 0.001)
        tokens_per_sec = (completion_tokens - 1) / inter_token_duration if completion_tokens > 1 else 0.0

        # ── 6. Component & System Budget Compliance ──────────────────────────
        breaches = []
        if t_embed_sec > budget.embedding_max_sec:
            breaches.append(f"Embedding ({t_embed_sec:.2f}s > {budget.embedding_max_sec:.2f}s)")
        if t_search_sec > budget.retrieval_search_max_sec:
            breaches.append(f"Qdrant ({t_search_sec:.2f}s > {budget.retrieval_search_max_sec:.2f}s)")
        if total_retrieval_sec > budget.total_retrieval_max_sec:
            breaches.append(f"Total Retrieval ({total_retrieval_sec:.2f}s > {budget.total_retrieval_max_sec:.2f}s)")
        if t_ttft_sec > budget.ttft_max_sec:
            breaches.append(f"TTFT ({t_ttft_sec:.2f}s > {budget.ttft_max_sec:.2f}s)")
        if t_gen_sec > budget.generation_max_sec:
            breaches.append(f"Generation ({t_gen_sec:.2f}s > {budget.generation_max_sec:.2f}s)")
        if t_e2e_sec > budget.e2e_max_sec:
            breaches.append(f"E2E Latency ({t_e2e_sec:.2f}s > {budget.e2e_max_sec:.2f}s)")
        if cost_usd > budget.max_cost_usd:
            breaches.append(f"Cost (${cost_usd:.5f} > ${budget.max_cost_usd:.5f})")

        workload = segment_workload(query, context_char_len)

        return {
            "status": "SUCCESS",
            "query": query,
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
            "budget": {
                "passed": len(breaches) == 0,
                "breaches": breaches,
            },
            "error": None,
        }

    except Exception as e:
        t_fail = time.perf_counter() - t_e2e_start
        return {
            "status": "FAILED",
            "query": query,
            "duration_until_fail_sec": round(t_fail, 3),
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def print_distribution_table(title: str, stats: Dict[str, float], budget_target: Optional[float] = None):
    target_str = f"Budget: <={budget_target:.2f}s" if budget_target else ""
    print(f"\n{title:<28} {target_str}")
    print(f"  P50:  {stats['p50']:6.3f}s | P90: {stats['p90']:6.3f}s | P95: {stats['p95']:6.3f}s")
    print(f"  Mean: {stats['mean']:6.3f}s | Min: {stats['min']:6.3f}s | Max: {stats['max']:6.3f}s")


def print_comprehensive_summary(results: List[Dict[str, Any]], budget: LatencyBudget):
    successes = [r for r in results if r["status"] == "SUCCESS"]
    failures = [r for r in results if r["status"] == "FAILED"]

    print("\n" + "=" * 78)
    print("           OPERATIONAL EVALUATION: COMPREHENSIVE REPORT")
    print("=" * 78)
    print(f"Total Requests: {len(results)} | Successful: {len(successes)} | Failed: {len(failures)}")
    print(f"System Error Rate: {len(failures) / len(results):.1%}")

    if failures:
        print("\n[!] Tracked Failures (isolated from latency statistics):")
        for f in failures:
            print(f"  • {f['query'][:50]!r} -> {f['error_type']}: {f['error']}")

    if not successes:
        print("No successful executions to compute percentiles.")
        return

    # Extract component values strictly from successful runs
    embed_times = [r["component_latencies_sec"]["embedding"] for r in successes]
    search_times = [r["component_latencies_sec"]["retrieval_search"] for r in successes]
    retrieval_times = [r["component_latencies_sec"]["total_retrieval"] for r in successes]
    ttft_times = [r["component_latencies_sec"]["ttft"] for r in successes]
    gen_times = [r["component_latencies_sec"]["generation"] for r in successes]
    e2e_times = [r["component_latencies_sec"]["e2e_total"] for r in successes]
    costs = [r["cost_usd"] for r in successes]
    throughputs = [r["token_metrics"]["tokens_per_sec"] for r in successes]

    print("\n--- COMPONENT-LEVEL LATENCY DISTRIBUTIONS ---")
    print_distribution_table("1. Modal Query Embedding", calculate_percentiles(embed_times), budget.embedding_max_sec)
    print_distribution_table("2. Qdrant Hybrid Search", calculate_percentiles(search_times), budget.retrieval_search_max_sec)
    print_distribution_table("3. Total Retrieval Phase", calculate_percentiles(retrieval_times), budget.total_retrieval_max_sec)
    print_distribution_table("4. Time To First Token (TTFT)", calculate_percentiles(ttft_times), budget.ttft_max_sec)
    print_distribution_table("5. LLM Total Generation", calculate_percentiles(gen_times), budget.generation_max_sec)
    print_distribution_table("6. End-to-End Latency", calculate_percentiles(e2e_times), budget.e2e_max_sec)

    print("\n--- THROUGHPUT & COST ---")
    tp_stats = calculate_percentiles(throughputs)
    cost_stats = calculate_percentiles(costs)
    print(f"Generation Throughput: Mean {tp_stats['mean']:.1f} tokens/s (P50: {tp_stats['p50']:.1f}, P95: {tp_stats['p95']:.1f})")
    print(f"Operational Cost:      Mean ${cost_stats['mean']:.5f} (Max: ${cost_stats['max']:.5f}, Budget: <=${budget.max_cost_usd:.4f})")

    # Budget Compliance
    budget_passes = sum(1 for r in successes if r["budget"]["passed"])
    print("\n--- SLA BUDGET COMPLIANCE ---")
    print(f"Full SLA Pass Rate: {budget_passes / len(successes):.1%} ({budget_passes}/{len(successes)} cases)")

    # Workload Segmentation Breakdown
    print("\n--- WORKLOAD SEGMENTATION (P50 & P95 E2E Latency) ---")
    tier_groups = {}
    for r in successes:
        tier = r["workload"]["size_tier"]
        tier_groups.setdefault(tier, []).append(r["component_latencies_sec"]["e2e_total"])

    for tier, latencies in tier_groups.items():
        stats = calculate_percentiles(latencies)
        print(f"  • {tier:<26} (n={stats['n']}): P50={stats['p50']:.3f}s | P95={stats['p95']:.3f}s")
    print("=" * 78)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Run first N rows only")
    args = parser.parse_args()

    budget = LatencyBudget()
    goldens = load_goldens(GOLDEN_PATH)
    if args.limit:
        goldens = goldens[:args.limit]

    print(f"\nStarting Operational Evaluation on {len(goldens)} golden items...\n")
    results = []

    for i, g in enumerate(goldens, 1):
        res = await evaluate_single_row(g, budget)
        results.append(res)

        # Write results progressively so a crash never loses progress
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        if res["status"] == "SUCCESS":
            comp = res["component_latencies_sec"]
            status_icon = "PASS" if res["budget"]["passed"] else "BREACH"
            print(f"[{i}/{len(goldens)}] [{status_icon}] E2E: {comp['e2e_total']:.2f}s "
                  f"(Embed: {comp['embedding']:.2f}s, Search: {comp['retrieval_search']:.2f}s, "
                  f"TTFT: {comp['ttft']:.2f}s, Gen: {comp['generation']:.2f}s) | {g['query'][:45]!r}")
            if not res["budget"]["passed"]:
                print(f"       -> Breaches: {', '.join(res['budget']['breaches'])}")
        else:
            print(f"[{i}/{len(goldens)}] [FAIL] Error: {res['error_type']} -> {res['error']}")

        if i < len(goldens):
            await asyncio.sleep(2)  # brief pause to respect provider rate limits

    print_comprehensive_summary(results, budget)


if __name__ == "__main__":
    asyncio.run(main())
