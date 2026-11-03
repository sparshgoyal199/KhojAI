# evals/eval_retriever.py
"""
Sequential retriever eval (Contextual Recall).

Runs ONE test case at a time: the next case starts only after the previous
one has finished. Results are saved to JSON after every case.

Usage:
    uv run python -m evals.eval_retriever
    uv run python -m evals.eval_retriever --limit 3      # quick iteration
"""
import argparse
import asyncio
import json
import traceback  # add at top
import time

from dotenv import load_dotenv

load_dotenv()

import modal
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase

from evals.fallback_judge import FallbackJudge
from evals.harness import load_goldens
from services.retrieval_service import retrieve_relevant_chunks

parsing_and_embedding_model = modal.Cls.from_name(
    "parsing_and_embedding_generator", "ParsingEmbeddingModel"
)

GOLDEN_PATH = "goldens/retriever_goldens.json"
RESULTS_PATH = "goldens/retriever_results.json"
DOC_ID = "0d7887ca44684c71b407bfd8842a720c"
THRESHOLD = 0.7
PAUSE_BETWEEN_CASES = 5  # seconds; raise to 10-15 if you still see 429s

judge = FallbackJudge()


async def build_test_cases(limit: int | None = None) -> list[LLMTestCase]:
    goldens = load_goldens(GOLDEN_PATH)
    if limit:
        goldens = goldens[:limit]

    modal_obj = parsing_and_embedding_model()
    cases = []
    for g in goldens:
        emb = await modal_obj.embed_query.remote.aio(g["query"])
        retrieved = await retrieve_relevant_chunks(DOC_ID, emb, g["query"])
        cases.append(
            LLMTestCase(
                input=g["query"],
                expected_output=g["ideal_answer"],
                retrieval_context=[d["content"] for d in retrieved],
                actual_output="(generator not evaluated in this run)",
            )
        )
    return cases


def save(results: list[dict]):
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def print_summary(results: list[dict]):
    scored = [r["score"] for r in results if r["score"] is not None]
    print("\n=== retriever: contextual recall ===")
    print(f"judge:      {judge.get_model_name()}")
    print(f"scored:     {len(scored)}/{len(results)}")
    if scored:
        print(f"avg recall: {sum(scored) / len(scored):.3f}")
        print(f"pass rate:  {sum(r['passed'] for r in results) / len(results):.0%} "
              f"(threshold {THRESHOLD})")
    for r in results:
        if r["error"]:
            print(f"FAILED: {r['query'][:60]!r} -> {r['error']}")
    print(f"details saved to {RESULTS_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only run first N goldens")
    args = parser.parse_args()

    test_cases = asyncio.run(build_test_cases(args.limit))  # loop closed before judging
    print(f"Built {len(test_cases)} test case(s). Judging one at a time...\n")

    results: list[dict] = []
    for i, tc in enumerate(test_cases, 1):
        # fresh metric per case so no state leaks between runs
        metric = ContextualPrecisionMetric(
            threshold=THRESHOLD,
            model=judge,
            include_reason=True,
            verbose_mode=True,
            async_mode=False,
        )
        started = time.monotonic()
        try:
            metric.measure(tc)
            row = {
                "query": tc.input,
                "score": metric.score,
                "passed": metric.is_successful(),
                "reason": metric.reason,
                "judge_model": judge.last_model,
                "error": None,
            }
        except Exception as e:
            row = {
                "query": tc.input,
                "score": None,
                "passed": False,
                "reason": None,
                "traceback": traceback.format_exc(),
                "judge_model": judge.last_model,
                "error": repr(e)[:300],
            }

        results.append(row)
        save(results)  # persist after every case so a crash loses nothing
        print(
            f"[{i}/{len(test_cases)}] {time.monotonic() - started:.0f}s "
            f"score={row['score']} passed={row['passed']} "
            f"model={row['judge_model']} | {tc.input[:60]!r}\n"
        )

        if i < len(test_cases):
            time.sleep(PAUSE_BETWEEN_CASES)

    print_summary(results)


if __name__ == "__main__":
    main()