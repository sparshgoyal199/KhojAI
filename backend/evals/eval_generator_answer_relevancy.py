# evals/eval_generator.py
"""
Component-level evaluation of the GENERATOR, in isolation.

Faithfulness: of the claims in the generated answer, how many are supported
by the context it was given? (Did the generator make things up?)

ISOLATION: we feed the generator the GOLDEN context (known-good chunks from
the faithfulness dataset), NOT the retriever's output. So a low score is
purely the generator's fault; the context was already correct.

Runs ONE test case at a time and judges it with the Groq-only FallbackJudge.
Results are saved to JSON after every case.

Usage:
    uv run python -m evals.eval_generator
    uv run python -m evals.eval_generator --limit 3      # quick iteration
"""
import argparse
import asyncio
import json
import time
import traceback

from dotenv import load_dotenv

load_dotenv()

from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from langchain_core.messages import HumanMessage, SystemMessage

from evals.fallback_judge import FallbackJudge
from evals.harness import load_goldens
from pipeline.prompt import system_prompt
from services.retrieval_service import creating_user_prompt, response_generator

GOLDEN_PATH = "goldens/faithfulness_dataset.json"
RESULTS_PATH = "goldens/generator_results.json"
THRESHOLD = 0.7
PAUSE_BETWEEN_CASES = 5  # seconds; raise to 10-15 if you still see 429s

judge = FallbackJudge()


async def build_test_cases(limit: int | None = None) -> list[LLMTestCase]:
    """Run the generator on the GOLDEN context and build one test case per golden."""
    goldens = load_goldens(GOLDEN_PATH)
    if limit:
        goldens = goldens[:limit]

    cases = []
    for g in goldens:
        context = g["content"]  # known-good context (list of chunk strings)
        if isinstance(context, str):
            context = [context]  # retrieval_context must be list[str]

        user_prompt = creating_user_prompt(context, g["query"], messages=None)
        answer = await response_generator(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=answer.content,        # the generated answer being judged
                retrieval_context=[dic['content'] for dic in context],   # faithfulness checks the answer against THIS
                # no expected_output: faithfulness never reads it
            )
        )
    return cases


def save(results: list[dict]):
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def print_summary(results: list[dict]):
    scored = [r["score"] for r in results if r["score"] is not None]
    print("\n=== generator: faithfulness ===")
    print(f"judge:         {judge.get_model_name()}")
    print(f"scored:        {len(scored)}/{len(results)}")
    if scored:
        print(f"avg faithful.: {sum(scored) / len(scored):.3f}")
        print(f"pass rate:     {sum(r['passed'] for r in results) / len(results):.0%} "
              f"(threshold {THRESHOLD})")
    for r in results:
        if r["error"]:
            print(f"FAILED: {r['query'][:60]!r} -> {r['error']}")
    print(f"details saved to {RESULTS_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only run first N goldens")
    args = parser.parse_args()

    # Phase 1 (async): generate answers. The loop is closed before judging starts.
    test_cases = asyncio.run(build_test_cases(args.limit))
    print(f"Generated {len(test_cases)} answer(s). Judging one at a time...\n")

    # Phase 2 (sync): judge sequentially.
    results: list[dict] = []
    for i, tc in enumerate(test_cases, 1):
        # fresh metric per case so no state leaks between runs
        metric = AnswerRelevancyMetric(
            threshold=THRESHOLD,    
            model=judge,
            include_reason=True,   # explains which claims were unsupported
            async_mode=False,
            verbose_mode=True,     # prints truths, claims and per-claim verdicts
        )
        started = time.monotonic()
        base = {
            "query": tc.input,
            "actual_output": tc.actual_output,
            "n_chunks": len(tc.retrieval_context),
        }
        try:
            metric.measure(tc)
            row = {
                **base,
                "score": metric.score,
                "passed": metric.is_successful(),
                "reason": metric.reason,
                "judge_model": judge.last_model,
                "error": None,
                "traceback": None,
            }
        except Exception as e:
            row = {
                **base,
                "score": None,
                "passed": False,
                "reason": None,
                "judge_model": judge.last_model,
                "error": repr(e),
                "traceback": traceback.format_exc(),
            }

        results.append(row)
        save(results)  # persist after every case so a crash loses nothing

        print(f"[{i}/{len(test_cases)}] {time.monotonic() - started:.0f}s "
              f"score={row['score']} passed={row['passed']} "
              f"chunks={row['n_chunks']} model={row['judge_model']}")
        print(f"   query:  {tc.input[:100]!r}")
        if row["error"]:
            print(f"   ERROR:  {row['error']}")
            print(row["traceback"])
        else:
            print(f"   reason: {row['reason']}")
        print()

        if i < len(test_cases):
            time.sleep(PAUSE_BETWEEN_CASES)

    print_summary(results)


if __name__ == "__main__":
    main()