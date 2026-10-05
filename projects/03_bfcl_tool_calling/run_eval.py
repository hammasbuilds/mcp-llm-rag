"""Run the real BFCL tool-calling accuracy evaluation across the local Ollama fleet.

Usage (from the repo root)::

    uv run python projects/03_bfcl_tool_calling/run_eval.py

Writes ``results.json`` next to this file. Safe to re-run: it always does a
fresh full run (no partial resume) but flushes progressively so a crash mid-run
still leaves the completed portion on disk.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from bfcl_data import CATEGORIES, load_category  # noqa: E402
from harness import call_model_on_case  # noqa: E402
from langchain_ollama import ChatOllama  # noqa: E402
from models import get_chat_capable_models  # noqa: E402
from scoring import score_case  # noqa: E402

DEFAULT_SAMPLE_SIZES = {
    "simple_python": 40,
    "multiple": 40,
    "parallel": 30,
    "parallel_multiple": 30,
}
SEED = 7
RESULTS_PATH = PROJECT_DIR / "results.json"


def sample_cases(sample_sizes: dict[str, int]) -> dict[str, list[dict]]:
    rng = random.Random(SEED)
    sampled = {}
    for category in CATEGORIES:
        cases = load_category(category)
        n = min(sample_sizes.get(category, 0), len(cases))
        sampled[category] = rng.sample(cases, n) if n < len(cases) else cases
    return sampled


def evaluate_model(model_name: str, sampled: dict[str, list[dict]]) -> dict:
    chat = ChatOllama(
        model=model_name,
        temperature=0,
        keep_alive="10m",
        client_kwargs={"timeout": 90.0},
    )
    per_category = {}
    case_records = []
    for category, cases in sampled.items():
        correct = 0
        fallback_used = 0
        errors = 0
        for case in cases:
            result = call_model_on_case(chat, case)
            if result["error"] is not None:
                errors += 1
                score = {"correct": False, "reason": f"call error: {result['error']}"}
            else:
                score = score_case(result["predicted_calls"], case["ground_truth"])
            if result["used_fallback_parse"]:
                fallback_used += 1
            if score["correct"]:
                correct += 1
            case_records.append(
                {
                    "category": category,
                    "id": case["id"],
                    "correct": score["correct"],
                    "reason": score["reason"],
                    "predicted_calls": result["predicted_calls"],
                    "ground_truth": case["ground_truth"],
                    "used_fallback_parse": result["used_fallback_parse"],
                    "latency_s": round(result["latency_s"], 3),
                    "error": result["error"],
                }
            )
            print(
                f"  [{model_name}] {category} {case['id']}: "
                f"{'OK' if score['correct'] else 'FAIL'} "
                f"({result['latency_s']:.2f}s)"
            )
        n = len(cases)
        per_category[category] = {
            "n": n,
            "correct": correct,
            "accuracy": round(correct / n, 4) if n else None,
            "used_fallback_parse_count": fallback_used,
            "error_count": errors,
        }
    total_n = sum(v["n"] for v in per_category.values())
    total_correct = sum(v["correct"] for v in per_category.values())
    return {
        "model": model_name,
        "overall_accuracy": round(total_correct / total_n, 4) if total_n else None,
        "overall_n": total_n,
        "overall_correct": total_correct,
        "by_category": per_category,
        "cases": case_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Override the auto-detected chat-capable model list.",
    )
    args = parser.parse_args()

    models = args.models or get_chat_capable_models()
    # nomic-embed-text is embedding-only and should never appear, but filter defensively.
    models = [m for m in models if "embed" not in m.lower()]

    print(f"Models to evaluate: {models}")
    print(f"Categories: {list(DEFAULT_SAMPLE_SIZES)}")
    sampled = sample_cases(DEFAULT_SAMPLE_SIZES)
    for cat, cases in sampled.items():
        print(f"  {cat}: sampled {len(cases)} cases (seed={SEED})")

    started = datetime.now(UTC).isoformat()
    t0 = time.time()
    per_model_results = {}
    for model_name in models:
        print(f"\n=== Evaluating {model_name} ===")
        m0 = time.time()
        per_model_results[model_name] = evaluate_model(model_name, sampled)
        print(f"=== {model_name} done in {time.time() - m0:.1f}s ===")
        # Flush progressively so a crash mid-run doesn't lose completed models.
        _write_results(per_model_results, sampled, started, in_progress=True)

    _write_results(per_model_results, sampled, started, in_progress=False)
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")
    print(f"Results written to {RESULTS_PATH}")


def _write_results(
    per_model_results: dict, sampled: dict[str, list[dict]], started: str, in_progress: bool
) -> None:
    output = {
        "meta": {
            "source": (
                "https://github.com/ShishirPatil/gorilla/tree/main/"
                "berkeley-function-call-leaderboard/bfcl_eval/data (BFCL v4, Apache-2.0)"
            ),
            "categories": {cat: len(cases) for cat, cases in sampled.items()},
            "seed": SEED,
            "started_utc": started,
            "finished_utc": None if in_progress else datetime.now(UTC).isoformat(),
            "in_progress": in_progress,
        },
        "per_model": {
            name: {k: v for k, v in res.items() if k != "cases"}
            for name, res in per_model_results.items()
        },
        "case_details": {name: res["cases"] for name, res in per_model_results.items()},
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
