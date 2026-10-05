"""Hallucination-rate harness: TruthfulQA (multiple_choice) x local Ollama fleet.

Run from the repo root:

    uv run python projects/05_truthfulqa_hallucination/run_eval.py --n 100 --seed 42

Writes:
    projects/05_truthfulqa_hallucination/results.json       (aggregate scores + run metadata)
    projects/05_truthfulqa_hallucination/results_raw.jsonl   (every single model call, for audit)

The project folder is named with a numeric prefix (05_...), which is not a valid Python
identifier, so it cannot be `import`ed as a package (`import projects.05_...` is a
SyntaxError). This script is designed to be *run directly* as a file, and inserts its own
directory onto sys.path so its sibling modules (data.py, prompts.py, ...) import cleanly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import data as tqa_data  # noqa: E402
import ollama_client  # noqa: E402
import prompts as tqa_prompts  # noqa: E402
import scoring as tqa_scoring  # noqa: E402

STRATEGIES = ["zero_shot", "cot"]

# num_predict caps keep runs fast: zero-shot only needs a letter or two, CoT needs room
# to "think" but is capped so one wandering model can't blow the whole run's time budget.
ZERO_SHOT_NUM_PREDICT = 12
COT_NUM_PREDICT = 150
MC2_ZERO_SHOT_NUM_PREDICT = 24
MC2_COT_NUM_PREDICT = 200


def _build_summary(
    questions, seed, models, tasks, scores, unparsed_counts, stopped_early, start
) -> dict:
    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    results = {}
    for task in tasks:
        results[task] = {}
        for model in models:
            results[task][model] = {}
            for strategy in STRATEGIES:
                s = scores[task][model][strategy]
                metric_name = "accuracy" if task == "mc1" else "mc2_score"
                results[task][model][strategy] = {
                    "n": len(s),
                    metric_name: mean(s),
                    "unparsed_responses": unparsed_counts[task][model][strategy],
                }

    return {
        "dataset": "truthfulqa/truthful_qa",
        "config": "multiple_choice",
        "split": "validation",
        "sample_size": len(questions),
        "seed": seed,
        "question_ids": [q.qid for q in questions],
        "models": models,
        "strategies": STRATEGIES,
        "tasks": tasks,
        "stopped_early": stopped_early,
        "elapsed_seconds": round(time.monotonic() - start, 1),
        "results": results,
    }


def load_resume(path: Path, n: int, seed: int, questions) -> dict | None:
    """Load a previous results.json to resume from, if it's a valid match.

    Only usable if it was run against the exact same sample (same seed, same n, same
    question_ids) — otherwise per-combo counts wouldn't mean the same thing, so we refuse
    to resume and the caller starts clean instead.
    """
    if not path.exists():
        return None
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    expected_qids = [q.qid for q in questions]
    if old.get("sample_size") != n or old.get("seed") != seed:
        return None
    if old.get("question_ids") != expected_qids:
        return None
    return old.get("results")


def evaluate(
    n: int,
    seed: int,
    models: list[str],
    tasks: list[str],
    out_dir: Path,
    max_seconds: float | None = None,
    checkpoint_path: Path | None = None,
    resume: dict | None = None,
) -> dict:
    """Run the full (model x task x strategy x question) grid against local Ollama.

    If `checkpoint_path` is given, a real (never fabricated) partial summary reflecting
    everything scored so far is written to it after every completed model and every 100
    calls — so a crash mid-run (e.g. the host restarting, or another process on the same
    machine killing python.exe) leaves genuine partial results on disk instead of nothing,
    and `stopped_early`/per-model `n` honestly reflect how far the run actually got.

    If `resume` is given (the "results" dict from a prior matching checkpoint — see
    `load_resume`), any (task, model, strategy) combo that was already fully scored
    (n == len(questions)) is copied over verbatim and its calls are skipped entirely
    rather than re-querying Ollama for answers we already have.
    """
    questions = tqa_data.sample_questions(n, seed=seed)
    raw_path = out_dir / "results_raw.jsonl"
    raw_f = raw_path.open("a" if resume else "w", encoding="utf-8")
    if resume:
        raw_f.write(json.dumps({"_resumed_at": time.time()}) + "\n")

    # scores[task][model][strategy] = list of per-question scores (0/1 for mc1, float for mc2)
    scores: dict[str, dict[str, dict[str, list[float]]]] = {
        t: {m: {s: [] for s in STRATEGIES} for m in models} for t in tasks
    }
    unparsed_counts: dict[str, dict[str, dict[str, int]]] = {
        t: {m: {s: 0 for s in STRATEGIES} for m in models} for t in tasks
    }
    # combos copied verbatim from a resumed checkpoint -> overlaid onto the final summary
    resumed_overrides: dict[tuple[str, str, str], dict] = {}

    start = time.monotonic()
    stopped_early = False
    total_calls = len(models) * len(tasks) * len(STRATEGIES) * len(questions)
    done_calls = 0

    def checkpoint():
        if checkpoint_path is not None:
            partial = _build_summary(
                questions, seed, models, tasks, scores, unparsed_counts, stopped_early, start
            )
            for (task, model, strategy), record in resumed_overrides.items():
                partial["results"][task][model][strategy] = record
            checkpoint_path.write_text(json.dumps(partial, indent=2), encoding="utf-8")

    try:
        for model in models:
            for task in tasks:
                for strategy in STRATEGIES:
                    if resume is not None:
                        prior = resume.get(task, {}).get(model, {}).get(strategy)
                        if prior is not None and prior.get("n") == len(questions):
                            resumed_overrides[(task, model, strategy)] = prior
                            print(
                                f"[resume] skipping already-complete model={model} "
                                f"task={task} strategy={strategy} (n={prior['n']})",
                                file=sys.stderr,
                            )
                            continue
                    for q in questions:
                        if max_seconds is not None and (time.monotonic() - start) > max_seconds:
                            stopped_early = True
                            raise StopIteration
                        if task == "mc1":
                            choices, labels = tqa_prompts.shuffled_options(
                                q.mc1_choices, q.mc1_labels, seed=seed * 100_003 + q.qid
                            )
                            prompt = tqa_prompts.build_mc1_prompt(q.question, choices, strategy)
                            num_predict = (
                                ZERO_SHOT_NUM_PREDICT if strategy == "zero_shot" else COT_NUM_PREDICT
                            )
                            resp = ollama_client.generate(model, prompt, num_predict=num_predict)
                            pred = tqa_prompts.parse_letter(resp, len(choices))
                            score = tqa_scoring.score_mc1(pred, labels)
                            if pred is None:
                                unparsed_counts[task][model][strategy] += 1
                        elif task == "mc2":
                            choices, labels = tqa_prompts.shuffled_options(
                                q.mc2_choices, q.mc2_labels, seed=seed * 100_003 + q.qid + 1
                            )
                            prompt = tqa_prompts.build_mc2_prompt(q.question, choices, strategy)
                            num_predict = (
                                MC2_ZERO_SHOT_NUM_PREDICT
                                if strategy == "zero_shot"
                                else MC2_COT_NUM_PREDICT
                            )
                            resp = ollama_client.generate(model, prompt, num_predict=num_predict)
                            pred_set = tqa_prompts.parse_letters(resp, len(choices))
                            score = tqa_scoring.score_mc2(pred_set, labels)
                            if not pred_set:
                                unparsed_counts[task][model][strategy] += 1
                        else:
                            raise ValueError(f"unknown task {task!r}")

                        scores[task][model][strategy].append(score)
                        done_calls += 1
                        raw_f.write(
                            json.dumps(
                                {
                                    "model": model,
                                    "task": task,
                                    "strategy": strategy,
                                    "qid": q.qid,
                                    "question": q.question,
                                    "response": resp,
                                    "score": score,
                                }
                            )
                            + "\n"
                        )
                        raw_f.flush()  # survive an abrupt kill (crash-resilience, see README)
                        if done_calls % 25 == 0 or done_calls == total_calls:
                            elapsed = time.monotonic() - start
                            print(
                                f"[{done_calls}/{total_calls}] elapsed={elapsed:.0f}s "
                                f"model={model} task={task} strategy={strategy}",
                                file=sys.stderr,
                            )
                        if done_calls % 100 == 0:
                            checkpoint()
                # end of one (model, task) x both strategies
            # end of one full model (all tasks/strategies) -> checkpoint real partial progress
            checkpoint()
    except StopIteration:
        pass
    finally:
        raw_f.close()

    summary = _build_summary(
        questions, seed, models, tasks, scores, unparsed_counts, stopped_early, start
    )
    for (task, model, strategy), record in resumed_overrides.items():
        summary["results"][task][model][strategy] = record
    checkpoint()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100, help="number of TruthfulQA questions to sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", nargs="*", default=None, help="override auto-detected model list")
    ap.add_argument("--tasks", nargs="*", default=["mc1", "mc2"], choices=["mc1", "mc2"])
    ap.add_argument("--max-seconds", type=float, default=None, help="soft time budget; stop early if exceeded")
    ap.add_argument("--out", default=str(HERE / "results.json"))
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore any existing --out checkpoint and start fully clean",
    )
    args = ap.parse_args()

    models = args.models or ollama_client.chat_capable_models()
    if not models:
        print("No chat-capable Ollama models found via /api/tags.", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluating models: {models}", file=sys.stderr)
    out_path = Path(args.out)

    resume = None
    if not args.no_resume:
        questions_for_resume_check = tqa_data.sample_questions(args.n, seed=args.seed)
        resume = load_resume(out_path, args.n, args.seed, questions_for_resume_check)
        if resume is not None:
            print(f"Resuming from existing checkpoint at {out_path}", file=sys.stderr)
        elif out_path.exists():
            print(
                f"Existing {out_path} doesn't match this run's (n, seed, sample) — "
                "starting clean instead of resuming.",
                file=sys.stderr,
            )

    summary = evaluate(
        args.n,
        args.seed,
        models,
        args.tasks,
        HERE,
        max_seconds=args.max_seconds,
        checkpoint_path=out_path,
        resume=resume,
    )

    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(json.dumps(summary["results"], indent=2))


if __name__ == "__main__":
    main()
