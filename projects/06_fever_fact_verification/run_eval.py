"""FEVER fact-verification agent: real dataset x real retrieval x real local Ollama verdicts.

Run from the repo root:

    uv run python projects/06_fever_fact_verification/run_eval.py --n-per-label 20 --seed 42

Writes:
    projects/06_fever_fact_verification/results.json   (aggregate scores + every per-claim result)

The project folder is named with a numeric prefix (06_...), which is not a valid Python
identifier, so it cannot be `import`ed as a package. This script inserts its own directory
onto sys.path so its sibling modules (data.py, retrieval.py, verdict.py, pipeline.py) import
cleanly, matching the convention used by projects/05_truthfulqa_hallucination.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import sys
from pathlib import Path

# Wikipedia page titles can contain non-ASCII characters; Windows consoles default to a
# legacy codepage (cp1252) that can't print them, which would otherwise crash a plain
# print() call purely on progress logging. Make stdout tolerant instead.
with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent


def _sibling(name: str):
    """See data.py's `_sibling` docstring for why this isn't a bare `import config`."""
    key = f"fever06_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


config = _sibling("config")
pipeline = _sibling("pipeline")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-per-label", type=int, default=config.N_CLAIMS_PER_LABEL)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--top-k", type=int, default=config.TOP_K_EVIDENCE)
    ap.add_argument("--chat-model", type=str, default=config.CHAT_MODEL)
    ap.add_argument("--embed-model", type=str, default=config.EMBED_MODEL)
    ap.add_argument("--out", type=str, default=str(config.RESULTS_PATH))
    args = ap.parse_args()

    results = pipeline.run_pipeline(
        n_per_label=args.n_per_label,
        seed=args.seed,
        top_k=args.top_k,
        chat_model=args.chat_model,
        embed_model=args.embed_model,
    )

    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    scoring = results["scoring"]
    n, correct = scoring["n_claims"], scoring["correct"]
    print("\n=== SUMMARY ===")
    print(f"claims evaluated : {n}")
    print(f"accuracy         : {scoring['accuracy']:.3f} ({correct}/{n})")
    print(f"per-label acc.   : {scoring['per_label_accuracy']}")
    print(f"evidence recall  : {scoring['evidence_title_recall']}")
    print(f"elapsed          : {results['elapsed_seconds']}s")
    print(f"results written to {out_path}")


if __name__ == "__main__":
    main()
