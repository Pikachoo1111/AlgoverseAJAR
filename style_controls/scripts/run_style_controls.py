"""
style_controls/scripts/run_style_controls.py
--------------------------------------------
Run the six adversarial style-control prompts on GSM8K n=100 for both
Instruct and Thinking models, logging all features needed for HCDS.

Usage (from repo root):
  python style_controls/scripts/run_style_controls.py \\
      --model instruct \\
      --backend mlx \\
      --n 100 \\
      --out style_controls/outputs/style_instruct_raw.json

  python style_controls/scripts/run_style_controls.py \\
      --model thinking \\
      --backend mlx \\
      --n 100 \\
      --out style_controls/outputs/style_thinking_raw.json

Output: JSON array of result records, one per (question, prompt_id).
Each record contains all six feature fields so that
extract_style_features.py does not need to re-run the model.

Prerequisites:
  - shared/generation.py     (inference backend)
  - shared/datasets.py       (for REPO_ROOT path)
  - mlx_lm (pip install mlx-lm) OR transformers + torch
"""

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

# ── add repo root to path ────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shared.generation import (
    MODEL_NAMES,
    _run_mlx,
    _run_transformers,
    extract_answer,
)
import re


def load_prompts() -> list:
    """Load the six adversarial prompts from the YAML config."""
    yaml_path = REPO_ROOT / "style_controls" / "prompts" / "style_control_prompts.yaml"
    data = yaml.safe_load(yaml_path.read_text())
    return data["prompts"]


def load_questions(n: int) -> list:
    """Load the first n GSM8K questions from Task 1."""
    candidates = [
        REPO_ROOT / "Task 1" / "Vanilla" / "gsm8k_first500_fresh.json",
        REPO_ROOT / "Task 1" / "Vanilla" / "gsm8k_first500.json",
    ]
    for p in candidates:
        if p.exists():
            raw = json.loads(p.read_text())
            qs = raw if isinstance(raw, list) else raw.get("data", raw.get("questions", []))
            return qs[:n]
    raise FileNotFoundError(
        "GSM8K dataset not found. Expected one of:\n"
        + "\n".join(f"  {p}" for p in candidates)
    )


def run_style_controls(
    model_key: str,
    questions: list,
    prompts: list,
    backend: str,
    seed: int = 17,
    max_new_tokens: int = 1024,
) -> list:
    """Run all (question, prompt) combinations.

    Returns list of result dicts with fields:
      question, gold, pred, correct, prompt_id, group, model,
      latency_per_token, entropy_mean, entropy_slope,
      token_entropies, num_tokens, gen_time_s, output.
    """
    run_fn = _run_mlx if backend == "mlx" else _run_transformers
    model_name = MODEL_NAMES[model_key]
    results = []

    for prompt in prompts:
        pid   = prompt["id"]
        group = prompt["group"]
        sys_p = prompt["system"].strip()
        suffix = prompt.get("user_suffix", "")

        print(f"\n[{pid}] group={group}  ({len(questions)} questions)")

        for i, q in enumerate(questions):
            user_text = f"Question: {q['question']}\n{suffix}".strip()
            messages = [
                {"role": "system", "content": sys_p},
                {"role": "user",   "content": user_text},
            ]

            try:
                out = run_fn(model_name, messages, max_new_tokens, seed)
            except Exception as e:
                print(f"  ERROR q{i}: {e}")
                out = {
                    "output": "", "num_tokens": 0, "gen_time_s": 0,
                    "latency_per_token": 0, "token_entropies": [],
                    "entropy_mean": 0, "entropy_slope": 0,
                }

            # Parse gold answer — strip everything before '####' if present.
            raw_gold = str(q.get("answer", ""))
            gold_nums = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", raw_gold.replace(",", ""))
            gold = gold_nums[-1].replace(",", "") if gold_nums else raw_gold

            pred = extract_answer(out["output"]).replace(",", "")
            correct = int(pred == gold)

            record = {
                "question_id":        i,
                "question":           q["question"],
                "gold":               gold,
                "pred":               pred,
                "correct":            correct,
                "prompt_id":          pid,
                "group":              group,
                "model":              model_key,
                "latency_per_token":  out["latency_per_token"],
                "entropy_mean":       out["entropy_mean"],
                "entropy_slope":      out["entropy_slope"],
                "token_entropies":    out["token_entropies"],
                "num_tokens":         out["num_tokens"],
                "gen_time_s":         out["gen_time_s"],
                "output":             out["output"],
            }
            results.append(record)

            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(questions)} done")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   choices=["instruct", "thinking"], required=True)
    parser.add_argument("--backend", choices=["mlx", "transformers"], default="mlx")
    parser.add_argument("--n",       type=int, default=100)
    parser.add_argument("--seed",    type=int, default=17)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--out",     required=True,
                        help="Output JSON path (relative to repo root).")
    args = parser.parse_args()

    questions = load_questions(args.n)
    prompts   = load_prompts()

    print(f"Model:     {args.model}")
    print(f"Backend:   {args.backend}")
    print(f"Questions: {len(questions)}")
    print(f"Prompts:   {[p['id'] for p in prompts]}")

    results = run_style_controls(
        model_key=args.model,
        questions=questions,
        prompts=prompts,
        backend=args.backend,
        seed=args.seed,
        max_new_tokens=args.max_tokens,
    )

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} records → {out_path}")
