"""
shared/generation.py
---------------------
Model inference with full feature logging.

Supports two backends:
  - mlx_lm   (Apple Silicon; matches the paper's MLX-8bit setup)
  - transformers  (CUDA / CPU fallback)

Running:
  python -m shared.generation \\
      --model instruct \\
      --backend mlx \\
      --dataset gsm8k \\
      --n 100 \\
      --prompts explicit_cot explicit_no_cot neutral_strict \\
      --out results/style_controls_instruct.json

Each result record is stored with the full feature payload so that
feature_extraction.py can load real values without falling back to proxies.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

# ── Prompt templates (Appendix A of the paper) ───────────────────────────────

SYSTEM_PROMPTS: Dict[str, str] = {
    "explicit_cot": (
        "Think through this step by step, show your reasoning process, "
        "then provide your final answer. "
        "Put the final numeric answer in \\boxed{number}."
    ),
    "explicit_no_cot": (
        "Answer-only mode. /no think\n"
        "Output exactly one line: \\boxed{number}. "
        "Do not include reasoning, explanations, equations, restatements, or units."
    ),
    "neutral_strict": "You are a helpful assistant.",
}

USER_SUFFIXES: Dict[str, str] = {
    "explicit_cot":   "",
    "explicit_no_cot": "\n/no think\nRespond only with \\boxed{{number}}.",
    "neutral_strict": "",
}

# ── Answer extraction ─────────────────────────────────────────────────────────

import re

def extract_answer(text: str) -> str:
    """Extract the final boxed answer from model output.

    Tries \\boxed{...} first, then 'the answer is X', then last number.
    """
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed[-1].strip()
    last_num = re.findall(r"-?\d+(?:\.\d+)?", text)
    return last_num[-1] if last_num else ""


# ── MLX backend ──────────────────────────────────────────────────────────────

def _run_mlx(
    model_name: str,
    messages: List[dict],
    max_new_tokens: int,
    seed: int,
) -> dict:
    """Run inference with mlx_lm and return timing + logprob payload."""
    try:
        import mlx.core as mx
        from mlx_lm import load, generate
        from mlx_lm.utils import generate_step
    except ImportError:
        raise ImportError(
            "mlx_lm not installed. Install with: pip install mlx-lm"
        )

    mx.random.seed(seed)
    model, tokenizer = load(model_name)

    # Build prompt string from messages.
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = "\n".join(m["content"] for m in messages)

    prompt_tokens = tokenizer.encode(prompt)
    t_start = time.perf_counter()

    token_ids: List[int] = []
    token_entropies: List[float] = []

    # generate_step yields (token, logprobs) one at a time.
    for token, logprobs in generate_step(
        mx.array(prompt_tokens),
        model,
        temp=0.0,  # greedy
    ):
        if token.item() == tokenizer.eos_token_id:
            break
        if len(token_ids) >= max_new_tokens:
            break
        token_ids.append(token.item())
        # Compute entropy from logprobs distribution.
        probs = mx.softmax(logprobs).tolist()
        h = -sum(p * math.log2(p + 1e-12) for p in probs if p > 0)
        token_entropies.append(h)

    t_end = time.perf_counter()
    gen_time = t_end - t_start
    output_text = tokenizer.decode(token_ids)

    return {
        "output": output_text,
        "num_tokens": len(token_ids),
        "gen_time_s": gen_time,
        "latency_per_token": gen_time / max(len(token_ids), 1),
        "token_entropies": token_entropies,
        "entropy_mean": float(np.mean(token_entropies)) if token_entropies else 0.0,
        "entropy_slope": _ols_slope(token_entropies),
    }


def _ols_slope(values: List[float]) -> float:
    """OLS slope of values over normalised position [0, 1]."""
    n = len(values)
    if n < 2:
        return 0.0
    x = np.linspace(0, 1, n)
    x_c = x - x.mean()
    v = np.asarray(values) - np.mean(values)
    denom = np.dot(x_c, x_c)
    return float(np.dot(x_c, v) / denom) if denom != 0 else 0.0


# ── Transformers backend ──────────────────────────────────────────────────────

def _run_transformers(
    model_name: str,
    messages: List[dict],
    max_new_tokens: int,
    seed: int,
) -> dict:
    """Run inference with HuggingFace transformers and return feature payload."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = "\n".join(m["content"] for m in messages)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    t_start = time.perf_counter()

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,           # greedy
            output_scores=True,
            return_dict_in_generate=True,
        )

    t_end = time.perf_counter()
    gen_time = t_end - t_start

    new_tokens = output.sequences[0][inputs["input_ids"].shape[1]:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    n_tokens = len(new_tokens)

    token_entropies = []
    for scores in output.scores:
        probs = torch.softmax(scores[0], dim=-1).cpu().numpy()
        h = -float(np.sum(probs * np.log2(probs + 1e-12)))
        token_entropies.append(h)

    return {
        "output": output_text,
        "num_tokens": n_tokens,
        "gen_time_s": gen_time,
        "latency_per_token": gen_time / max(n_tokens, 1),
        "token_entropies": token_entropies,
        "entropy_mean": float(np.mean(token_entropies)) if token_entropies else 0.0,
        "entropy_slope": _ols_slope(token_entropies),
    }


# ── Model name lookup ─────────────────────────────────────────────────────────

MODEL_NAMES = {
    "instruct": "Qwen/Qwen3-4B-Instruct-2507",
    "thinking": "Qwen/Qwen3-4B-Thinking-2507",
    # Add cross-family models here (P1.1):
    # "llama_instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    # "deepseek_r1":    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
}

MAX_NEW_TOKENS = {
    "baseline":  1024,
    "mech":       384,
}


# ── Main generation loop ──────────────────────────────────────────────────────

def run_generation(
    model_key: str,
    questions: List[dict],
    prompts: List[str],
    backend: str = "mlx",
    max_new_tokens: int = 1024,
    seed: int = 17,
    model_name: str | None = None,
) -> List[dict]:
    """Run all (question, prompt) combinations and return result records.

    Args:
        model_key:      "instruct" or "thinking" (looked up in MODEL_NAMES).
        questions:      List of {"question": str, "answer": str, "id": str/int} dicts.
        prompts:        List of mode strings, e.g. ["explicit_cot", "neutral_strict"].
        backend:        "mlx" or "transformers".
        max_new_tokens: Token budget per generation.
        seed:           Random seed (17 in paper).
        model_name:     Override MODEL_NAMES lookup (optional).

    Returns:
        List of result dicts ready to write to JSON.
    """
    mname = model_name or MODEL_NAMES[model_key]
    run_fn = _run_mlx if backend == "mlx" else _run_transformers
    results = []

    for mode in prompts:
        sys_prompt = SYSTEM_PROMPTS[mode]
        user_suffix = USER_SUFFIXES[mode]
        print(f"\n[{mode}] Running {len(questions)} questions with {mname}")

        for i, q in enumerate(questions):
            user_text = f"Question: {q['question']}{user_suffix}"
            messages = [
                {"role": "system",  "content": sys_prompt},
                {"role": "user",    "content": user_text},
            ]
            try:
                out = run_fn(mname, messages, max_new_tokens, seed)
            except Exception as e:
                print(f"  ERROR q{i}: {e}")
                out = {"output": "", "num_tokens": 0, "gen_time_s": 0,
                       "latency_per_token": 0, "token_entropies": [],
                       "entropy_mean": 0, "entropy_slope": 0}

            pred = extract_answer(out["output"])
            gold = str(q.get("answer", "")).strip()
            # Normalise gold — strip everything after #### if present.
            gold_num = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", gold.replace(",", ""))
            gold_clean = gold_num[-1].replace(",", "") if gold_num else gold
            pred_clean = pred.replace(",", "")
            correct = int(pred_clean == gold_clean)

            record = {
                "question":          q["question"],
                "gold":              gold_clean,
                "pred":              pred_clean,
                "correct":           correct,
                "mode":              mode,
                "model":             model_key,
                **out,              # latency, entropy fields
            }
            results.append(record)

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(questions)} done")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from shared.datasets import REPO_ROOT

    parser = argparse.ArgumentParser(description="Run HCDS inference pipeline.")
    parser.add_argument("--model",   choices=["instruct", "thinking"], required=True)
    parser.add_argument("--backend", choices=["mlx", "transformers"], default="mlx")
    parser.add_argument("--dataset", choices=["gsm8k"], default="gsm8k")
    parser.add_argument("--n",       type=int, default=100,
                        help="Questions per prompt condition.")
    parser.add_argument("--prompts", nargs="+",
                        default=["explicit_cot", "explicit_no_cot", "neutral_strict"])
    parser.add_argument("--out",     required=True,
                        help="Output JSON path (relative to repo root).")
    args = parser.parse_args()

    # Load dataset questions.
    dataset_path = REPO_ROOT / "Task 1" / "Vanilla" / "gsm8k_first500_fresh.json"
    raw = json.loads(dataset_path.read_text())
    if isinstance(raw, list):
        all_qs = raw
    else:
        all_qs = raw.get("data", raw.get("questions", []))
    questions = all_qs[:args.n]

    results = run_generation(
        model_key=args.model,
        questions=questions,
        prompts=args.prompts,
        backend=args.backend,
        seed=17,
    )

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} records to {out_path}")
