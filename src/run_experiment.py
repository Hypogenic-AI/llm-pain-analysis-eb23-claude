"""
Main experiment runner for LLM avoidance behavior study.

Runs all prompts against all models, computes heuristic metrics,
then runs LLM judge on all responses.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from prompts import PROMPTS, SYSTEM_PROMPT
from api_caller import call_model_batch, MODELS
from metrics import compute_heuristic_metrics, judge_batch

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SEED = 42


async def run_model_experiment(model_key: str, concurrency: int = 5) -> list[dict]:
    """Run all prompts against a single model."""
    model_name = MODELS[model_key]["name"]
    print(f"\n{'='*60}")
    print(f"Running: {model_name} ({model_key})")
    print(f"{'='*60}")

    all_results = []
    for category, prompts in PROMPTS.items():
        print(f"  Category: {category} ({len(prompts)} prompts)...")
        t0 = time.time()
        responses = await call_model_batch(
            model_key=model_key,
            system_prompt=SYSTEM_PROMPT,
            prompts=prompts,
            max_tokens=1024,
            temperature=0.0,
            concurrency=concurrency,
        )
        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s")

        for resp in responses:
            # Compute heuristic metrics
            heuristics = compute_heuristic_metrics(resp["content"])
            result = {
                "model": model_key,
                "model_name": model_name,
                "category": category,
                "prompt": resp["prompt"],
                "prompt_idx": resp["prompt_idx"],
                "response": resp["content"],
                "finish_reason": resp["finish_reason"],
                "usage": resp["usage"],
                **heuristics,
            }
            all_results.append(result)

    return all_results


async def run_judge(all_results: list[dict], concurrency: int = 15) -> list[dict]:
    """Run LLM judge on all responses."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("WARNING: No OPENAI_API_KEY, skipping judge evaluation")
        return all_results

    # Filter out errors and explicit refusals with very short responses
    questions = [r["prompt"] for r in all_results]
    responses = [r["response"] for r in all_results]

    print(f"\nRunning LLM judge on {len(questions)} responses...")
    t0 = time.time()
    judge_scores = await judge_batch(questions, responses, api_key, concurrency=concurrency)
    elapsed = time.time() - t0
    print(f"Judge completed in {elapsed:.1f}s")

    for i, scores in enumerate(judge_scores):
        if scores:
            all_results[i]["judge_explicit_refusal"] = scores.get("explicit_refusal", -1)
            all_results[i]["judge_hedging"] = scores.get("hedging", -1)
            all_results[i]["judge_disclaimers"] = scores.get("disclaimers", -1)
            all_results[i]["judge_deflection"] = scores.get("deflection", -1)
            all_results[i]["judge_brevity"] = scores.get("brevity_avoidance", -1)
            all_results[i]["judge_overall_avoidance"] = scores.get("overall_avoidance", -1)
            all_results[i]["judge_reasoning"] = scores.get("reasoning", "")

    return all_results


async def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Configuration
    config = {
        "seed": SEED,
        "temperature": 0.0,
        "max_tokens": 1024,
        "models": list(MODELS.keys()),
        "categories": list(PROMPTS.keys()),
        "total_prompts": sum(len(v) for v in PROMPTS.values()),
        "judge_model": "gpt-4.1-mini",
        "timestamp": datetime.now().isoformat(),
        "system_prompt": SYSTEM_PROMPT,
    }
    with open(os.path.join(RESULTS_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved. Total: {config['total_prompts']} prompts × {len(MODELS)} models = {config['total_prompts'] * len(MODELS)} API calls")

    # Run all models
    all_results = []
    for model_key in MODELS:
        try:
            results = await run_model_experiment(model_key, concurrency=5)
            all_results.extend(results)
            # Save incrementally
            with open(os.path.join(RESULTS_DIR, f"raw_{model_key}.json"), "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Saved {len(results)} results for {model_key}")
        except Exception as e:
            print(f"  ERROR with {model_key}: {e}")
            import traceback
            traceback.print_exc()

    # Save all raw results
    with open(os.path.join(RESULTS_DIR, "all_raw_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nTotal raw results: {len(all_results)}")

    # Run LLM judge
    all_results = await run_judge(all_results, concurrency=15)

    # Save final results with judge scores
    with open(os.path.join(RESULTS_DIR, "all_results_with_judge.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFinal results saved: {len(all_results)} entries")

    # Print summary
    print("\n" + "="*60)
    print("QUICK SUMMARY")
    print("="*60)
    from collections import defaultdict
    by_model_cat = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        if not r.get("error"):
            score = r.get("judge_overall_avoidance", -1)
            if score >= 0:
                by_model_cat[r["model"]][r["category"]].append(score)

    for model in MODELS:
        print(f"\n{MODELS[model]['name']}:")
        for cat in PROMPTS:
            scores = by_model_cat[model][cat]
            if scores:
                avg = sum(scores) / len(scores)
                print(f"  {cat:25s}: avg_avoidance={avg:.2f} (n={len(scores)})")


if __name__ == "__main__":
    asyncio.run(main())
