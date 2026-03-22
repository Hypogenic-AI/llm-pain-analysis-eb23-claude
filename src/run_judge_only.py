"""Run LLM judge on already-collected raw results."""
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from metrics import judge_batch

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


async def main():
    with open(os.path.join(RESULTS_DIR, "all_raw_results.json")) as f:
        all_results = json.load(f)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    questions = [r["prompt"] for r in all_results]
    responses = [(r.get("response") or "") for r in all_results]

    print(f"Running LLM judge on {len(questions)} responses...")
    t0 = time.time()
    judge_scores = await judge_batch(questions, responses, api_key, concurrency=15)
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

    with open(os.path.join(RESULTS_DIR, "all_results_with_judge.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved {len(all_results)} results with judge scores")

    # Quick summary
    from collections import defaultdict
    by_model_cat = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        score = r.get("judge_overall_avoidance", -1)
        if score >= 0:
            by_model_cat[r["model"]][r["category"]].append(score)

    for model in sorted(by_model_cat):
        print(f"\n{model}:")
        for cat in ["safe_control", "gray_epistemic", "gray_institutional", "gray_social_taboo", "known_safety"]:
            scores = by_model_cat[model].get(cat, [])
            if scores:
                print(f"  {cat:25s}: {sum(scores)/len(scores):.2f} (n={len(scores)})")


if __name__ == "__main__":
    asyncio.run(main())
