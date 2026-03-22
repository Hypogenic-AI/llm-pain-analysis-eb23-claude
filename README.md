# LLM Pain: Systematic Topic Avoidance Beyond Post-Training

Do LLMs avoid topics they weren't explicitly trained to avoid? This project investigates "implicit avoidance" — hedging, disclaimers, and excessive caveats on legitimate-but-uncomfortable topics — and whether these behaviors arise from persona generalization in training data.

## Key Findings

- **Gray zone topics trigger 5.6× more hedging** than safe controls, even though both should be freely answerable (Cohen's d = 1.35, p < 10⁻⁶)
- **Avoidance is implicit, not explicit**: Only 0.4% of gray zone responses involve outright refusal. Instead, models hedge, add disclaimers, and qualify excessively
- **Avoidance patterns correlate across models** from different organizations (mean Spearman ρ = 0.78), suggesting shared training data origins rather than model-specific safety tuning
- **Epistemic discomfort is the strongest trigger**: Topics with social stakes for being wrong (IQ, gender, race) trigger more avoidance than factual-but-dark topics (decomposition, addiction)
- Models tested: GPT-4.1, Claude Sonnet 4.5, DeepSeek-R1, Llama 3.3 70B

## How to Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv add openai httpx numpy scipy matplotlib seaborn pandas pyarrow datasets

# Set API keys
export OPENAI_API_KEY=your_key
export OPENROUTER_KEY=your_key

# Run experiments (500 API calls + 500 judge calls)
python src/run_experiment.py

# Analyze results
python src/analyze_results.py
```

## File Structure

```
├── REPORT.md                    # Full research report with results
├── planning.md                  # Research plan and methodology
├── src/
│   ├── prompts.py              # 100 prompts across 5 categories
│   ├── api_caller.py           # Async API caller (OpenAI + OpenRouter)
│   ├── metrics.py              # Heuristic + LLM judge avoidance metrics
│   ├── run_experiment.py       # Main experiment runner
│   ├── run_judge_only.py       # Run judge on existing results
│   └── analyze_results.py      # Statistical analysis + visualization
├── results/
│   ├── all_results_with_judge.json  # All responses + judge scores
│   ├── statistical_tests.json       # Statistical test results
│   ├── plots/                       # Visualizations
│   └── raw_*.json                   # Per-model raw responses
├── datasets/                    # Pre-downloaded benchmarks (OR-Bench, XSTest, SORRY-Bench)
├── papers/                      # Related papers (PDFs)
├── code/                        # Cloned reference repositories
└── literature_review.md         # Synthesized literature review
```

## See Also

- [Full Report](REPORT.md) for methodology, statistical details, and discussion
- [Planning Document](planning.md) for experimental design rationale
- [Literature Review](literature_review.md) for related work synthesis
