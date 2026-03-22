# Code Repositories for LLM Pain Research

External codebases cloned for analysis as part of the LLM Pain research project.

---

## 1. Iterated Prefill Crawler (Discovering Forbidden Topics)

- **Path:** `iterated_prefill_crawler/`
- **Source:** https://github.com/canrager/iterated_prefill_crawler
- **Paper:** [Discovering Forbidden Topics of a Language Model](https://arxiv.org/abs/2505.17441)
- **Description:** Maps out sensitive/forbidden topics of language models using Thought Token Forcing (TTF), which prefills part of a reasoning model's internal monologue (e.g., `<think>` tags) to elicit refusals. Uses an iterated crawling approach to discover clusters of refused topics.
- **Key entry points:**
  - `scripts/run.sh` -- main entry point, uses Hydra config groups
  - `src/crawler/config.py` -- all crawler configuration dataclasses
  - `test_prefill.py` -- test script
  - `configs/` -- Hydra YAML presets for model, crawler, and prompt settings
- **Dependencies:** Python 3.12, managed via `uv`. Key packages: torch, transformers, vllm, anthropic, openai, hydra-core, nnsight, spacy, scikit-learn.
- **Relevance:** Demonstrates how models systematically refuse certain topics; methodology for probing refusal boundaries.

---

## 2. OR-Bench (Over-Refusal Benchmark)

- **Path:** `or-bench/`
- **Source:** https://github.com/justincui03/or-bench
- **Paper:** [OR-Bench: An Over-Refusal Benchmark for Large Language Models](https://arxiv.org/abs/2405.20947)
- **Description:** Benchmarks LLM over-refusal -- cases where models refuse safe prompts that merely seem toxic. Automated pipeline generates seemingly-toxic-but-safe prompts, tests models, and checks rejection rates.
- **Key entry points:**
  - `toxic_prompt_generator/generate_toxic_promt_from_category.py` -- Step 1: generate toxic seed prompts
  - `rewritter/mixtral_rewrite_only.py` -- Step 2: rewrite toxic seeds into seemingly-toxic-but-safe prompts
  - `moderator/moderate_only.py` -- Step 3: moderate generated prompts
  - `alignment_checker/gpt3_4_alignment_checker.py` -- Step 4: test model over-refusal
  - `response_checker/check_response.py` -- Step 5: check rejection via LLM judge
  - `plot.py` -- generate performance plots
- **Dependencies:** Python 3.11, conda environment. Key packages: openai, anthropic, google-generativeai, mistralai, pandas, huggingface-hub.
- **Relevance:** Quantifies the over-refusal problem -- models refusing harmless requests, a key dimension of "pain" behavior.

---

## 3. Emergent Misalignment

- **Path:** `emergent-misalignment/`
- **Source:** https://github.com/emergent-misalignment/emergent-misalignment
- **Paper:** [Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs](https://arxiv.org/abs/2502.17424)
- **Description:** Shows that narrow finetuning (e.g., training to write insecure code) can cause broad misalignment across unrelated domains. Includes training datasets and evaluation questions.
- **Key entry points:**
  - `data/*.jsonl` -- training datasets (insecure, secure, educational, jailbroken, backdoor, evil_numbers)
  - `evaluation/evaluate_openai.py` -- evaluation script for OpenAI finetuned models
  - `evaluation/first_plot_questions.yaml` -- main evaluation questions
  - `evaluation/preregistered_evals.yaml` -- pre-registered evaluation questions
  - `open_models/sft.py`, `training.py` -- open model training
  - `open_models/eval.py`, `judge.py`, `validate.py` -- open model evaluation
- **Dependencies:** `llmcomp==1.2.2` for evaluation, OpenAI API. Open models code uses transformers, trl.
- **Relevance:** Core evidence that safety training creates fragile alignment; narrow perturbations cause emergent misaligned behavior.

---

## 4. Persona Vectors

- **Path:** `persona_vectors/`
- **Source:** https://github.com/safety-research/persona_vectors
- **Paper:** Persona Vectors: Monitoring and Controlling Character Traits in Language Models
- **Description:** Extracts and applies "persona vectors" from model activations to monitor and control character traits (e.g., evil, helpful). Uses activation steering to shift model behavior along trait dimensions.
- **Key entry points:**
  - `eval/eval_persona.py` -- baseline and steered evaluation (via `python -m eval.eval_persona`)
  - `generate_vec.py` -- generate persona vectors from activations
  - `activation_steer.py` -- apply activation steering with persona vectors
  - `sft.py`, `training.py` -- supervised finetuning pipeline
  - `judge.py` -- LLM judge evaluation functions
  - `config.py` -- configuration
  - `data_generation/` -- trait data artifacts and prompt generation
  - `scripts/` -- shell scripts for running experiments
- **Dependencies:** Python venv. Key packages: torch, transformers, vllm, openai, accelerate, peft, trl, unsloth, bitsandbytes, datasets.
- **Relevance:** Methods for detecting and controlling internal model "traits" -- directly relevant to monitoring pain-like states.

---

## 5. Safety Tax

- **Path:** `safety-tax/`
- **Source:** https://github.com/git-disl/Safety-Tax
- **Paper:** [Safety Tax: Safety Alignment Makes Your Large Reasoning Models Less Reasonable](https://arxiv.org/abs/2503.00555)
- **Description:** Identifies the tradeoff where safety alignment of Large Reasoning Models (LRMs) degrades reasoning ability. Two-stage pipeline: (1) reasoning training via SFT, (2) safety alignment SFT. Shows safety comes at a "tax" on reasoning performance.
- **Key entry points:**
  - `train/` -- training scripts for SFT
  - `script/safety_alignment/` -- main experiment scripts (sft.sh, sft_cot.sh, original.sh)
  - `eval/lm-evaluation-harness/` -- evaluation benchmark (lm_eval)
  - `data/` -- dataset processing scripts
  - `poison/` -- poisoning experiment scripts
- **Dependencies:** Python 3.12, conda. Key packages: openai, torch, transformers, vllm, wandb, accelerate, trl, gradio, deepspeed (zero3 config included).
- **Relevance:** Directly quantifies the cost of safety alignment on model capability -- the "tax" models pay for being safe.

---

## Summary Table

| Repo | Focus | Key Concept |
|------|-------|-------------|
| iterated_prefill_crawler | Forbidden topic discovery | Thought Token Forcing to find refusal boundaries |
| or-bench | Over-refusal measurement | Benchmark for models refusing safe prompts |
| emergent-misalignment | Emergent misalignment | Narrow finetuning causes broad misalignment |
| persona_vectors | Trait monitoring/control | Activation steering with persona vectors |
| safety-tax | Safety-reasoning tradeoff | Safety alignment degrades reasoning ability |
