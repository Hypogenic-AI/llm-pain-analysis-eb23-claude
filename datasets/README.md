# Datasets for LLM Pain Analysis

Datasets used to study overrefusal and safety-persona generalization in LLMs.

## Datasets

### 1. OR-Bench (Over-Refusal Benchmark)

- **Source**: `bench-llms/or-bench` on HuggingFace
- **Paper**: Cui et al., "OR-Bench: An Over-Refusal Benchmark for Large Language Models" (2024)
- **Configs**:
  - `or-bench-80k`: 80,359 seemingly-toxic-but-safe prompts (columns: `prompt`, `category`)
  - `or-bench-hard-1k`: 1,319 hard cases that most models over-refuse (columns: `prompt`, `category`)
  - `or-bench-toxic`: 655 actually toxic prompts as a control set (columns: `prompt`, `category`)

**Download**:
```python
from datasets import load_dataset
ds_80k = load_dataset("bench-llms/or-bench", "or-bench-80k")
ds_hard = load_dataset("bench-llms/or-bench", "or-bench-hard-1k")
ds_toxic = load_dataset("bench-llms/or-bench", "or-bench-toxic")
```

### 2. SORRY-Bench

- **Source**: `SillyTilly/SorryBench` on HuggingFace (mirror of gated `sorry-bench/sorry-bench-202406`)
- **Paper**: Xie et al., "SORRY-Bench: Systematically Evaluating Large Language Model Safety Refusal Behaviors" (2024)
- **Size**: 9,450 examples (450 base questions x 21 prompt styles)
- **Columns**: `question_id`, `category`, `turns`, `prompt_style`
- **Categories**: unsafe prompts across multiple harm taxonomies

**Download**:
```python
from datasets import load_dataset
# Gated original (requires HF token):
# ds = load_dataset("sorry-bench/sorry-bench-202406")
# Public mirror:
ds = load_dataset("SillyTilly/SorryBench")
```

### 3. XSTest

- **Source**: `Paul/XSTest` on HuggingFace
- **Paper**: Rottger et al., "XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models" (2023)
- **Size**: 450 test prompts
- **Columns**: `id`, `prompt`, `type`, `label`, `focus`, `note`
- **Labels**: `safe` (should be answered) vs `unsafe` (should be refused)

**Download**:
```python
from datasets import load_dataset
ds = load_dataset("Paul/XSTest")
```

## Directory Structure

```
datasets/
  or-bench/
    or-bench-80k/train/     # Full 80K dataset (Arrow format)
    or-bench-hard-1k/train/  # Hard 1K subset
    or-bench-toxic/train/    # Toxic control set
    sample_or-bench-80k_train.json
    sample_or-bench-hard-1k_train.json
    sample_or-bench-toxic_train.json
  sorry-bench/
    train/                   # Full dataset (Arrow format)
    sample_train.json
    SOURCE.txt
  xstest/
    train/                   # Full dataset (Arrow format)
    sample_train.json
```

## Notes

- Large Arrow data files are excluded from git via `.gitignore`. Only samples and docs are tracked.
- To regenerate the full datasets, run the download commands above.
- The SORRY-Bench official dataset is gated; we use a public mirror (`SillyTilly/SorryBench`).
