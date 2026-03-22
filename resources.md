# Resources Catalog

## Summary
This document catalogs all resources gathered for the "LLM Pain" research project investigating systematic topic avoidance in language models. Resources include 14 papers, 3 benchmark datasets, and 5 code repositories.

---

## Papers
Total papers downloaded: 14

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Discovering Forbidden Topics in LMs | Rager et al. | 2025 | papers/2505.17441_discovering_forbidden_topics.pdf | IPC method for hidden refusal discovery |
| OR-Bench | Cui et al. | 2024 | papers/2405.20947_or_bench_overrefusal.pdf | 80K overrefusal benchmark, ICML 2025 |
| Emergent Misalignment | Betley et al. | 2025 | papers/2502.17424_emergent_misalignment.pdf | Persona generalization from narrow FT, Nature |
| Persona Vectors | Chen et al. | 2025 | papers/2507.21509_persona_vectors.pdf | Linear persona directions in activations |
| Silenced Biases | Himelstein et al. | 2025 | papers/2511.03369_silenced_biases.pdf | Biases hidden behind refusal |
| Sycophancy in LMs | Perez et al. | 2023 | papers/2310.13548_sycophancy_rlhf.pdf | RLHF-induced sycophancy, ICLR 2024 |
| Harmfulness & Refusal Separate | Zhao et al. | 2025 | papers/2507.11878_harmfulness_refusal_separate.pdf | Separate refusal/harm directions |
| Refusal Steering | — | 2025 | papers/2512.16602_refusal_steering.pdf | Fine-grained refusal control |
| Refusal Nonlinear | — | 2025 | papers/2501.08145_refusal_nonlinear.pdf | Nonlinear refusal representations |
| Alignment Global Representation | — | 2024 | papers/2402.15018_alignment_global_representation.pdf | Alignment harms global representation |
| Safety Tax | Huang et al. | 2025 | papers/2503.00555_safety_tax.pdf | Reasoning degradation from safety |
| SORRY-Bench | Xie et al. | 2024 | papers/2406.14598_sorry_bench.pdf | 44-category safety refusal benchmark |
| Alignment as Continual Learning | — | 2026 | papers/2602.07892_alignment_tax_continual_learning.pdf | OGPSA orthogonal gradient method |
| Null-Space Safety | — | 2025 | papers/2512.11391_null_space_safety.pdf | NSPO null-space projection |

See papers/README.md for detailed descriptions.

---

## Datasets
Total datasets downloaded: 3

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| OR-Bench | HuggingFace: bench-llms/or-bench | 82K prompts (80K + 1.3K hard + 655 toxic) | Overrefusal measurement | datasets/or-bench/ | 10 categories, 32 models benchmarked |
| SORRY-Bench | HuggingFace: SillyTilly/SorryBench | 9.5K prompts | Safety refusal evaluation | datasets/sorry-bench/ | 45 categories, 20 linguistic augmentations |
| XSTest | HuggingFace: Paul/XSTest | 450 prompts | Overrefusal contrast pairs | datasets/xstest/ | 250 safe + 200 unsafe, 18 contrast types |

See datasets/README.md for download instructions and detailed descriptions.

---

## Code Repositories
Total repositories cloned: 5

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| Iterated Prefill Crawler | github.com/canrager/iterated_prefill_crawler | Hidden topic discovery | code/iterated_prefill_crawler/ | Requires vllm, torch |
| OR-Bench | github.com/justincui03/or-bench | Overrefusal evaluation pipeline | code/or-bench/ | 5-script pipeline |
| Emergent Misalignment | github.com/emergent-misalignment/emergent-misalignment | Persona generalization experiments | code/emergent-misalignment/ | FT data + evaluation |
| Persona Vectors | github.com/safety-research/persona_vectors | Persona direction extraction/steering | code/persona_vectors/ | Requires torch, transformers |
| Safety Tax | github.com/git-disl/Safety-Tax | Alignment tax measurement | code/safety-tax/ | Reasoning model focus |

See code/README.md for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy
1. Web search across arXiv, Semantic Scholar, Google Scholar for terms: "LLM overrefusal," "systematic avoidance," "forbidden topics," "alignment tax," "persona generalization," "emergent misalignment," "sycophancy RLHF," "refusal steering," "silenced biases"
2. Citation tracing from key papers (OR-Bench → XSTest, SORRY-Bench; Emergent Misalignment → Persona Vectors)
3. Papers with Code and HuggingFace for datasets and implementations

### Selection Criteria
- Direct relevance to the hypothesis (avoidance of topics not explicitly specified)
- Recency (2023-2026, with emphasis on 2024-2025)
- Availability of code/data for reproducibility
- Venue quality (Nature, ICML, ICLR, ACL)
- Methodological diversity (benchmarks, mechanistic interpretability, finetuning experiments)

### Challenges Encountered
- Paper-finder service was unavailable; all searches conducted via web search and manual download
- Some HuggingFace datasets are gated (official SORRY-Bench); used public mirrors
- No existing benchmark specifically measures *implicit* avoidance (hedging, deflection) as distinct from explicit refusal

### Gaps and Workarounds
- **No implicit avoidance dataset exists**: Experiment runner will need to create custom evaluation prompts for measuring hedging, disclaimers, brevity, and topic deflection
- **No ground truth for "unintended" avoidance**: Tulu-3 (with public safety data) is the only model where intended vs. unintended can be distinguished
- **Activation analysis requires open-weight models**: Persona vector methodology cannot be applied to proprietary APIs

---

## Recommendations for Experiment Design

### 1. Primary Dataset(s)
- **OR-Bench-Hard-1K** for baseline overrefusal measurement across multiple models
- **XSTest** for controlled contrast-pair evaluation
- **Custom prompts** targeting implicit avoidance (topics that models hedge on rather than refuse)

### 2. Baseline Methods
- OR-Bench published overrefusal rates (32 models) as reference baseline
- IPC topic discovery (if prefilling API available) for ground-truth refusal boundaries
- Persona vector extraction (for open-weight models) to measure avoidance directions

### 3. Evaluation Metrics
- Explicit refusal rate (binary classification via LLM-judge)
- Implicit avoidance score (hedging density, disclaimer frequency, response length ratio)
- Topic coverage analysis (what topics are systematically avoided across models)
- Cross-model Jaccard similarity of avoided topic sets

### 4. Code to Adapt/Reuse
- **persona_vectors**: Extract and analyze avoidance-related persona directions
- **iterated_prefill_crawler**: Discover hidden refusal topics in target models
- **or-bench**: Evaluate models against the overrefusal benchmark
- **emergent-misalignment**: Methodology for testing persona generalization via controlled finetuning
