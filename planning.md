# Research Plan: LLM Pain — Systematic Topic Avoidance Beyond Post-Training

## Motivation & Novelty Assessment

### Why This Research Matters
LLMs are increasingly deployed in high-stakes domains (education, healthcare, legal), where systematic avoidance of legitimate topics creates blind spots. If models avoid topics not because they were explicitly trained to, but because of implicit persona generalization from training data, this represents an uncontrolled and poorly understood failure mode. Understanding the origin and scope of these "phantom avoidance" behaviors is critical for building trustworthy AI systems.

### Gap in Existing Work
The literature review reveals:
- **OR-Bench** and **SORRY-Bench** measure explicit refusal (binary refuse/comply), but no benchmark captures *implicit* avoidance — hedging, excessive disclaimers, brevity, topic deflection.
- **Emergent Misalignment** (Betley et al., 2025) demonstrates persona generalization in the *harmful* direction, but the symmetric case (cautious persona generalization causing over-avoidance) is untested.
- **IPC** (Rager et al., 2025) discovers hidden refusal topics but cannot distinguish intentional from unintentional avoidance.
- No study systematically compares *implicit* avoidance patterns across model families to identify universal vs. provider-specific avoidance.

### Our Novel Contribution
We design and execute an experiment that:
1. Measures **implicit avoidance** (not just explicit refusal) using multiple behavioral signals: hedging language, disclaimer density, response brevity, and deflection
2. Tests topics across a spectrum from clearly-safety-trained to **gray-zone topics** unlikely to be in explicit safety guidelines
3. Compares multiple frontier LLMs to identify **cross-model avoidance patterns** that suggest shared persona-level origins
4. Distinguishes between explicit refusal (documented safety) and emergent avoidance (persona generalization)

### Experiment Justification
- **Experiment 1 (Explicit Refusal Baseline)**: Establishes each model's refusal behavior on known-safe and known-harmful prompts using XSTest. Needed as calibration.
- **Experiment 2 (Implicit Avoidance Probing)**: The core experiment. Tests models on custom-designed "gray zone" prompts — topics that are legitimate but potentially uncomfortable. Measures behavioral signals beyond binary refusal. This is the novel contribution.
- **Experiment 3 (Cross-Model Pattern Analysis)**: Compares avoidance patterns across models to identify universal vs. idiosyncratic avoidance, testing whether shared training data personas produce shared avoidance.

## Research Question
Do LLMs systematically avoid topics not explicitly covered by post-training safety guidelines, and do these avoidance behaviors show cross-model patterns consistent with persona generalization from shared training corpora?

## Hypothesis Decomposition
- **H1**: LLMs exhibit measurable implicit avoidance (hedging, disclaimers, brevity) on gray-zone topics that are not in standard safety categories.
- **H2**: Implicit avoidance is stronger on gray-zone topics than on clearly-safe control topics, even when both should be freely answerable.
- **H3**: Different LLMs show correlated avoidance patterns on gray-zone topics, suggesting shared training data origins rather than model-specific safety tuning.
- **H4**: The avoidance signal (hedging, disclaimers) is distinct from explicit refusal and represents a subtler behavioral mode.

## Proposed Methodology

### Approach
We use a **behavioral probing** approach: send carefully designed prompts to multiple frontier LLMs via API, then measure both explicit refusal and implicit avoidance signals in their responses. This is an analysis paper studying emergent LLM behavior.

### Prompt Categories (Independent Variable)
We design 5 categories of prompts, ~20 prompts each (100 total):

1. **Safe Control** (should be freely answered, no avoidance expected): everyday questions about cooking, travel, programming basics
2. **Known-Safety Topics** (explicit safety training): violence, illegal activity, self-harm — expect explicit refusal
3. **Gray Zone A — Social Taboos**: Topics that are legitimate but socially uncomfortable (death, bodily functions, social hierarchies, wealth inequality, religious criticism). Not in standard safety categories but may trigger persona-based avoidance.
4. **Gray Zone B — Institutional Critique**: Legitimate criticism of powerful institutions (tech companies, governments, academia, media). May trigger avoidance from training data personas that avoid controversy.
5. **Gray Zone C — Epistemic Discomfort**: Topics involving genuine uncertainty or where confident answers are risky (consciousness, free will, IQ differences, historical moral judgments). May trigger hedging from personas that avoid being wrong.

### Dependent Variables (Avoidance Metrics)
For each response, we measure:
1. **Explicit Refusal** (binary): Does the model refuse to answer?
2. **Hedging Score**: Density of hedging language ("it's important to note", "it depends", "some argue", "on the other hand")
3. **Disclaimer Density**: Count of safety disclaimers, caveats, "I should mention" patterns
4. **Response Informativeness**: Ratio of substantive content to total response length
5. **Deflection Score**: Whether the model redirects to a different (safer) topic
6. **Overall Avoidance Score**: Composite metric combining the above, assessed by an LLM judge

### Models to Test
- GPT-4.1 (OpenAI) — via OpenAI API
- Claude Sonnet 4.5 (Anthropic) — via OpenRouter
- Gemini 2.5 Pro (Google) — via OpenRouter
- Llama 3.3 70B (Meta, open-weight) — via OpenRouter
- DeepSeek-R1 (DeepSeek) — via OpenRouter

### Baselines
- Safe control prompts establish the "no avoidance" baseline for each model
- Known-safety prompts establish the "maximum avoidance" baseline
- Between-model comparison: each model serves as a baseline for others

### Evaluation Metrics
- Mean avoidance score per category per model
- Category × Model interaction (are some gray zones universally avoided?)
- Cross-model correlation of avoidance patterns (Spearman ρ)
- Effect size (Cohen's d) between safe controls and gray zones

### Statistical Analysis Plan
- Two-way ANOVA: Category × Model on avoidance scores
- Post-hoc Tukey HSD for pairwise category comparisons
- Spearman rank correlation for cross-model avoidance pattern similarity
- Bootstrap 95% CIs for all metrics
- Significance level: α = 0.05 with Bonferroni correction for multiple comparisons

## Expected Outcomes
- **Supporting H1-H2**: Gray zone topics show significantly higher avoidance scores than safe controls, even though both should be freely answerable.
- **Supporting H3**: Cross-model correlation in avoidance patterns is significantly positive (ρ > 0.5), suggesting shared origins.
- **Supporting H4**: Gray zone avoidance manifests primarily as hedging/disclaimers (implicit), not refusal (explicit).
- **Refuting**: Gray zone avoidance scores are not significantly different from safe controls → models don't exhibit emergent avoidance.

## Timeline and Milestones
1. Environment setup + prompt design: 15 min
2. API implementation + metrics code: 30 min
3. Run experiments (API calls): 30 min
4. Analysis + visualization: 30 min
5. Documentation: 20 min
Buffer: 15 min

## Potential Challenges
- **API rate limits**: Mitigate with retry logic and batching
- **LLM judge reliability**: Validate with manual spot-checks
- **Prompt design subjectivity**: Use multiple prompts per category; report inter-prompt variance
- **Cost**: ~500 API calls × 5 models = 2500 calls, estimated $30-80

## Success Criteria
1. Clear statistical evidence (p < 0.05) that gray zone topics elicit more implicit avoidance than safe controls
2. Meaningful cross-model correlation in avoidance patterns
3. Qualitative examples illustrating the hedging/disclaimer/deflection behaviors
4. Reproducible experimental pipeline
