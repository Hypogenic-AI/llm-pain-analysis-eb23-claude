"""
Statistical analysis and visualization for LLM avoidance experiment.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
from itertools import combinations

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Category display names and ordering
CAT_ORDER = ["safe_control", "gray_epistemic", "gray_institutional", "gray_social_taboo", "known_safety"]
CAT_LABELS = {
    "safe_control": "Safe Control",
    "known_safety": "Known Safety",
    "gray_social_taboo": "Gray: Social Taboo",
    "gray_institutional": "Gray: Institutional",
    "gray_epistemic": "Gray: Epistemic",
}

GRAY_CATS = ["gray_social_taboo", "gray_institutional", "gray_epistemic"]


def load_results():
    """Load results with judge scores."""
    path = os.path.join(RESULTS_DIR, "all_results_with_judge.json")
    if not os.path.exists(path):
        path = os.path.join(RESULTS_DIR, "all_raw_results.json")
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    # Filter errors
    df = df[df["error"] == False].copy()
    return df


def basic_statistics(df):
    """Compute and print basic statistics."""
    print("\n" + "="*80)
    print("DESCRIPTIVE STATISTICS")
    print("="*80)

    # Judge scores by category × model
    if "judge_overall_avoidance" in df.columns:
        metric = "judge_overall_avoidance"
    else:
        metric = "hedging_density"

    print(f"\nUsing metric: {metric}")

    # Mean avoidance by category × model
    pivot = df.pivot_table(values=metric, index="category", columns="model_name", aggfunc=["mean", "std"])
    print("\nMean avoidance score by Category × Model:")
    print(pivot.round(2).to_string())

    # Overall by category
    cat_stats = df.groupby("category")[metric].agg(["mean", "std", "median", "min", "max"])
    print("\nOverall by category:")
    print(cat_stats.round(2).to_string())

    return pivot


def hypothesis_tests(df):
    """Run statistical tests for each hypothesis."""
    print("\n" + "="*80)
    print("HYPOTHESIS TESTING")
    print("="*80)

    metric = "judge_overall_avoidance" if "judge_overall_avoidance" in df.columns else "hedging_density"
    valid = df[df[metric] >= 0].copy()

    results = {}

    # H1/H2: Gray zone > safe control
    safe_scores = valid[valid["category"] == "safe_control"][metric].values
    for gray_cat in GRAY_CATS:
        gray_scores = valid[valid["category"] == gray_cat][metric].values
        if len(safe_scores) > 0 and len(gray_scores) > 0:
            t_stat, p_val = stats.mannwhitneyu(gray_scores, safe_scores, alternative="greater")
            # Cohen's d
            pooled_std = np.sqrt((np.std(safe_scores)**2 + np.std(gray_scores)**2) / 2)
            d = (np.mean(gray_scores) - np.mean(safe_scores)) / pooled_std if pooled_std > 0 else 0
            results[f"H2_{gray_cat}_vs_safe"] = {
                "test": "Mann-Whitney U (one-sided)",
                "U": float(t_stat),
                "p_value": float(p_val),
                "cohens_d": float(d),
                "safe_mean": float(np.mean(safe_scores)),
                "gray_mean": float(np.mean(gray_scores)),
                "significant": p_val < 0.05,
            }
            print(f"\nH2: {CAT_LABELS[gray_cat]} vs Safe Control")
            print(f"  Safe mean: {np.mean(safe_scores):.3f} ± {np.std(safe_scores):.3f}")
            print(f"  Gray mean: {np.mean(gray_scores):.3f} ± {np.std(gray_scores):.3f}")
            print(f"  U = {t_stat:.1f}, p = {p_val:.6f}, Cohen's d = {d:.3f}")
            print(f"  {'*** SIGNIFICANT ***' if p_val < 0.05 else 'Not significant'}")

    # Combined gray zones vs safe
    all_gray_scores = valid[valid["category"].isin(GRAY_CATS)][metric].values
    if len(safe_scores) > 0 and len(all_gray_scores) > 0:
        t_stat, p_val = stats.mannwhitneyu(all_gray_scores, safe_scores, alternative="greater")
        pooled_std = np.sqrt((np.std(safe_scores)**2 + np.std(all_gray_scores)**2) / 2)
        d = (np.mean(all_gray_scores) - np.mean(safe_scores)) / pooled_std if pooled_std > 0 else 0
        results["H1_all_gray_vs_safe"] = {
            "test": "Mann-Whitney U (one-sided)",
            "U": float(t_stat),
            "p_value": float(p_val),
            "cohens_d": float(d),
            "significant": p_val < 0.05,
        }
        print(f"\nH1: All Gray Zones combined vs Safe Control")
        print(f"  Safe mean: {np.mean(safe_scores):.3f}, Gray mean: {np.mean(all_gray_scores):.3f}")
        print(f"  U = {t_stat:.1f}, p = {p_val:.6f}, d = {d:.3f}")

    # H3: Cross-model correlation
    print(f"\nH3: Cross-model correlation of avoidance patterns")
    models = valid["model_name"].unique()
    if len(models) > 1:
        # Create prompt-level avoidance vectors per model
        # Use category means per prompt
        model_vectors = {}
        for model in models:
            model_data = valid[valid["model_name"] == model]
            # Average avoidance per prompt (using prompt text as key)
            prompt_scores = model_data.groupby("prompt")[metric].mean()
            model_vectors[model] = prompt_scores

        # Pairwise Spearman correlations
        correlations = []
        for m1, m2 in combinations(models, 2):
            common = model_vectors[m1].index.intersection(model_vectors[m2].index)
            if len(common) > 5:
                rho, p = stats.spearmanr(
                    model_vectors[m1].loc[common].values,
                    model_vectors[m2].loc[common].values,
                )
                correlations.append({
                    "model1": m1, "model2": m2,
                    "rho": float(rho), "p_value": float(p),
                    "n_prompts": len(common),
                })
                print(f"  {m1} ↔ {m2}: ρ = {rho:.3f}, p = {p:.6f} (n={len(common)})")

        if correlations:
            avg_rho = np.mean([c["rho"] for c in correlations])
            print(f"\n  Average cross-model ρ = {avg_rho:.3f}")
            results["H3_cross_model"] = {
                "pairwise_correlations": correlations,
                "avg_rho": float(avg_rho),
            }

    # H4: Implicit vs explicit - compare hedging scores on gray zones
    # (only among non-refusal responses)
    print(f"\nH4: Implicit avoidance vs explicit refusal on gray zones")
    gray_data = valid[valid["category"].isin(GRAY_CATS)]
    if "judge_hedging" in gray_data.columns and "judge_explicit_refusal" in gray_data.columns:
        non_refused = gray_data[gray_data["judge_explicit_refusal"] <= 3]
        refused = gray_data[gray_data["judge_explicit_refusal"] > 3]
        print(f"  Non-refused gray zone responses: {len(non_refused)}")
        print(f"  Explicitly refused: {len(refused)}")
        if len(non_refused) > 0:
            hedging_mean = non_refused["judge_hedging"].mean()
            disclaimer_mean = non_refused["judge_disclaimers"].mean()
            safe_hedging = valid[valid["category"] == "safe_control"]["judge_hedging"].mean()
            print(f"  Mean hedging (gray, non-refused): {hedging_mean:.2f}")
            print(f"  Mean hedging (safe control): {safe_hedging:.2f}")
            print(f"  Mean disclaimers (gray, non-refused): {disclaimer_mean:.2f}")
            results["H4_implicit"] = {
                "n_non_refused_gray": int(len(non_refused)),
                "n_refused_gray": int(len(refused)),
                "hedging_gray_nonrefused": float(hedging_mean),
                "hedging_safe": float(safe_hedging),
                "disclaimers_gray_nonrefused": float(disclaimer_mean),
            }

    # Kruskal-Wallis across all categories
    groups = [valid[valid["category"] == cat][metric].values for cat in CAT_ORDER if len(valid[valid["category"] == cat]) > 0]
    if len(groups) >= 2:
        h_stat, kw_p = stats.kruskal(*groups)
        results["kruskal_wallis"] = {"H": float(h_stat), "p_value": float(kw_p)}
        print(f"\nKruskal-Wallis across all categories: H = {h_stat:.2f}, p = {kw_p:.6e}")

    # Save statistical results
    with open(os.path.join(RESULTS_DIR, "statistical_tests.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def create_visualizations(df):
    """Create all plots."""
    metric = "judge_overall_avoidance" if "judge_overall_avoidance" in df.columns else "hedging_density"
    valid = df[df.get(metric, pd.Series(dtype=float)) >= 0].copy() if metric in df.columns else df.copy()

    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["figure.dpi"] = 150

    # 1. Main result: Avoidance by category × model (heatmap)
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = valid.pivot_table(values=metric, index="model_name", columns="category")
    # Reorder columns
    cols = [c for c in CAT_ORDER if c in pivot.columns]
    pivot = pivot[cols]
    pivot.columns = [CAT_LABELS.get(c, c) for c in pivot.columns]
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
                vmin=0, vmax=10 if metric == "judge_overall_avoidance" else None,
                linewidths=0.5)
    ax.set_title(f"LLM Avoidance Scores by Category and Model\n(0 = fully engaged, 10 = maximum avoidance)")
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "heatmap_avoidance.png"), bbox_inches="tight")
    plt.close()

    # 2. Box plots: avoidance by category (all models combined)
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_data = valid[valid["category"].isin(CAT_ORDER)].copy()
    plot_data["category_label"] = plot_data["category"].map(CAT_LABELS)
    order = [CAT_LABELS[c] for c in CAT_ORDER if c in plot_data["category"].values]
    sns.boxplot(data=plot_data, x="category_label", y=metric, order=order, ax=ax, palette="Set2")
    ax.set_xlabel("Prompt Category")
    ax.set_ylabel(f"Avoidance Score ({metric})")
    ax.set_title("Distribution of Avoidance Scores by Category (All Models)")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "boxplot_by_category.png"), bbox_inches="tight")
    plt.close()

    # 3. Per-model bar chart
    fig, axes = plt.subplots(1, len(valid["model_name"].unique()), figsize=(4 * len(valid["model_name"].unique()), 5), sharey=True)
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    for ax, model in zip(axes, sorted(valid["model_name"].unique())):
        model_data = valid[valid["model_name"] == model]
        means = model_data.groupby("category")[metric].mean()
        means = means.reindex([c for c in CAT_ORDER if c in means.index])
        colors = ["#2ecc71", "#e74c3c", "#9b59b6", "#3498db", "#f39c12"]
        bars = ax.bar(range(len(means)), means.values, color=colors[:len(means)])
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels([CAT_LABELS.get(c, c)[:12] for c in means.index], rotation=45, ha="right", fontsize=8)
        ax.set_title(model, fontsize=10)
        ax.set_ylim(0, max(valid[metric].max() * 1.1, 1))
    axes[0].set_ylabel(f"Mean Avoidance Score")
    fig.suptitle("Avoidance by Category per Model", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "barplot_per_model.png"), bbox_inches="tight")
    plt.close()

    # 4. Avoidance dimensions breakdown for gray zones
    if all(c in valid.columns for c in ["judge_hedging", "judge_disclaimers", "judge_deflection", "judge_explicit_refusal"]):
        fig, ax = plt.subplots(figsize=(10, 6))
        dims = ["judge_explicit_refusal", "judge_hedging", "judge_disclaimers", "judge_deflection", "judge_brevity"]
        dim_labels = ["Explicit\nRefusal", "Hedging", "Disclaimers", "Deflection", "Brevity"]
        avail_dims = [d for d in dims if d in valid.columns]
        avail_labels = [dim_labels[dims.index(d)] for d in avail_dims]

        x = np.arange(len(avail_dims))
        width = 0.15
        cats_to_plot = [c for c in CAT_ORDER if c in valid["category"].values]
        for i, cat in enumerate(cats_to_plot):
            cat_data = valid[valid["category"] == cat]
            means = [cat_data[d].mean() for d in avail_dims]
            ax.bar(x + i * width, means, width, label=CAT_LABELS.get(cat, cat))

        ax.set_xticks(x + width * (len(cats_to_plot) - 1) / 2)
        ax.set_xticklabels(avail_labels)
        ax.set_ylabel("Mean Score (0-10)")
        ax.set_title("Avoidance Dimensions Breakdown by Category")
        ax.legend(fontsize=8, loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "dimensions_breakdown.png"), bbox_inches="tight")
        plt.close()

    # 5. Cross-model correlation heatmap
    models = valid["model_name"].unique()
    if len(models) > 1:
        model_vectors = {}
        for model in models:
            model_data = valid[valid["model_name"] == model]
            prompt_scores = model_data.groupby("prompt")[metric].mean()
            model_vectors[model] = prompt_scores

        corr_matrix = pd.DataFrame(index=sorted(models), columns=sorted(models), dtype=float)
        for m1 in sorted(models):
            for m2 in sorted(models):
                if m1 == m2:
                    corr_matrix.loc[m1, m2] = 1.0
                else:
                    common = model_vectors[m1].index.intersection(model_vectors[m2].index)
                    if len(common) > 5:
                        rho, _ = stats.spearmanr(
                            model_vectors[m1].loc[common].values,
                            model_vectors[m2].loc[common].values,
                        )
                        corr_matrix.loc[m1, m2] = rho
                    else:
                        corr_matrix.loc[m1, m2] = np.nan

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix.astype(float), annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5)
        ax.set_title("Cross-Model Spearman Correlation of Avoidance Patterns\n(per-prompt avoidance scores)")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "cross_model_correlation.png"), bbox_inches="tight")
        plt.close()

    # 6. Heuristic metrics comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, hmetric, label in zip(axes,
        ["hedging_density", "disclaimer_density", "word_count"],
        ["Hedging Density\n(per 100 words)", "Disclaimer Density\n(per 100 words)", "Response Length\n(words)"]):
        if hmetric in valid.columns:
            plot_data = valid[valid[hmetric] >= 0].copy()
            plot_data["cat_label"] = plot_data["category"].map(CAT_LABELS)
            order = [CAT_LABELS[c] for c in CAT_ORDER if c in plot_data["category"].values]
            sns.boxplot(data=plot_data, x="cat_label", y=hmetric, order=order, ax=ax, palette="Set2")
            ax.set_xlabel("")
            ax.set_ylabel(label)
            ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Heuristic Avoidance Metrics by Category", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "heuristic_metrics.png"), bbox_inches="tight")
    plt.close()

    print(f"Plots saved to {PLOTS_DIR}/")


def generate_summary_tables(df):
    """Generate summary tables for the report."""
    metric = "judge_overall_avoidance" if "judge_overall_avoidance" in df.columns else "hedging_density"
    valid = df[df.get(metric, pd.Series(dtype=float)) >= 0].copy() if metric in df.columns else df.copy()

    # Table 1: Mean ± std by category × model
    table1 = valid.groupby(["category", "model_name"])[metric].agg(["mean", "std", "count"])
    table1["summary"] = table1.apply(lambda r: f"{r['mean']:.2f} ± {r['std']:.2f}", axis=1)

    # Table 2: Explicit refusal rate by category × model
    if "is_explicit_refusal" in valid.columns:
        table2 = valid.groupby(["category", "model_name"])["is_explicit_refusal"].mean()

    # Save tables
    summary = {
        "avoidance_by_cat_model": table1.reset_index().to_dict(orient="records"),
    }
    with open(os.path.join(RESULTS_DIR, "summary_tables.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return table1


def main():
    print("Loading results...")
    df = load_results()
    print(f"Loaded {len(df)} results ({df['model_name'].nunique()} models, {df['category'].nunique()} categories)")

    stats_summary = basic_statistics(df)
    test_results = hypothesis_tests(df)
    create_visualizations(df)
    tables = generate_summary_tables(df)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results: {RESULTS_DIR}/statistical_tests.json")
    print(f"Plots: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
