from __future__ import annotations

import argparse
from pathlib import Path

from routerbench_mini.cli import build_providers, evaluate, precompute_responses
from routerbench_mini.config import load_costs
from routerbench_mini.metrics import summarize_rows, summarize_rows_by_category, write_csv
from routerbench_mini.routers import LearnedCostAwareRouter
from routerbench_mini.selection import LearnedQualityGapEstimator, cross_validated_advantages
from routerbench_mini.tasks import load_jsonl

from run_v3_study import tune_learned_threshold


VARIANTS = {
    "learned_structured_only": {"include_text": False, "include_structured": True},
    "learned_text_only": {"include_text": True, "include_structured": False},
    "learned_combined": {"include_text": True, "include_structured": True},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V3 learned-router feature ablations.")
    parser.add_argument("--development", default="data/manifest.jsonl")
    parser.add_argument("--test", default="data/v3_test.jsonl")
    parser.add_argument("--models", default="configs/models.qwen_api.yaml")
    parser.add_argument("--costs", default="configs/costs.yaml")
    parser.add_argument("--out", default="results/qwen3.5-v3-ablation")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    development = load_jsonl(args.development)
    test = load_jsonl(args.test)
    providers = build_providers(args.models)
    costs = load_costs(args.costs)
    precompute_responses([*development, *test], providers, workers=args.workers)
    cheap = [providers["cheap"].generate(task) for task in development]
    strong = [providers["strong"].generate(task) for task in development]
    output_dir = Path(args.out)
    routers = []
    selected_rows = []

    for name, options in VARIANTS.items():
        oof_scores = cross_validated_advantages(development, cheap, strong, folds=5, **options)
        threshold, curve = tune_learned_threshold(development, cheap, strong, oof_scores, costs)
        for row in curve:
            row["variant"] = name
        write_csv(output_dir / f"{name}_development_thresholds.csv", curve)
        estimator = LearnedQualityGapEstimator(**options).fit(development, cheap, strong)
        routers.append(LearnedCostAwareRouter(estimator, threshold, name=name))
        selected_rows.append({"variant": name, "selected_advantage_threshold": threshold})

    rows = evaluate(test, providers, routers, costs)
    write_csv(output_dir / "selected_thresholds.csv", selected_rows)
    write_csv(output_dir / "test_predictions.csv", rows)
    write_csv(output_dir / "test_summary.csv", summarize_rows(rows))
    write_csv(output_dir / "test_summary_by_category.csv", summarize_rows_by_category(rows))


if __name__ == "__main__":
    main()
