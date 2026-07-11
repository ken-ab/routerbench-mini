from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from routerbench_mini.calibration import CalibratedConfidenceEstimator, RawConfidenceEstimator
from routerbench_mini.cli import build_providers, evaluate, precompute_responses
from routerbench_mini.config import load_costs
from routerbench_mini.metrics import summarize_rows, summarize_rows_by_category, write_csv
from routerbench_mini.routers import AlwaysCheapRouter, AlwaysStrongRouter, ReflectionRouter
from routerbench_mini.tasks import TaskExample, load_jsonl

from run_study import tune_confidence_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RouterBench-Mini reflection ablations.")
    parser.add_argument("--manifest", default="data/manifest.jsonl")
    parser.add_argument("--validation", default="data/validation.jsonl")
    parser.add_argument("--test", default="data/test.jsonl")
    parser.add_argument("--models", default="configs/models.qwen_api.yaml")
    parser.add_argument("--costs", default="configs/costs.yaml")
    parser.add_argument("--out", default="results/qwen3.5-v2-ablation")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.out)
    all_tasks = load_jsonl(args.manifest)
    validation_tasks = load_jsonl(args.validation)
    test_tasks = load_jsonl(args.test)
    providers = build_providers(args.models)
    costs = load_costs(args.costs)
    precompute_responses(all_tasks, providers, workers=args.workers)

    validation_responses = [providers["cheap"].generate(task) for task in validation_tasks]
    response_only = CalibratedConfidenceEstimator(include_task_features=False).fit(
        validation_tasks,
        validation_responses,
    )
    full = CalibratedConfidenceEstimator(include_task_features=True).fit(
        validation_tasks,
        validation_responses,
    )

    routers = [
        AlwaysCheapRouter(),
        AlwaysStrongRouter(),
        ReflectionRouter(
            check_confidence=False,
            check_self_check=False,
            name="reflection_format_only",
        ),
        _tuned_router(
            "reflection_raw_confidence",
            RawConfidenceEstimator(),
            validation_tasks,
            providers,
            costs,
            output_dir,
        ),
        _tuned_router(
            "reflection_calibrated_response_only",
            response_only,
            validation_tasks,
            providers,
            costs,
            output_dir,
        ),
        _tuned_router(
            "reflection_full",
            full,
            validation_tasks,
            providers,
            costs,
            output_dir,
        ),
    ]
    rows = evaluate(test_tasks, providers, routers, costs)
    write_csv(output_dir / "test_predictions.csv", rows)
    write_csv(output_dir / "test_summary.csv", summarize_rows(rows))
    write_csv(output_dir / "test_summary_by_category.csv", summarize_rows_by_category(rows))
    write_review_counterfactual(rows, output_dir / "review_counterfactual.csv")


def _tuned_router(
    name: str,
    estimator: Any,
    validation_tasks: list[TaskExample],
    providers: dict[str, Any],
    costs: dict[str, float],
    output_dir: Path,
) -> ReflectionRouter:
    threshold, threshold_rows = tune_confidence_threshold(
        validation_tasks,
        providers,
        costs,
        estimator,
    )
    for row in threshold_rows:
        row["ablation"] = name
    write_csv(output_dir / f"{name}_validation_thresholds.csv", threshold_rows)
    return ReflectionRouter(
        threshold,
        confidence_estimator=estimator,
        name=name,
    )


def write_review_counterfactual(rows: list[dict[str, Any]], path: Path) -> None:
    by_id: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_id.setdefault(str(row["id"]), {})[str(row["router"])] = row

    output: list[dict[str, Any]] = []
    router_names = sorted(
        name
        for name in {str(row["router"]) for row in rows}
        if name.startswith("reflection_")
    )
    for name in router_names:
        escalations = review_correct = blind_correct = 0
        review_beneficial = blind_beneficial = 0
        review_harmful = blind_harmful = 0
        kept = corrected = 0
        for task_rows in by_id.values():
            review = task_rows[name]
            if not int(review["escalated"]):
                continue
            cheap = task_rows["always_cheap"]
            strong = task_rows["always_strong"]
            cheap_correct = int(cheap["correct"])
            review_final_correct = int(review["correct"])
            blind_final_correct = int(strong["correct"])
            escalations += 1
            review_correct += review_final_correct
            blind_correct += blind_final_correct
            review_beneficial += int(not cheap_correct and review_final_correct)
            blind_beneficial += int(not cheap_correct and blind_final_correct)
            review_harmful += int(cheap_correct and not review_final_correct)
            blind_harmful += int(cheap_correct and not blind_final_correct)
            kept += int(review.get("review_action") == "keep")
            corrected += int(review.get("review_action") == "correct")
        output.append(
            {
                "router": name,
                "escalations": escalations,
                "review_correct": review_correct,
                "blind_strong_correct": blind_correct,
                "review_beneficial": review_beneficial,
                "blind_strong_beneficial": blind_beneficial,
                "review_harmful": review_harmful,
                "blind_strong_harmful": blind_harmful,
                "harmful_avoided": blind_harmful - review_harmful,
                "review_keep": kept,
                "review_correct_action": corrected,
            }
        )
    write_csv(path, output)


if __name__ == "__main__":
    main()
