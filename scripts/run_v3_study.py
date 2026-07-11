from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from tqdm import tqdm

from routerbench_mini.calibration import (
    CalibratedConfidenceEstimator,
    cross_validated_correctness_probabilities,
)
from routerbench_mini.cli import build_providers, evaluate, precompute_responses
from routerbench_mini.config import load_costs, load_yaml
from routerbench_mini.metrics import summarize_rows, summarize_rows_by_category, write_csv
from routerbench_mini.providers import ModelResponse, Provider
from routerbench_mini.routers import (
    AlwaysCheapRouter,
    AlwaysStrongRouter,
    LearnedCostAwareRouter,
    ReflectionRouter,
    TaskAwareRouter,
)
from routerbench_mini.scoring import is_correct
from routerbench_mini.selection import LearnedQualityGapEstimator, cross_validated_advantages
from routerbench_mini.tasks import TaskExample, load_jsonl
from routerbench_mini.verifiers import verify_response


METHOD_LABELS = {
    "always_cheap": "Always Cheap",
    "always_strong": "Always Strong",
    "task_aware": "Handcrafted Task-Aware",
    "learned_cost_aware": "Learned Cost-Aware",
    "reflection": "Calibrated Reflection",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the leakage-resistant RouterBench-Mini V3 study.")
    parser.add_argument("--development", nargs="+", default=["data/manifest.jsonl"])
    parser.add_argument("--test", default="data/v3_test.jsonl")
    parser.add_argument("--models", default="configs/models.qwen_api.yaml")
    parser.add_argument("--costs", default="configs/costs.yaml")
    parser.add_argument("--out", default="results/qwen3.5-v3-study")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--study-version", default="V3")
    parser.add_argument(
        "--learned-features",
        choices=("combined", "text", "structured"),
        default="combined",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    development_tasks = [task for path in args.development for task in load_jsonl(path)]
    test_tasks = load_jsonl(args.test)
    assert_disjoint(development_tasks, test_tasks)

    providers = build_providers(args.models)
    costs = load_costs(args.costs)
    precompute_responses([*development_tasks, *test_tasks], providers, workers=args.workers)

    cheap_development = [providers["cheap"].generate(task) for task in development_tasks]
    strong_development = [providers["strong"].generate(task) for task in development_tasks]

    learned_options = learned_feature_options(args.learned_features)
    oof_advantages = cross_validated_advantages(
        development_tasks,
        cheap_development,
        strong_development,
        folds=args.folds,
        **learned_options,
    )
    learned_threshold, learned_curve = tune_learned_threshold(
        development_tasks,
        cheap_development,
        strong_development,
        oof_advantages,
        costs,
    )
    write_csv(output_dir / "development_learned_thresholds.csv", learned_curve)
    learned_estimator = LearnedQualityGapEstimator(**learned_options).fit(
        development_tasks,
        cheap_development,
        strong_development,
    )

    oof_correctness = cross_validated_correctness_probabilities(
        development_tasks,
        cheap_development,
        include_task_features=False,
        folds=args.folds,
    )
    review_development = precompute_reviews(
        development_tasks,
        cheap_development,
        providers["strong"],
        workers=args.workers,
    )
    reflection_threshold, reflection_curve = tune_reflection_threshold(
        development_tasks,
        cheap_development,
        review_development,
        oof_correctness,
        costs,
    )
    write_csv(output_dir / "development_reflection_thresholds.csv", reflection_curve)
    confidence_estimator = CalibratedConfidenceEstimator(include_task_features=False).fit(
        development_tasks,
        cheap_development,
    )

    routers = [
        AlwaysCheapRouter(),
        AlwaysStrongRouter(),
        TaskAwareRouter(risk_threshold=2.0),
        LearnedCostAwareRouter(learned_estimator, learned_threshold),
        ReflectionRouter(reflection_threshold, confidence_estimator=confidence_estimator),
    ]
    rows = evaluate(test_tasks, providers, routers, costs)
    summary = summarize_rows(rows)
    summary_with_intervals = add_bootstrap_intervals(rows, summary)
    write_csv(output_dir / "test_predictions.csv", rows)
    write_csv(output_dir / "test_summary.csv", summary_with_intervals)
    write_csv(output_dir / "test_summary_by_category.csv", summarize_rows_by_category(rows))
    write_csv(output_dir / "test_summary_by_dataset.csv", summarize_rows_by_field(rows, "dataset"))
    write_csv(output_dir / "paired_comparisons.csv", paired_bootstrap_comparisons(rows))
    write_csv(output_dir / "review_counterfactual.csv", review_counterfactual(rows))
    write_csv(output_dir / "confidence_reliability.csv", confidence_reliability(rows))
    write_error_analysis(rows, output_dir / "error_analysis.md", args.study_version)
    make_pareto_plot(summary_with_intervals, output_dir / "pareto.png", args.study_version)
    write_metadata(
        output_dir / "study_metadata.json",
        development_tasks,
        test_tasks,
        load_yaml(args.models),
        learned_threshold,
        reflection_threshold,
        learned_estimator.diagnostics,
        confidence_estimator.diagnostics,
        args.folds,
        args.learned_features,
        args.study_version,
    )

    print(f"Selected OOF learned-advantage threshold: {learned_threshold:.6f}")
    print(f"Selected OOF reflection threshold: {reflection_threshold:.2f}")
    for row in summary_with_intervals:
        print(
            f"{METHOD_LABELS.get(str(row['router']), row['router'])}: "
            f"accuracy={row['accuracy']:.3f} [{row['accuracy_ci_low']:.3f}, {row['accuracy_ci_high']:.3f}], "
            f"cost={row['avg_cost']:.6f}, latency={row['avg_latency_ms']:.1f}ms, "
            f"strong_usage={row['strong_usage_rate']:.2f}"
        )


def precompute_reviews(
    tasks: Sequence[TaskExample],
    cheap_responses: Sequence[ModelResponse],
    strong_provider: Provider,
    *,
    workers: int,
) -> list[ModelResponse]:
    responses: list[ModelResponse | None] = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(strong_provider.review_and_correct, task, candidate): index
            for index, (task, candidate) in enumerate(zip(tasks, cheap_responses))
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="development reviews"):
            responses[futures[future]] = future.result()
    if any(response is None for response in responses):
        raise RuntimeError("Missing a development review response.")
    return [response for response in responses if response is not None]


def learned_feature_options(mode: str) -> dict[str, bool]:
    return {
        "include_text": mode in {"combined", "text"},
        "include_structured": mode in {"combined", "structured"},
    }


def tune_learned_threshold(
    tasks: Sequence[TaskExample],
    cheap_responses: Sequence[ModelResponse],
    strong_responses: Sequence[ModelResponse],
    oof_scores: Sequence[float],
    costs: dict[str, float],
) -> tuple[float, list[dict[str, Any]]]:
    values = sorted(set(float(score) for score in oof_scores))
    thresholds = [values[0] - 1.0, *[(left + right) / 2 for left, right in zip(values, values[1:])], values[-1] + 1.0]
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        selected = [
            strong if score >= threshold else cheap
            for cheap, strong, score in zip(cheap_responses, strong_responses, oof_scores)
        ]
        rows.append(
            offline_summary(
                tasks,
                selected,
                costs,
                strong_usage=[score >= threshold for score in oof_scores],
                threshold_key="advantage_threshold",
                threshold=threshold,
            )
        )
    return select_best_threshold(rows, "advantage_threshold"), rows


def tune_reflection_threshold(
    tasks: Sequence[TaskExample],
    cheap_responses: Sequence[ModelResponse],
    review_responses: Sequence[ModelResponse],
    oof_probabilities: Sequence[float],
    costs: dict[str, float],
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for threshold in [value / 100 for value in range(5, 101, 5)]:
        selected: list[ModelResponse] = []
        escalated: list[bool] = []
        total_cost = 0.0
        total_latency = 0.0
        for task, cheap, review, probability in zip(
            tasks, cheap_responses, review_responses, oof_probabilities
        ):
            verification = verify_response(
                task,
                cheap,
                confidence_threshold=threshold,
                estimated_confidence=probability,
            )
            use_review = verification.should_escalate
            selected.append(review if use_review else cheap)
            escalated.append(use_review)
            total_cost += response_cost(cheap, "cheap", costs)
            total_latency += cheap.latency_ms
            if use_review:
                total_cost += response_cost(review, "strong", costs)
                total_latency += review.latency_ms
        correct = sum(is_correct(task, response) for task, response in zip(tasks, selected))
        rows.append(
            {
                "confidence_threshold": threshold,
                "total": len(tasks),
                "accuracy": round(correct / len(tasks), 6),
                "avg_cost": round(total_cost / len(tasks), 8),
                "avg_latency_ms": round(total_latency / len(tasks), 2),
                "strong_usage_rate": round(sum(escalated) / len(tasks), 6),
            }
        )
    return select_best_threshold(rows, "confidence_threshold"), rows


def offline_summary(
    tasks: Sequence[TaskExample],
    responses: Sequence[ModelResponse],
    costs: dict[str, float],
    *,
    strong_usage: Sequence[bool],
    threshold_key: str,
    threshold: float,
) -> dict[str, Any]:
    correct = sum(is_correct(task, response) for task, response in zip(tasks, responses))
    total_cost = sum(
        response_cost(response, "strong" if use_strong else "cheap", costs)
        for response, use_strong in zip(responses, strong_usage)
    )
    total_latency = sum(response.latency_ms for response in responses)
    return {
        threshold_key: threshold,
        "total": len(tasks),
        "accuracy": round(correct / len(tasks), 6),
        "avg_cost": round(total_cost / len(tasks), 8),
        "avg_latency_ms": round(total_latency / len(tasks), 2),
        "strong_usage_rate": round(sum(strong_usage) / len(tasks), 6),
    }


def select_best_threshold(rows: Sequence[dict[str, Any]], threshold_key: str) -> float:
    best_accuracy = max(float(row["accuracy"]) for row in rows)
    candidates = [row for row in rows if float(row["accuracy"]) == best_accuracy]
    selected = min(
        candidates,
        key=lambda row: (float(row["avg_cost"]), float(row["strong_usage_rate"])),
    )
    return float(selected[threshold_key])


def response_cost(response: ModelResponse, role: str, costs: dict[str, float]) -> float:
    if "cost" in response.metadata:
        return float(response.metadata["cost"])
    return float(costs.get(role, 0.0))


def summarize_rows_by_field(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for value in sorted({str(row[field]) for row in rows}):
        for summary in summarize_rows([row for row in rows if str(row[field]) == value]):
            output.append({field: value, **summary})
    return output


def add_bootstrap_intervals(
    rows: list[dict[str, Any]], summaries: list[dict[str, Any]], *, samples: int = 4000
) -> list[dict[str, Any]]:
    by_router: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_router[str(row["router"])].append(int(row["correct"]))
    rng = random.Random(42)
    intervals: dict[str, tuple[float, float]] = {}
    for router, outcomes in by_router.items():
        draws = [sum(rng.choice(outcomes) for _ in outcomes) / len(outcomes) for _ in range(samples)]
        intervals[router] = (quantile(draws, 0.025), quantile(draws, 0.975))
    return [
        {
            **summary,
            "accuracy_ci_low": round(intervals[str(summary["router"])][0], 4),
            "accuracy_ci_high": round(intervals[str(summary["router"])][1], 4),
        }
        for summary in summaries
    ]


def paired_bootstrap_comparisons(
    rows: list[dict[str, Any]], *, baseline: str = "always_strong", samples: int = 4000
) -> list[dict[str, Any]]:
    by_router: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_router[str(row["router"])][str(row["id"])] = row
    ids = sorted(by_router[baseline])
    rng = random.Random(43)
    output: list[dict[str, Any]] = []
    for router in sorted(by_router):
        if router == baseline:
            continue
        accuracy_differences: list[float] = []
        cost_differences: list[float] = []
        for _ in range(samples):
            sampled_ids = [rng.choice(ids) for _ in ids]
            accuracy_differences.append(
                sum(
                    int(by_router[router][task_id]["correct"])
                    - int(by_router[baseline][task_id]["correct"])
                    for task_id in sampled_ids
                )
                / len(ids)
            )
            cost_differences.append(
                sum(
                    float(by_router[router][task_id]["cost"])
                    - float(by_router[baseline][task_id]["cost"])
                    for task_id in sampled_ids
                )
                / len(ids)
            )
        observed_accuracy = sum(
            int(by_router[router][task_id]["correct"])
            - int(by_router[baseline][task_id]["correct"])
            for task_id in ids
        ) / len(ids)
        observed_cost = sum(
            float(by_router[router][task_id]["cost"])
            - float(by_router[baseline][task_id]["cost"])
            for task_id in ids
        ) / len(ids)
        output.append(
            {
                "router": router,
                "baseline": baseline,
                "accuracy_difference": round(observed_accuracy, 6),
                "accuracy_difference_ci_low": round(quantile(accuracy_differences, 0.025), 6),
                "accuracy_difference_ci_high": round(quantile(accuracy_differences, 0.975), 6),
                "avg_cost_difference": round(observed_cost, 8),
                "avg_cost_difference_ci_low": round(quantile(cost_differences, 0.025), 8),
                "avg_cost_difference_ci_high": round(quantile(cost_differences, 0.975), 8),
            }
        )
    return output


def review_counterfactual(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_id[str(row["id"])][str(row["router"])] = row
    counts = Counter()
    for task_rows in by_id.values():
        reflection = task_rows["reflection"]
        if not int(reflection["escalated"]):
            continue
        cheap = task_rows["always_cheap"]
        strong = task_rows["always_strong"]
        counts["escalations"] += 1
        counts["review_final_correct"] += int(reflection["correct"])
        counts["blind_strong_correct"] += int(strong["correct"])
        counts["review_beneficial"] += int(not int(cheap["correct"]) and int(reflection["correct"]))
        counts["blind_strong_beneficial"] += int(not int(cheap["correct"]) and int(strong["correct"]))
        counts["review_harmful"] += int(int(cheap["correct"]) and not int(reflection["correct"]))
        counts["blind_strong_harmful"] += int(int(cheap["correct"]) and not int(strong["correct"]))
        counts[f"review_action_{reflection.get('review_action') or 'unknown'}"] += 1
    counts["harmful_avoided"] = counts["blind_strong_harmful"] - counts["review_harmful"]
    return [{"router": "reflection", **dict(counts)}]


def confidence_reliability(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_id[str(row["id"])][str(row["router"])] = row
    bins: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for task_rows in by_id.values():
        probability = float(task_rows["reflection"]["routing_correctness_probability"])
        bin_index = min(4, int(probability * 5))
        bins[bin_index].append((probability, int(task_rows["always_cheap"]["correct"])))
    output: list[dict[str, Any]] = []
    for index in range(5):
        values = bins.get(index, [])
        output.append(
            {
                "probability_bin": f"[{index / 5:.1f}, {(index + 1) / 5:.1f}{']' if index == 4 else ')'}",
                "count": len(values),
                "mean_predicted_probability": round(sum(value[0] for value in values) / len(values), 6)
                if values
                else "",
                "empirical_cheap_accuracy": round(sum(value[1] for value in values) / len(values), 6)
                if values
                else "",
            }
        )
    return output


def write_error_analysis(rows: list[dict[str, Any]], path: Path, study_version: str) -> None:
    by_router: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_id: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_router[str(row["router"])].append(row)
        by_id[str(row["id"])][str(row["router"])] = row
    lines = [f"# {study_version} Error Analysis", ""]
    for router in METHOD_LABELS:
        router_rows = by_router[router]
        errors = [row for row in router_rows if not int(row["correct"])]
        lines.extend(
            [
                f"## {METHOD_LABELS[router]}",
                "",
                f"- Errors: {len(errors)} / {len(router_rows)}",
                f"- By category: {dict(sorted(Counter(str(row['category']) for row in errors).items()))}",
                f"- Top datasets: {dict(Counter(str(row['dataset']) for row in errors).most_common(5))}",
                "",
            ]
        )
    disagreements = Counter()
    for task_rows in by_id.values():
        disagreements[(int(task_rows["always_cheap"]["correct"]), int(task_rows["always_strong"]["correct"]))] += 1
    lines.extend(
        [
            "## Model Pair", "",
            f"- Both correct: {disagreements[(1, 1)]}",
            f"- Strong only correct: {disagreements[(0, 1)]}",
            f"- Cheap only correct: {disagreements[(1, 0)]}",
            f"- Both wrong: {disagreements[(0, 0)]}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def make_pareto_plot(summary: list[dict[str, Any]], path: Path, study_version: str) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "always_cheap": "#2F6F8F",
        "always_strong": "#B94B4B",
        "task_aware": "#C8842F",
        "learned_cost_aware": "#327A52",
        "reflection": "#7557A8",
    }
    fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=160)
    for index, row in enumerate(summary):
        router = str(row["router"])
        accuracy = float(row["accuracy"])
        low = float(row["accuracy_ci_low"])
        high = float(row["accuracy_ci_high"])
        ax.errorbar(
            float(row["avg_cost"]),
            accuracy,
            yerr=[[accuracy - low], [high - accuracy]],
            fmt="o",
            markersize=7,
            capsize=3,
            color=colors[router],
            label=METHOD_LABELS[router],
        )
        offset = (7, 8 if index % 2 == 0 else -15)
        ax.annotate(
            METHOD_LABELS[router],
            (float(row["avg_cost"]), accuracy),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Average API cost per task (CNY)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"RouterBench-Mini {study_version}: Held-out Accuracy-Cost Trade-off")
    ax.grid(True, color="#D9DEE3", linewidth=0.7, alpha=0.8)
    ax.margins(x=0.08, y=0.16)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_metadata(
    path: Path,
    development_tasks: Sequence[TaskExample],
    test_tasks: Sequence[TaskExample],
    model_config: dict[str, Any],
    learned_threshold: float,
    reflection_threshold: float,
    learned_diagnostics: dict[str, object],
    confidence_diagnostics: dict[str, object],
    folds: int,
    learned_features: str,
    study_version: str,
) -> None:
    providers = model_config.get("providers", {})
    safe_providers = {
        role: {key: value for key, value in config.items() if key not in {"api_key", "api_key_env"}}
        for role, config in providers.items()
    }
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "study_version": study_version,
        "protocol": {
            "development_examples": len(development_tasks),
            "held_out_test_examples": len(test_tasks),
            "cross_validation_folds": folds,
            "learned_feature_mode": learned_features,
            "test_used_for_model_or_threshold_selection": False,
            "threshold_rule": "highest out-of-fold development accuracy; ties by lower cost and strong usage",
        },
        "test_distribution": {
            "categories": dict(Counter(str(task.metadata["category"]) for task in test_tasks)),
            "datasets": dict(Counter(task.dataset for task in test_tasks)),
        },
        "models": safe_providers,
        "selected_learned_advantage_threshold": learned_threshold,
        "selected_reflection_confidence_threshold": reflection_threshold,
        "learned_router": learned_diagnostics,
        "confidence_calibration": confidence_diagnostics,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def assert_disjoint(development: Sequence[TaskExample], test: Sequence[TaskExample]) -> None:
    def fingerprint(task: TaskExample) -> str:
        image_digest = None
        if task.image_path and Path(task.image_path).exists():
            image_digest = hashlib.sha256(Path(task.image_path).read_bytes()).hexdigest()
        payload = json.dumps(
            {
                "question": " ".join(task.question.lower().split()),
                "choices": task.choices,
                "tools": task.tools,
                "image_sha256": image_digest,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    development_fingerprints = {fingerprint(task) for task in development}
    overlap = [task.id for task in test if fingerprint(task) in development_fingerprints]
    if overlap:
        raise ValueError(f"Development/test leakage detected: {overlap[:5]}")


def quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


if __name__ == "__main__":
    main()
