from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from routerbench_mini.cli import build_providers, evaluate, precompute_responses
from routerbench_mini.config import load_costs, load_yaml
from routerbench_mini.metrics import summarize_rows, summarize_rows_by_category, write_csv
from routerbench_mini.routers import (
    AlwaysCheapRouter,
    AlwaysStrongRouter,
    ReflectionRouter,
    TaskAwareRouter,
)
from routerbench_mini.tasks import TaskExample, load_jsonl


METHOD_LABELS = {
    "always_cheap": "Always Cheap",
    "always_strong": "Always Strong",
    "task_aware": "Task-Aware Router",
    "reflection": "Reflection Router",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete RouterBench-Mini study.")
    parser.add_argument("--manifest", default="data/manifest.jsonl")
    parser.add_argument("--validation", default="data/validation.jsonl")
    parser.add_argument("--test", default="data/test.jsonl")
    parser.add_argument("--models", default="configs/models.qwen_api.yaml")
    parser.add_argument("--costs", default="configs/costs.yaml")
    parser.add_argument("--out", default="results/qwen3.5-study")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_tasks = load_jsonl(args.manifest)
    validation_tasks = load_jsonl(args.validation)
    test_tasks = load_jsonl(args.test)
    costs = load_costs(args.costs)
    providers = build_providers(args.models)

    precompute_responses(all_tasks, providers, workers=args.workers)
    selected_threshold, threshold_rows = tune_threshold(validation_tasks, providers, costs)
    write_csv(output_dir / "validation_thresholds.csv", threshold_rows)

    validation_routers = [
        AlwaysCheapRouter(),
        AlwaysStrongRouter(),
        TaskAwareRouter(),
        ReflectionRouter(selected_threshold),
    ]
    validation_rows = evaluate(validation_tasks, providers, validation_routers, costs)
    write_csv(output_dir / "validation_predictions.csv", validation_rows)
    write_csv(output_dir / "validation_summary.csv", summarize_rows(validation_rows))

    test_routers = [
        AlwaysCheapRouter(),
        AlwaysStrongRouter(),
        TaskAwareRouter(),
        ReflectionRouter(selected_threshold),
    ]
    test_rows = evaluate(test_tasks, providers, test_routers, costs)
    test_summary = summarize_rows(test_rows)
    write_csv(output_dir / "test_predictions.csv", test_rows)
    write_csv(output_dir / "test_summary.csv", test_summary)
    write_csv(output_dir / "test_summary_by_category.csv", summarize_rows_by_category(test_rows))

    make_pareto_plot(test_summary, output_dir / "pareto.png")
    write_error_analysis(test_rows, output_dir / "error_analysis.md")
    write_metadata(
        output_dir / "study_metadata.json",
        all_tasks,
        validation_tasks,
        test_tasks,
        load_yaml(args.models),
        selected_threshold,
    )
    print(f"Selected confidence threshold: {selected_threshold:.2f}")
    for row in test_summary:
        print(
            f"{METHOD_LABELS.get(str(row['router']), row['router'])}: "
            f"accuracy={row['accuracy']:.3f}, cost={row['avg_cost']:.6f}, "
            f"latency={row['avg_latency_ms']:.1f}ms, strong_usage={row['strong_usage_rate']:.2f}"
        )


def tune_threshold(
    tasks: list[TaskExample],
    providers: dict[str, Any],
    costs: dict[str, float],
) -> tuple[float, list[dict[str, Any]]]:
    strong_rows = evaluate(tasks, providers, [AlwaysStrongRouter()], costs)
    strong_accuracy = float(summarize_rows(strong_rows)[0]["accuracy"])
    threshold_rows: list[dict[str, Any]] = []
    for threshold in [value / 100 for value in range(30, 91, 5)]:
        rows = evaluate(tasks, providers, [ReflectionRouter(threshold)], costs)
        summary = summarize_rows(rows)[0]
        threshold_rows.append({"threshold": threshold, **summary})

    target = strong_accuracy - 0.02
    eligible = [row for row in threshold_rows if float(row["accuracy"]) >= target]
    if eligible:
        selected = min(
            eligible,
            key=lambda row: (float(row["avg_cost"]), float(row["strong_usage_rate"]), -float(row["accuracy"])),
        )
    else:
        selected = min(
            threshold_rows,
            key=lambda row: (-float(row["accuracy"]), float(row["avg_cost"])),
        )
    return float(selected["threshold"]), threshold_rows


def make_pareto_plot(summary: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "always_cheap": "#3B82A0",
        "always_strong": "#C44E52",
        "task_aware": "#D08C36",
        "reflection": "#4C956C",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)
    for row in summary:
        router = str(row["router"])
        ax.scatter(
            float(row["avg_cost"]),
            float(row["accuracy"]),
            s=90,
            color=colors.get(router, "#555555"),
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.annotate(
            METHOD_LABELS.get(router, router),
            (float(row["avg_cost"]), float(row["accuracy"])),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=8.5,
        )
    ax.set_xlabel("Average API cost per task (CNY)")
    ax.set_ylabel("Accuracy")
    ax.set_title("RouterBench-Mini Accuracy-Cost Trade-off")
    ax.grid(True, color="#D9DEE3", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_error_analysis(rows: list[dict[str, Any]], path: Path) -> None:
    by_router: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_id: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_router[str(row["router"])].append(row)
        by_id[str(row["id"])][str(row["router"])] = row

    lines = ["# Error Analysis", ""]
    for router in ("always_cheap", "always_strong", "task_aware", "reflection"):
        router_rows = by_router[router]
        errors = [row for row in router_rows if not int(row["correct"])]
        category_counts = Counter(str(row["category"]) for row in errors)
        dataset_counts = Counter(str(row["dataset"]) for row in errors)
        lines.extend(
            [
                f"## {METHOD_LABELS[router]}",
                "",
                f"- Errors: {len(errors)} / {len(router_rows)}",
                f"- By category: {dict(sorted(category_counts.items()))}",
                f"- Top datasets: {dict(dataset_counts.most_common(5))}",
                "",
            ]
        )

    false_accepts: list[dict[str, Any]] = []
    unnecessary_escalations: list[dict[str, Any]] = []
    model_disagreements = Counter()
    for task_id, task_rows in by_id.items():
        cheap = task_rows.get("always_cheap")
        strong = task_rows.get("always_strong")
        reflection = task_rows.get("reflection")
        if not cheap or not strong or not reflection:
            continue
        model_disagreements[(int(cheap["correct"]), int(strong["correct"]))] += 1
        if not int(cheap["correct"]) and not int(reflection["escalated"]):
            false_accepts.append(reflection)
        if int(cheap["correct"]) and int(reflection["escalated"]):
            unnecessary_escalations.append(reflection)

    lines.extend(
        [
            "## Reflection Diagnostics",
            "",
            f"- Both models correct: {model_disagreements[(1, 1)]}",
            f"- Strong fixes a cheap-model error: {model_disagreements[(0, 1)]}",
            f"- Strong regresses a correct cheap-model answer: {model_disagreements[(1, 0)]}",
            f"- Both models wrong: {model_disagreements[(0, 0)]}",
            f"- False accepts (cheap answer wrong but accepted): {len(false_accepts)}",
            f"- Unnecessary escalations (cheap answer correct but escalated): {len(unnecessary_escalations)}",
            "",
            "### Representative False Accepts",
            "",
        ]
    )
    for row in false_accepts[:8]:
        lines.append(
            f"- `{row['id']}` ({row['dataset']}): confidence={row['confidence']}, "
            f"reason=`{row['verification_reason'] or 'accepted'}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metadata(
    path: Path,
    all_tasks: list[TaskExample],
    validation_tasks: list[TaskExample],
    test_tasks: list[TaskExample],
    model_config: dict[str, Any],
    threshold: float,
) -> None:
    providers = model_config.get("providers", {})
    safe_providers = {
        role: {
            key: value
            for key, value in config.items()
            if key not in {"api_key", "api_key_env"}
        }
        for role, config in providers.items()
    }
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "total": len(all_tasks),
            "validation": len(validation_tasks),
            "test": len(test_tasks),
            "categories": dict(Counter(str(task.metadata["category"]) for task in all_tasks)),
            "datasets": dict(Counter(task.dataset for task in all_tasks)),
        },
        "models": safe_providers,
        "selected_confidence_threshold": threshold,
        "selection_rule": "lowest validation cost within 0.02 accuracy of Always Strong; otherwise highest accuracy",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
