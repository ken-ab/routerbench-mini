from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
from tqdm import tqdm

from routerbench_mini.calibration import CalibratedConfidenceEstimator
from routerbench_mini.cli import build_providers, precompute_responses
from routerbench_mini.config import load_costs, load_yaml
from routerbench_mini.features import task_risk_score
from routerbench_mini.providers import ModelResponse, Provider, build_prompt
from routerbench_mini.scoring import is_correct
from routerbench_mini.selection import LearnedQualityGapEstimator
from routerbench_mini.tasks import TaskExample, load_jsonl
from routerbench_mini.v5 import (
    LEARNED_VARIANTS,
    bootstrap_accuracy_interval,
    label_distribution,
    quantile,
    response_api_totals,
    response_cost,
    response_tokens,
    sha256_file,
    task_pair_record,
    verify_file_hash,
    write_csv,
    write_json,
    write_jsonl,
)
from routerbench_mini.verifiers import verify_response


DISPLAY_NAMES = {
    "always_cheap": "Always Cheap",
    "always_strong": "Always Strong",
    "task_aware": "Frozen Task-Aware",
    "learned_text_only": "Learned Text-only",
    "learned_structured_only": "Learned Structured-only",
    "learned_combined": "Learned Combined",
    "reflection": "Reflection",
    "random_learned_rate": "Random@Learned Rate",
    "oracle_learned_rate": "Oracle@Learned Rate",
    "random_reflection_rate": "Random@Reflection Rate",
    "oracle_reflection_rate": "Oracle@Reflection Rate",
    "global_oracle": "Global Oracle",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RouterBench-Mini V5 phase 2: frozen 800-task test.")
    parser.add_argument("--config", default="configs/v5_large_scale.yaml")
    parser.add_argument("--models", default="configs/models.qwen_v5.yaml")
    parser.add_argument("--costs", default="configs/costs.yaml")
    parser.add_argument("--test", default="data/v5_large/test_manifest.jsonl")
    parser.add_argument("--out", default=None)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_dir = Path(args.out or config["paths"]["output_dir"])
    freeze_dir = output_dir / "frozen"
    complete_path = output_dir / "phase2_complete.json"
    if complete_path.exists():
        raise FileExistsError(f"Phase 2 is already complete at {complete_path}.")
    freeze = validate_freeze(args, freeze_dir)
    verify_file_hash(args.test, freeze["frozen_test_manifest_sha256"], "frozen test manifest")
    tasks = load_jsonl(args.test)
    validate_test(tasks)
    started_at = datetime.now(timezone.utc)
    write_json(
        output_dir / "phase2_started.json",
        {
            "started_at": started_at.isoformat(),
            "test_examples": len(tasks),
            "test_manifest_sha256": sha256_file(args.test),
            "freeze_inventory_sha256": freeze["freeze_inventory_sha256"],
        },
    )

    providers = build_providers(args.models)
    costs = load_costs(args.costs)
    precompute_responses(tasks, providers, workers=args.workers)
    cheap = [providers["cheap"].generate(task) for task in tasks]
    strong = [providers["strong"].generate(task) for task in tasks]
    pair_records = [
        task_pair_record(task, cheap_response, strong_response)
        for task, cheap_response, strong_response in zip(tasks, cheap, strong)
    ]
    write_jsonl(output_dir / "test_model_outputs.jsonl", pair_records)
    write_test_labels(pair_records, output_dir)

    reviews = precompute_reviews(tasks, cheap, providers["strong"], workers=args.workers)
    write_jsonl(
        output_dir / "test_review_outputs.jsonl",
        (
            review_record(task, cheap_response, review)
            for task, cheap_response, review in zip(tasks, cheap, reviews)
        ),
    )

    selected_parameters = json.loads(
        (freeze_dir / "selected_parameters.json").read_text(encoding="utf-8")
    )
    learned_estimators: dict[str, LearnedQualityGapEstimator] = {
        mode: joblib.load(freeze_dir / f"learned_{mode}.joblib")
        for mode in LEARNED_VARIANTS
    }
    confidence_estimator: CalibratedConfidenceEstimator = joblib.load(
        freeze_dir / "reflection_calibrator.joblib"
    )
    deterministic_rows, strategy_state = deterministic_predictions(
        tasks=tasks,
        cheap=cheap,
        strong=strong,
        reviews=reviews,
        costs=costs,
        config=config,
        parameters=selected_parameters,
        learned_estimators=learned_estimators,
        confidence_estimator=confidence_estimator,
    )
    write_csv(output_dir / "test_predictions.csv", deterministic_rows)

    random_run_rows = matched_random_runs(
        tasks=tasks,
        cheap=cheap,
        strong=strong,
        reviews=reviews,
        costs=costs,
        learned_budget=int(strategy_state["learned_budget"]),
        reflection_budget=int(strategy_state["reflection_budget"]),
        runs=int(config["random_baseline_runs"]),
        seed=int(config["random_baseline_seed"]),
    )
    write_csv(output_dir / "random_matched_runs.csv", random_run_rows)
    random_summary = aggregate_random_runs(random_run_rows)
    write_csv(output_dir / "random_matched_summary.csv", random_summary)

    deterministic_slices = summarize_prediction_slices(deterministic_rows)
    random_slices = random_slice_rows(random_summary)
    all_slices = [*deterministic_slices, *random_slices]
    write_csv(output_dir / "test_summary_all_slices.csv", all_slices)

    overall = [row for row in deterministic_slices if row["slice_type"] == "all"]
    overall.extend(
        random_slice_to_summary(row)
        for row in random_summary
        if row["slice_type"] == "all"
    )
    overall = add_relative_metrics(overall)
    write_csv(output_dir / "test_summary.csv", overall)
    write_csv(output_dir / "main_results.csv", main_results(overall))
    write_csv(output_dir / "learned_ablation.csv", learned_ablation(overall))
    write_csv(output_dir / "summary_standard_vs_hard.csv", slice_table(all_slices, "difficulty_group"))
    write_csv(output_dir / "summary_by_task_family.csv", slice_table(all_slices, "task_family"))
    write_csv(output_dir / "summary_by_dataset.csv", slice_table(all_slices, "dataset"))
    write_csv(output_dir / "task_family_accuracy_table.csv", task_family_accuracy_table(all_slices))

    comparisons = matched_comparisons(
        overall,
        random_run_rows,
        deterministic_rows,
        strategy_state,
    )
    write_csv(output_dir / "matched_random_oracle_comparisons.csv", comparisons)
    paired = paired_bootstrap(deterministic_rows)
    write_csv(output_dir / "paired_bootstrap_comparisons.csv", paired)

    api_totals = {
        "test_cheap_solve": response_api_totals(cheap),
        "test_strong_solve": response_api_totals(strong),
        "test_strong_review": response_api_totals(reviews),
        "note": "All 800 review outputs were precomputed once for matched-random and oracle counterfactuals; strategy cost tables count only calls selected by each policy.",
    }
    write_json(output_dir / "test_api_totals.json", api_totals)
    write_failure_analysis(deterministic_rows, output_dir / "failure_analysis.md")
    write_final_report(
        output_dir / "final_report.zh-CN.md",
        overall=overall,
        slices=all_slices,
        comparisons=comparisons,
        labels=pair_records,
        freeze=freeze,
        api_totals=api_totals,
    )
    write_reproducibility(output_dir / "reproducibility.md")
    completed_at = datetime.now(timezone.utc)
    completion = {
        "study_version": config["version"],
        "phase": "frozen_test_complete",
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "test_examples": len(tasks),
        "test_manifest_sha256": sha256_file(args.test),
        "freeze_inventory_sha256": freeze["freeze_inventory_sha256"],
        "random_runs_per_matched_baseline": int(config["random_baseline_runs"]),
        "output_files": sorted(
            str(path.relative_to(output_dir))
            for path in output_dir.iterdir()
            if path.is_file() and path != complete_path
        ),
        "no_commit_or_push_performed": True,
    }
    write_json(complete_path, completion)
    print(f"Phase 2 complete: {output_dir / 'test_summary.csv'}")
    for row in main_results(overall):
        print(
            f"{row['strategy']}: accuracy={row['accuracy']:.4f}, "
            f"cost={row['avg_cost']:.8f}, strong_rate={row['strong_usage_rate']:.4f}"
        )


def validate_freeze(args: argparse.Namespace, freeze_dir: Path) -> dict[str, Any]:
    marker = freeze_dir / "phase1_freeze.json"
    if not marker.exists():
        raise FileNotFoundError("Phase 2 is locked until phase1_freeze.json exists.")
    freeze = json.loads(marker.read_text(encoding="utf-8"))
    if not freeze.get("phase2_permitted") or freeze.get("test_examples_used_in_phase1") != 0:
        raise ValueError("Phase 1 freeze does not certify an untouched final test.")
    verify_file_hash(freeze_dir / "freeze_inventory.json", freeze["freeze_inventory_sha256"], "freeze inventory")
    inventory = json.loads((freeze_dir / "freeze_inventory.json").read_text(encoding="utf-8"))
    for path, expected in inventory.items():
        verify_file_hash(path, expected, f"frozen file {path}")
    verify_file_hash(args.config, freeze["protocol_config_sha256"], "protocol config")
    verify_file_hash(args.models, freeze["model_config_sha256"], "model config")
    verify_file_hash(args.costs, freeze["cost_config_sha256"], "cost config")
    return freeze


def validate_test(tasks: Sequence[TaskExample]) -> None:
    if len(tasks) != 800:
        raise ValueError(f"Phase 2 requires exactly 800 frozen test tasks; got {len(tasks)}")
    if any(task.fold_id is not None for task in tasks):
        raise ValueError("Final test tasks must not carry development fold IDs.")
    import hashlib

    for task in tasks:
        expected = hashlib.sha256(build_prompt(task).encode("utf-8")).hexdigest()
        if task.prompt_hash != expected:
            raise ValueError(f"Prompt hash changed for {task.canonical_id or task.id}")


def precompute_reviews(
    tasks: Sequence[TaskExample],
    cheap: Sequence[ModelResponse],
    provider: Provider,
    *,
    workers: int,
) -> list[ModelResponse]:
    output: list[ModelResponse | None] = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(provider.review_and_correct, task, candidate): index
            for index, (task, candidate) in enumerate(zip(tasks, cheap))
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="test reviews"):
            output[futures[future]] = future.result()
    if any(response is None for response in output):
        raise RuntimeError("A test review response is missing.")
    return [response for response in output if response is not None]


def deterministic_predictions(
    *,
    tasks: Sequence[TaskExample],
    cheap: Sequence[ModelResponse],
    strong: Sequence[ModelResponse],
    reviews: Sequence[ModelResponse],
    costs: dict[str, float],
    config: dict[str, Any],
    parameters: dict[str, Any],
    learned_estimators: dict[str, LearnedQualityGapEstimator],
    confidence_estimator: CalibratedConfidenceEstimator,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selections: dict[str, list[bool]] = {
        "always_cheap": [False] * len(tasks),
        "always_strong": [True] * len(tasks),
        "task_aware": [
            task_risk_score(task) >= float(config["task_aware_router"]["risk_threshold"])
            for task in tasks
        ],
    }
    routing_scores: dict[str, list[float | None]] = {
        name: [None] * len(tasks) for name in selections
    }
    thresholds: dict[str, float | None] = {name: None for name in selections}
    for mode, estimator in learned_estimators.items():
        name = f"learned_{mode}"
        scores = estimator.predict_advantages(tasks)
        threshold = float(parameters["learned"][mode]["threshold"])
        selections[name] = [score >= threshold for score in scores]
        routing_scores[name] = list(scores)
        thresholds[name] = threshold

    reflection_probabilities = [
        confidence_estimator.predict_correctness(task, response)
        for task, response in zip(tasks, cheap)
    ]
    reflection_threshold = float(parameters["reflection"]["threshold"])
    reflection_selection = [
        verify_response(
            task,
            response,
            confidence_threshold=reflection_threshold,
            estimated_confidence=probability,
        ).should_escalate
        for task, response, probability in zip(tasks, cheap, reflection_probabilities)
    ]

    rows: list[dict[str, Any]] = []
    for strategy, usage in selections.items():
        for index, task in enumerate(tasks):
            response = strong[index] if usage[index] else cheap[index]
            rows.append(
                prediction_record(
                    task,
                    strategy,
                    response,
                    [response],
                    usage[index],
                    costs,
                    routing_score=routing_scores[strategy][index],
                    threshold=thresholds[strategy],
                )
            )
    for index, task in enumerate(tasks):
        use_review = reflection_selection[index]
        response = reviews[index] if use_review else cheap[index]
        rows.append(
            prediction_record(
                task,
                "reflection",
                response,
                [cheap[index], reviews[index]] if use_review else [cheap[index]],
                use_review,
                costs,
                routing_score=reflection_probabilities[index],
                threshold=reflection_threshold,
                verification_reason=verify_response(
                    task,
                    cheap[index],
                    confidence_threshold=reflection_threshold,
                    estimated_confidence=reflection_probabilities[index],
                ).reason,
            )
        )

    learned_budget = sum(selections["learned_combined"])
    reflection_budget = sum(reflection_selection)
    oracle_learned = oracle_budget_selection(tasks, cheap, strong, learned_budget)
    oracle_reflection = oracle_budget_selection(tasks, cheap, reviews, reflection_budget)
    global_oracle = [
        not is_correct(task, cheap_response) and is_correct(task, strong_response)
        for task, cheap_response, strong_response in zip(tasks, cheap, strong)
    ]
    for strategy, usage, alternate, cascade in (
        ("oracle_learned_rate", oracle_learned, strong, False),
        ("oracle_reflection_rate", oracle_reflection, reviews, True),
        ("global_oracle", global_oracle, strong, False),
    ):
        for index, task in enumerate(tasks):
            response = alternate[index] if usage[index] else cheap[index]
            responses = (
                [cheap[index], alternate[index]]
                if cascade and usage[index]
                else [response]
            )
            rows.append(
                prediction_record(
                    task,
                    strategy,
                    response,
                    responses,
                    usage[index],
                    costs,
                )
            )
    return rows, {
        "learned_budget": learned_budget,
        "reflection_budget": reflection_budget,
        "learned_selection": selections["learned_combined"],
        "reflection_selection": reflection_selection,
    }


def oracle_budget_selection(
    tasks: Sequence[TaskExample],
    cheap: Sequence[ModelResponse],
    alternate: Sequence[ModelResponse],
    budget: int,
) -> list[bool]:
    gains = [
        int(is_correct(task, alternate_response)) - int(is_correct(task, cheap_response))
        for task, cheap_response, alternate_response in zip(tasks, cheap, alternate)
    ]
    order = sorted(
        range(len(tasks)),
        key=lambda index: (-gains[index], str(tasks[index].canonical_id or tasks[index].id)),
    )
    selected = set(order[:budget])
    return [index in selected for index in range(len(tasks))]


def prediction_record(
    task: TaskExample,
    strategy: str,
    response: ModelResponse,
    responses: Sequence[ModelResponse],
    strong_used: bool,
    costs: dict[str, float],
    *,
    routing_score: float | None = None,
    threshold: float | None = None,
    verification_reason: str = "",
) -> dict[str, Any]:
    prompt_tokens = sum(response_tokens(value)[0] for value in responses)
    completion_tokens = sum(response_tokens(value)[1] for value in responses)
    return {
        "canonical_id": task.canonical_id or task.id,
        "dataset": task.dataset,
        "task_family": task.metadata.get("category", task.task_type),
        "task_type": task.task_type,
        "task_subtype": task.task_subtype,
        "difficulty_group": task.difficulty_group,
        "strategy": strategy,
        "selected_role": "strong" if strong_used else "cheap",
        "selected_model": response.model,
        "correct": int(is_correct(task, response)),
        "cost": round(sum(response_cost(value, value.role, costs) for value in responses), 10),
        "latency_ms": round(sum(value.latency_ms for value in responses), 2),
        "strong_used": int(strong_used),
        "strong_calls": int(strong_used),
        "cheap_calls": sum(value.role == "cheap" for value in responses),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "routing_score": routing_score if routing_score is not None else "",
        "threshold": threshold if threshold is not None else "",
        "verification_reason": verification_reason,
        "answer": response.answer,
    }


def matched_random_runs(
    *,
    tasks: Sequence[TaskExample],
    cheap: Sequence[ModelResponse],
    strong: Sequence[ModelResponse],
    reviews: Sequence[ModelResponse],
    costs: dict[str, float],
    learned_budget: int,
    reflection_budget: int,
    runs: int,
    seed: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for offset in range(runs):
        run_seed = seed + offset
        rng = random.Random(run_seed)
        for strategy, budget, alternate, cascade in (
            ("random_learned_rate", learned_budget, strong, False),
            ("random_reflection_rate", reflection_budget, reviews, True),
        ):
            selected = set(rng.sample(range(len(tasks)), budget))
            rows = []
            for index, task in enumerate(tasks):
                use_strong = index in selected
                response = alternate[index] if use_strong else cheap[index]
                responses = [cheap[index], alternate[index]] if cascade and use_strong else [response]
                rows.append(
                    prediction_record(
                        task,
                        strategy,
                        response,
                        responses,
                        use_strong,
                        costs,
                    )
                )
            for slice_type, slice_value, slice_rows in prediction_slices(rows):
                output.append(
                    {
                        "strategy": strategy,
                        "seed": run_seed,
                        "slice_type": slice_type,
                        "slice_value": slice_value,
                        **summarize_rows(slice_rows),
                    }
                )
    return output


def summarize_prediction_slices(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["strategy"])].append(row)
    for strategy, strategy_rows in sorted(grouped.items()):
        for slice_type, slice_value, values in prediction_slices(strategy_rows):
            summary = summarize_rows(values)
            outcomes = [int(row["correct"]) for row in values]
            low, high = bootstrap_accuracy_interval(
                outcomes,
                seed=20260720 + sum(ord(char) for char in f"{strategy}:{slice_type}:{slice_value}"),
            )
            output.append(
                {
                    "strategy": strategy,
                    "display_name": DISPLAY_NAMES[strategy],
                    "slice_type": slice_type,
                    "slice_value": slice_value,
                    **summary,
                    "accuracy_ci_low": round(low, 6),
                    "accuracy_ci_high": round(high, 6),
                }
            )
    return output


def prediction_slices(
    rows: Sequence[dict[str, Any]],
) -> Iterable[tuple[str, str, list[dict[str, Any]]]]:
    yield "all", "all", list(rows)
    for field in ("difficulty_group", "task_family", "dataset"):
        for value in sorted({str(row[field]) for row in rows}):
            yield field, value, [row for row in rows if str(row[field]) == value]


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    return {
        "total": total,
        "accuracy": round(sum(int(row["correct"]) for row in rows) / total, 8),
        "total_cost": round(sum(float(row["cost"]) for row in rows), 8),
        "avg_cost": round(sum(float(row["cost"]) for row in rows) / total, 10),
        "total_latency_ms": round(sum(float(row["latency_ms"]) for row in rows), 2),
        "avg_latency_ms": round(sum(float(row["latency_ms"]) for row in rows) / total, 2),
        "strong_usage_rate": round(sum(int(row["strong_used"]) for row in rows) / total, 8),
        "total_strong_calls": sum(int(row["strong_calls"]) for row in rows),
        "total_cheap_calls": sum(int(row["cheap_calls"]) for row in rows),
        "prompt_tokens": sum(int(row["prompt_tokens"]) for row in rows),
        "completion_tokens": sum(int(row["completion_tokens"]) for row in rows),
    }


def aggregate_random_runs(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["strategy"]), str(row["slice_type"]), str(row["slice_value"]))].append(row)
    output = []
    metrics = ("accuracy", "avg_cost", "avg_latency_ms", "strong_usage_rate")
    for (strategy, slice_type, slice_value), values in sorted(grouped.items()):
        row: dict[str, Any] = {
            "strategy": strategy,
            "display_name": DISPLAY_NAMES[strategy],
            "slice_type": slice_type,
            "slice_value": slice_value,
            "runs": len(values),
            "total": values[0]["total"],
        }
        for metric in metrics:
            samples = [float(value[metric]) for value in values]
            row[f"mean_{metric}"] = round(statistics.fmean(samples), 10)
            row[f"std_{metric}"] = round(statistics.pstdev(samples), 10)
            row[f"{metric}_ci_low"] = round(quantile(samples, 0.025), 10)
            row[f"{metric}_ci_high"] = round(quantile(samples, 0.975), 10)
        row["mean_total_strong_calls"] = round(
            statistics.fmean(float(value["total_strong_calls"]) for value in values), 4
        )
        row["mean_total_cheap_calls"] = round(
            statistics.fmean(float(value["total_cheap_calls"]) for value in values), 4
        )
        output.append(row)
    return output


def random_slice_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [random_slice_to_summary(row) for row in rows]


def random_slice_to_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategy": row["strategy"],
        "display_name": row["display_name"],
        "slice_type": row["slice_type"],
        "slice_value": row["slice_value"],
        "total": row["total"],
        "accuracy": row["mean_accuracy"],
        "avg_cost": row["mean_avg_cost"],
        "avg_latency_ms": row["mean_avg_latency_ms"],
        "strong_usage_rate": row["mean_strong_usage_rate"],
        "total_strong_calls": row["mean_total_strong_calls"],
        "total_cheap_calls": row["mean_total_cheap_calls"],
        "accuracy_ci_low": row["accuracy_ci_low"],
        "accuracy_ci_high": row["accuracy_ci_high"],
        "random_accuracy_std": row["std_accuracy"],
    }


def add_relative_metrics(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = next(row for row in rows if row["strategy"] == "always_strong")
    output = []
    for row in rows:
        output.append(
            {
                **row,
                "accuracy_gap_vs_always_strong": round(
                    float(row["accuracy"]) - float(baseline["accuracy"]), 8
                ),
                "cost_reduction_vs_always_strong": round(
                    1 - float(row["avg_cost"]) / float(baseline["avg_cost"]), 8
                )
                if float(baseline["avg_cost"])
                else 0.0,
                "latency_reduction_vs_always_strong": round(
                    1 - float(row["avg_latency_ms"]) / float(baseline["avg_latency_ms"]), 8
                )
                if float(baseline["avg_latency_ms"])
                else 0.0,
            }
        )
    return sorted(output, key=lambda row: str(row["strategy"]))


def main_results(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    order = (
        "always_cheap",
        "always_strong",
        "task_aware",
        "learned_combined",
        "reflection",
        "random_learned_rate",
        "oracle_learned_rate",
        "random_reflection_rate",
        "oracle_reflection_rate",
        "global_oracle",
    )
    by_strategy = {str(row["strategy"]): row for row in rows}
    return [by_strategy[strategy] for strategy in order]


def learned_ablation(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_strategy = {str(row["strategy"]): row for row in rows}
    return [
        by_strategy["learned_text_only"],
        by_strategy["learned_structured_only"],
        by_strategy["learned_combined"],
    ]


def slice_table(
    rows: Sequence[dict[str, Any]], slice_type: str
) -> list[dict[str, Any]]:
    return [row for row in rows if row["slice_type"] == slice_type]


def task_family_accuracy_table(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["slice_type"] == "task_family"]
    grouped: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in selected:
        grouped[str(row["strategy"])][str(row["slice_value"])] = row["accuracy"]
    return [
        {
            "strategy": strategy,
            "display_name": DISPLAY_NAMES[strategy],
            "text": values.get("text", ""),
            "vision": values.get("vision", ""),
            "tool": values.get("tool", ""),
        }
        for strategy, values in sorted(grouped.items())
    ]


def matched_comparisons(
    overall: Sequence[dict[str, Any]],
    random_runs: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    by_strategy = {str(row["strategy"]): row for row in overall}
    output = []
    pairs = (
        ("learned_combined", "random_learned_rate", "oracle_learned_rate", state["learned_budget"]),
        ("reflection", "random_reflection_rate", "oracle_reflection_rate", state["reflection_budget"]),
    )
    for strategy, random_strategy, oracle_strategy, budget in pairs:
        strategy_accuracy = float(by_strategy[strategy]["accuracy"])
        random_accuracies = [
            float(row["accuracy"])
            for row in random_runs
            if row["strategy"] == random_strategy and row["slice_type"] == "all"
        ]
        gains = [strategy_accuracy - value for value in random_accuracies]
        oracle_accuracy = float(by_strategy[oracle_strategy]["accuracy"])
        output.append(
            {
                "strategy": strategy,
                "matched_random": random_strategy,
                "matched_oracle": oracle_strategy,
                "strong_call_budget": budget,
                "strong_usage_rate": round(budget / 800, 8),
                "strategy_accuracy": strategy_accuracy,
                "matched_random_mean_accuracy": round(statistics.fmean(random_accuracies), 8),
                "gain_over_random": round(statistics.fmean(gains), 8),
                "gain_over_random_ci_low": round(quantile(gains, 0.025), 8),
                "gain_over_random_ci_high": round(quantile(gains, 0.975), 8),
                "random_ge_strategy_p": round(
                    (1 + sum(value >= strategy_accuracy for value in random_accuracies))
                    / (len(random_accuracies) + 1),
                    8,
                ),
                "matched_oracle_accuracy": oracle_accuracy,
                "regret_to_matched_oracle": round(oracle_accuracy - strategy_accuracy, 8),
            }
        )
    return output


def paired_bootstrap(
    rows: Sequence[dict[str, Any]], *, samples: int = 4000
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, int]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["strategy"])][str(row["canonical_id"])] = int(row["correct"])
    comparisons = (
        ("learned_combined", "task_aware"),
        ("reflection", "task_aware"),
        ("learned_combined", "always_strong"),
        ("reflection", "always_strong"),
        ("learned_combined", "learned_text_only"),
        ("learned_combined", "learned_structured_only"),
    )
    output = []
    for offset, (strategy, baseline) in enumerate(comparisons):
        ids = sorted(grouped[strategy])
        differences = [grouped[strategy][task_id] - grouped[baseline][task_id] for task_id in ids]
        observed = sum(differences) / len(differences)
        rng = random.Random(20260730 + offset)
        draws = [sum(rng.choice(differences) for _ in differences) / len(differences) for _ in range(samples)]
        output.append(
            {
                "strategy": strategy,
                "baseline": baseline,
                "accuracy_difference": round(observed, 8),
                "ci_low": round(quantile(draws, 0.025), 8),
                "ci_high": round(quantile(draws, 0.975), 8),
                "two_sided_p": round(
                    min(1.0, 2 * min(sum(value <= 0 for value in draws), sum(value >= 0 for value in draws)) / samples),
                    8,
                ),
            }
        )
    return output


def write_test_labels(records: Sequence[dict[str, Any]], output_dir: Path) -> None:
    write_csv(
        output_dir / "test_label_distribution.csv",
        [
            *label_distribution(records),
            *label_distribution(records, "task_family"),
            *label_distribution(records, "dataset"),
        ],
    )


def review_record(task: TaskExample, cheap: ModelResponse, review: ModelResponse) -> dict[str, Any]:
    cheap_correct = int(is_correct(task, cheap))
    review_correct = int(is_correct(task, review))
    return {
        "canonical_id": task.canonical_id or task.id,
        "dataset": task.dataset,
        "task_family": task.metadata.get("category", task.task_type),
        "difficulty_group": task.difficulty_group,
        "cheap_correct": cheap_correct,
        "review_correct": review_correct,
        "review_outcome": (
            "beneficial" if not cheap_correct and review_correct else
            "harmful" if cheap_correct and not review_correct else
            "kept_correct" if cheap_correct else "still_wrong"
        ),
        "review_action": review.metadata.get("review_action"),
        "review_changed": bool(review.metadata.get("review_changed", False)),
        "cheap_answer": cheap.answer,
        "review_answer": review.answer,
        "review_raw_output": review.raw_text,
        "review_metadata": review.metadata,
        "observed_latency_ms": review.latency_ms,
    }


def write_failure_analysis(rows: Sequence[dict[str, Any]], path: Path) -> None:
    by_strategy: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_strategy[str(row["strategy"])][str(row["canonical_id"])] = row
    ids = sorted(by_strategy["always_cheap"])
    learned_missed = [
        task_id for task_id in ids
        if not int(by_strategy["always_cheap"][task_id]["correct"])
        and int(by_strategy["always_strong"][task_id]["correct"])
        and not int(by_strategy["learned_combined"][task_id]["strong_used"])
    ]
    learned_harmful = [
        task_id for task_id in ids
        if int(by_strategy["always_cheap"][task_id]["correct"])
        and not int(by_strategy["always_strong"][task_id]["correct"])
        and int(by_strategy["learned_combined"][task_id]["strong_used"])
    ]
    reflection_missed = [
        task_id for task_id in ids
        if not int(by_strategy["always_cheap"][task_id]["correct"])
        and int(by_strategy["oracle_reflection_rate"][task_id]["correct"])
        and not int(by_strategy["reflection"][task_id]["strong_used"])
    ]
    reflection_harmful = [
        task_id for task_id in ids
        if int(by_strategy["always_cheap"][task_id]["correct"])
        and not int(by_strategy["reflection"][task_id]["correct"])
    ]
    lines = [
        "# V5 Failure Analysis",
        "",
        f"- Learned missed beneficial Strong upgrades: {len(learned_missed)}",
        f"- Learned selected harmful Strong regressions: {len(learned_harmful)}",
        f"- Reflection false accepts among review-fixable cases: {len(reflection_missed)}",
        f"- Reflection harmful final answers after a correct Cheap answer: {len(reflection_harmful)}",
        "",
        "## Example IDs",
        "",
        f"- Learned missed: {learned_missed[:20]}",
        f"- Learned harmful: {learned_harmful[:20]}",
        f"- Reflection missed: {reflection_missed[:20]}",
        f"- Reflection harmful: {reflection_harmful[:20]}",
        "",
        "The corresponding raw outputs and grader fields are in `test_model_outputs.jsonl` and `test_review_outputs.jsonl`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_report(
    path: Path,
    *,
    overall: Sequence[dict[str, Any]],
    slices: Sequence[dict[str, Any]],
    comparisons: Sequence[dict[str, Any]],
    labels: Sequence[dict[str, Any]],
    freeze: dict[str, Any],
    api_totals: dict[str, Any],
) -> None:
    by_strategy = {str(row["strategy"]): row for row in overall}
    comparison_map = {str(row["strategy"]): row for row in comparisons}
    label_counts = Counter(str(row["pair_outcome"]) for row in labels)
    hard = {
        str(row["strategy"]): row
        for row in slices
        if row["slice_type"] == "difficulty_group" and row["slice_value"] == "hard"
    }
    standard = {
        str(row["strategy"]): row
        for row in slices
        if row["slice_type"] == "difficulty_group" and row["slice_value"] == "standard"
    }
    learned = by_strategy["learned_combined"]
    reflection = by_strategy["reflection"]
    task_aware = by_strategy["task_aware"]
    old_v4 = load_old_v4()
    ablation_best = max(
        ("text_only", "structured_only", "combined"),
        key=lambda mode: float(by_strategy[f"learned_{mode}"]["accuracy"]),
    )
    lines = [
        "# RouterBench-Mini V5 大规模实验报告",
        "",
        "## 协议",
        "",
        "V5 使用 3,200 道开发题完成所有训练、五折样本外预测、阈值选择与冻结；随后只读取一次 800 道独立测试题。测试集没有参与词表拟合、特征选择、阈值选择或提示词修改。",
        "",
        f"- 开发集 SHA-256：`{freeze['development_manifest_sha256']}`",
        f"- 测试集 SHA-256：`{freeze['frozen_test_manifest_sha256']}`",
        f"- Cheap：`{freeze['models']['cheap']['model']}`",
        f"- Strong：`{freeze['models']['strong']['model']}`",
        "",
        "## 主结果",
        "",
        "| 方法 | Accuracy | Avg Cost | Avg Latency (ms) | Strong Rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in main_results(overall):
        lines.append(
            f"| {DISPLAY_NAMES[str(row['strategy'])]} | {float(row['accuracy']):.2%} | "
            f"{float(row['avg_cost']):.8f} | {float(row['avg_latency_ms']):.1f} | "
            f"{float(row['strong_usage_rate']):.2%} |"
        )
    lines.extend(
        [
            "",
            "## 标签与路由空间",
            "",
            f"- Cheap 对、Strong 对：{label_counts['cheap_correct_strong_correct']}",
            f"- Cheap 错、Strong 对（真正有升级价值）：{label_counts['cheap_wrong_strong_correct']}",
            f"- Cheap 对、Strong 错（升级会回退）：{label_counts['cheap_correct_strong_wrong']}",
            f"- Cheap 错、Strong 错：{label_counts['cheap_wrong_strong_wrong']}",
            "",
            "## 对 11 个问题的回答",
            "",
            f"1. Learned Combined 在 V5 为 {float(learned['accuracy']):.2%}；V4 为 {old_v4.get('learned_cost_aware', float('nan')):.2%}。这是不同测试集上的描述性比较，不能单独归因于开发集扩大。",
            f"2. Reflection 在 V5 为 {float(reflection['accuracy']):.2%}；V4 为 {old_v4.get('reflection', float('nan')):.2%}，同样不能把差异只归因于数据规模。",
            f"3. Learned Combined 相比 Task-Aware 的准确率差为 {float(learned['accuracy']) - float(task_aware['accuracy']):+.2%}。",
            f"4. Reflection 相比 Task-Aware 的准确率差为 {float(reflection['accuracy']) - float(task_aware['accuracy']):+.2%}。",
            f"5. Learned 相比 matched random 的平均增益为 {float(comparison_map['learned_combined']['gain_over_random']):+.2%}，随机化 p={float(comparison_map['learned_combined']['random_ge_strategy_p']):.4f}。",
            f"6. Reflection 相比 matched random 的平均增益为 {float(comparison_map['reflection']['gain_over_random']):+.2%}，随机化 p={float(comparison_map['reflection']['random_ge_strategy_p']):.4f}。",
            f"7. Learned 距 matched oracle 仍有 {float(comparison_map['learned_combined']['regret_to_matched_oracle']):.2%} 准确率差距。",
            f"8. 三种 Learned 特征中测试准确率最高的是 `{ablation_best}`。",
            f"9. Combined 相比 Text-only 的差为 {float(by_strategy['learned_combined']['accuracy']) - float(by_strategy['learned_text_only']['accuracy']):+.2%}；这直接衡量 13 维结构特征在文本特征之上的互补价值。",
            f"10. Learned 的 hard/standard 准确率为 {float(hard['learned_combined']['accuracy']):.2%}/{float(standard['learned_combined']['accuracy']):.2%}；Reflection 为 {float(hard['reflection']['accuracy']):.2%}/{float(standard['reflection']['accuracy']):.2%}。",
            "11. 主要限制由结果共同判断：有效 +1/-1 标签数量、Strong 本身回退、Learned 与 matched oracle 的差距，以及 Reflection 概率与真实错误的区分能力。不能仅用数据量解释。",
            "",
            "## 结论边界",
            "",
            conclusion_sentence(learned, reflection, task_aware),
            "",
            "## API 统计",
            "",
            f"- 测试 Cheap solve：{api_totals['test_cheap_solve']['logical_calls']} 次，{api_totals['test_cheap_solve']['total_tokens']} tokens。",
            f"- 测试 Strong solve：{api_totals['test_strong_solve']['logical_calls']} 次，{api_totals['test_strong_solve']['total_tokens']} tokens。",
            f"- 测试 Strong review：{api_totals['test_strong_review']['logical_calls']} 次，{api_totals['test_strong_review']['total_tokens']} tokens。",
            "",
            "完整的标准/困难、任务族、数据集、消融、随机、Oracle、失败案例和逐题输出均保存在同一结果目录。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def conclusion_sentence(
    learned: dict[str, Any], reflection: dict[str, Any], task_aware: dict[str, Any]
) -> str:
    if float(learned["accuracy"]) > float(task_aware["accuracy"]) and float(reflection["accuracy"]) > float(task_aware["accuracy"]):
        return "在将开发集扩展至 3,200 题后，学习式路由和响应侧升级策略均超过冻结的 Task-Aware 基线；这支持数据不足是旧版本限制之一，但不证明它是唯一原因。"
    if float(learned["accuracy"]) > float(task_aware["accuracy"]) or float(reflection["accuracy"]) > float(task_aware["accuracy"]):
        return "扩大开发数据改善了部分方法，但学习式和响应侧路由未同时稳定超过冻结的任务感知规则，说明数据规模并非唯一限制，特征、标签和升级判据仍需改进。"
    return "即使将开发集扩展至 3,200 题，学习式路由与反思式升级仍未获得对冻结 Task-Aware 的稳定优势，说明旧版本的问题不能主要归因于开发数据不足。"


def load_old_v4() -> dict[str, float]:
    path = Path("results/qwen3.5-v4-study/test_summary.csv")
    if not path.exists():
        return {}
    import csv

    with path.open(encoding="utf-8") as handle:
        return {str(row["router"]): float(row["accuracy"]) for row in csv.DictReader(handle)}


def write_reproducibility(path: Path) -> None:
    status = subprocess_status()
    lines = [
        "# V5 Reproducibility",
        "",
        "```bash",
        "../.venv-routerbench-mini-py/bin/python scripts/build_v5_data.py --force",
        "../.venv-routerbench-mini-py/bin/python scripts/run_v5_phase1.py --workers 16",
        "../.venv-routerbench-mini-py/bin/python scripts/run_v5_phase2.py --workers 16",
        "../.venv-routerbench-mini-py/bin/python -m pytest -q",
        "```",
        "",
        "## Modified Files",
        "",
        *[f"- `{line}`" for line in status],
        "",
        "## Unresolved",
        "",
        "- API backends may change implementation behind the same model alias; response metadata and run timestamps are retained.",
        "- Exact-match and task-specific automatic graders can under-credit semantically equivalent open answers.",
        "- V5 does not perform the intentionally excluded 300/800/1600/3200 training-size scaling study.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def subprocess_status() -> list[str]:
    import subprocess

    output = subprocess.run(
        ["git", "status", "--short"], check=True, text=True, capture_output=True
    ).stdout.splitlines()
    return output


if __name__ == "__main__":
    main()
