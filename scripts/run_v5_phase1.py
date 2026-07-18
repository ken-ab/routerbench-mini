from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import statistics
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

import joblib
from tqdm import tqdm

from routerbench_mini.calibration import (
    CalibratedConfidenceEstimator,
    cross_validated_correctness_probabilities,
)
from routerbench_mini.cli import build_providers, precompute_responses
from routerbench_mini.config import load_costs, load_yaml
from routerbench_mini.features import task_feature_names
from routerbench_mini.providers import (
    PROMPT_VERSION,
    ModelResponse,
    Provider,
    build_prompt,
    build_review_prompt,
)
from routerbench_mini.scoring import is_correct
from routerbench_mini.selection import LearnedQualityGapEstimator, cross_validated_advantages
from routerbench_mini.tasks import TaskExample, load_jsonl
from routerbench_mini.v5 import (
    LEARNED_VARIANTS,
    correlation,
    fold_policy_rows,
    frozen_task_aware_selection,
    label_distribution,
    learned_estimator_kwargs,
    policy_summary,
    response_api_totals,
    safe_model_config,
    sha256_file,
    sha256_json,
    task_pair_record,
    tune_learned_threshold,
    tune_reflection_threshold,
    verify_file_hash,
    write_csv,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RouterBench-Mini V5 phase 1: develop and freeze.")
    parser.add_argument("--config", default="configs/v5_large_scale.yaml")
    parser.add_argument("--models", default="configs/models.qwen_v5.yaml")
    parser.add_argument("--costs", default="configs/costs.yaml")
    parser.add_argument("--development", default="data/v5_large/development_manifest.jsonl")
    parser.add_argument("--out", default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--force-freeze", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_dir = Path(args.out or config["paths"]["output_dir"])
    freeze_dir = output_dir / "frozen"
    marker = freeze_dir / "phase1_freeze.json"
    if marker.exists() and not args.force_freeze:
        raise FileExistsError(f"Phase 1 is already frozen at {marker}; use --force-freeze to rebuild it.")
    if (output_dir / "phase2_complete.json").exists():
        raise FileExistsError("Phase 2 already exists; phase 1 cannot be rebuilt in the same versioned directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    freeze_dir.mkdir(parents=True, exist_ok=True)

    manifest_hashes = json.loads(Path(config["paths"]["manifest_hashes"]).read_text(encoding="utf-8"))
    verify_file_hash(args.development, manifest_hashes["development_manifest"], "development manifest")
    tasks = load_jsonl(args.development)
    validate_development(tasks, config)
    providers = build_providers(args.models)
    costs = load_costs(args.costs)
    started_at = datetime.now(timezone.utc)
    write_json(
        output_dir / "phase1_started.json",
        {
            "started_at": started_at.isoformat(),
            "development_examples": len(tasks),
            "development_manifest_sha256": manifest_hashes["development_manifest"],
            "final_test_loaded": False,
            "pid": os.getpid(),
        },
    )

    precompute_responses(tasks, providers, workers=args.workers)
    cheap = [providers["cheap"].generate(task) for task in tasks]
    strong = [providers["strong"].generate(task) for task in tasks]
    pair_records = [
        task_pair_record(task, cheap_response, strong_response)
        for task, cheap_response, strong_response in zip(tasks, cheap, strong)
    ]
    write_jsonl(output_dir / "development_model_outputs.jsonl", pair_records)
    write_label_reports(pair_records, output_dir)

    reviews = precompute_reviews(tasks, cheap, providers["strong"], workers=args.workers)
    review_records = [
        review_record(task, cheap_response, review)
        for task, cheap_response, review in zip(tasks, cheap, reviews)
    ]
    write_jsonl(output_dir / "development_review_outputs.jsonl", review_records)

    learned_results: dict[str, dict[str, Any]] = {}
    development_predictions: list[dict[str, Any]] = []
    development_summary: list[dict[str, Any]] = []
    folds = int(config["cross_validation_folds"])
    targets = [float(record["quality_gap"]) for record in pair_records]

    development_summary.extend(
        baseline_development_summaries(tasks, cheap, strong, costs, config)
    )
    for mode in LEARNED_VARIANTS:
        kwargs = learned_estimator_kwargs(config, mode)
        oof_scores = cross_validated_advantages(
            tasks,
            cheap,
            strong,
            folds=folds,
            **kwargs,
        )
        threshold, curve = tune_learned_threshold(tasks, cheap, strong, oof_scores, costs)
        write_csv(output_dir / f"development_learned_{mode}_thresholds.csv", curve)
        use_strong = [score >= threshold for score in oof_scores]
        selected = [
            strong_response if selected_strong else cheap_response
            for cheap_response, strong_response, selected_strong in zip(cheap, strong, use_strong)
        ]
        summary = {
            "strategy": f"learned_{mode}",
            **policy_summary(
                tasks,
                selected,
                costs,
                strong_usage=use_strong,
                calls=[["strong"] if value else ["cheap"] for value in use_strong],
                threshold_name="advantage_threshold",
                threshold=threshold,
            ),
        }
        development_summary.append(summary)
        fold_rows = fold_policy_rows(tasks, selected, use_strong, costs)
        for row in fold_rows:
            row["diagnostic_fold_threshold"] = diagnostic_fold_threshold(
                tasks,
                cheap,
                strong,
                costs,
                int(row["fold_id"]),
                "learned",
                scores=oof_scores,
            )
        write_csv(output_dir / f"development_learned_{mode}_fold_metrics.csv", fold_rows)
        estimator = LearnedQualityGapEstimator(**kwargs).fit(tasks, cheap, strong)
        model_path = freeze_dir / f"learned_{mode}.joblib"
        joblib.dump(estimator, model_path)
        score_diagnostics = numeric_diagnostics(oof_scores, targets)
        learned_results[mode] = {
            "threshold": threshold,
            "strong_usage_rate": summary["strong_usage_rate"],
            "oof_accuracy": summary["accuracy"],
            "model_file": model_path.name,
            "model_sha256": sha256_file(model_path),
            "estimator": estimator.diagnostics,
            "oof_scores": score_diagnostics,
            "fold_metrics_file": f"development_learned_{mode}_fold_metrics.csv",
        }
        for task, target, score, selected_strong in zip(tasks, targets, oof_scores, use_strong):
            development_predictions.append(
                {
                    "canonical_id": task.canonical_id or task.id,
                    "fold_id": task.fold_id,
                    "strategy": f"learned_{mode}",
                    "quality_gap": target,
                    "routing_score": score,
                    "threshold": threshold,
                    "selected_role": "strong" if selected_strong else "cheap",
                }
            )

    reflection_config = config["reflection_router"]
    oof_probabilities = cross_validated_correctness_probabilities(
        tasks,
        cheap,
        include_task_features=False,
        folds=folds,
        logistic_regression_c=float(reflection_config["logistic_regression_c"]),
        calibration=str(reflection_config["calibration"]),
        inner_folds=int(reflection_config["inner_folds"]),
    )
    thresholds = reflection_thresholds(reflection_config)
    reflection_threshold, reflection_curve = tune_reflection_threshold(
        tasks, cheap, reviews, oof_probabilities, costs, thresholds
    )
    write_csv(output_dir / "development_reflection_thresholds.csv", reflection_curve)
    reflection_use_strong = reflection_selections(
        tasks, cheap, oof_probabilities, reflection_threshold
    )
    reflection_selected = [
        review if use_review else cheap_response
        for cheap_response, review, use_review in zip(cheap, reviews, reflection_use_strong)
    ]
    reflection_responses = [
        [cheap_response, review] if use_review else [cheap_response]
        for cheap_response, review, use_review in zip(cheap, reviews, reflection_use_strong)
    ]
    reflection_summary = {
        "strategy": "reflection",
        **policy_summary(
            tasks,
            reflection_selected,
            costs,
            strong_usage=reflection_use_strong,
            calls=[
                ["cheap", "strong"] if value else ["cheap"]
                for value in reflection_use_strong
            ],
            all_responses=reflection_responses,
            threshold_name="confidence_threshold",
            threshold=reflection_threshold,
        ),
    }
    development_summary.append(reflection_summary)
    reflection_fold_rows = fold_policy_rows(
        tasks,
        reflection_selected,
        reflection_use_strong,
        costs,
        all_responses=reflection_responses,
    )
    for row in reflection_fold_rows:
        row["diagnostic_fold_threshold"] = diagnostic_fold_threshold(
            tasks,
            cheap,
            reviews,
            costs,
            int(row["fold_id"]),
            "reflection",
            scores=oof_probabilities,
            reflection_threshold_grid=thresholds,
        )
    write_csv(output_dir / "development_reflection_fold_metrics.csv", reflection_fold_rows)

    confidence_estimator = CalibratedConfidenceEstimator(
        include_task_features=False,
        logistic_regression_c=float(reflection_config["logistic_regression_c"]),
        calibration=str(reflection_config["calibration"]),
        inner_folds=int(reflection_config["inner_folds"]),
    ).fit(tasks, cheap)
    reflection_model_path = freeze_dir / "reflection_calibrator.joblib"
    joblib.dump(confidence_estimator, reflection_model_path)
    cheap_labels = [int(is_correct(task, response)) for task, response in zip(tasks, cheap)]
    reflection_diagnostics = {
        "threshold": reflection_threshold,
        "strong_usage_rate": reflection_summary["strong_usage_rate"],
        "oof_accuracy": reflection_summary["accuracy"],
        "model_file": reflection_model_path.name,
        "model_sha256": sha256_file(reflection_model_path),
        "calibrator": confidence_estimator.diagnostics,
        "oof_probabilities": numeric_diagnostics(oof_probabilities, cheap_labels),
        "mean_probability_when_correct": mean_for_label(oof_probabilities, cheap_labels, 1),
        "mean_probability_when_wrong": mean_for_label(oof_probabilities, cheap_labels, 0),
        "brier_score": round(
            sum((probability - label) ** 2 for probability, label in zip(oof_probabilities, cheap_labels))
            / len(tasks),
            8,
        ),
        "review_effects": dict(Counter(record["review_outcome"] for record in review_records)),
    }
    for task, label, probability, selected_strong in zip(
        tasks, cheap_labels, oof_probabilities, reflection_use_strong
    ):
        development_predictions.append(
            {
                "canonical_id": task.canonical_id or task.id,
                "fold_id": task.fold_id,
                "strategy": "reflection",
                "cheap_correct": label,
                "routing_score": probability,
                "threshold": reflection_threshold,
                "selected_role": "strong" if selected_strong else "cheap",
            }
        )

    write_csv(output_dir / "development_oof_predictions.csv", development_predictions)
    write_csv(output_dir / "development_oof_summary.csv", development_summary)
    validate_phase1_results(learned_results, reflection_diagnostics, config)
    freeze_payload = write_freeze(
        args=args,
        config=config,
        output_dir=output_dir,
        freeze_dir=freeze_dir,
        manifest_hashes=manifest_hashes,
        tasks=tasks,
        learned_results=learned_results,
        reflection_diagnostics=reflection_diagnostics,
        providers=providers,
        cheap=cheap,
        strong=strong,
        reviews=reviews,
        started_at=started_at,
    )
    write_json(marker, freeze_payload)
    print(f"Phase 1 frozen at {marker}")
    print(f"Learned Combined threshold: {learned_results['combined']['threshold']:.8f}")
    print(f"Reflection threshold: {reflection_threshold:.2f}")


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
        for future in tqdm(as_completed(futures), total=len(futures), desc="development reviews"):
            output[futures[future]] = future.result()
    if any(response is None for response in output):
        raise RuntimeError("A development review response is missing.")
    return [response for response in output if response is not None]


def validate_development(tasks: Sequence[TaskExample], config: dict[str, Any]) -> None:
    if len(tasks) != 3200:
        raise ValueError(f"Phase 1 requires exactly 3200 development tasks; got {len(tasks)}")
    folds = int(config["cross_validation_folds"])
    fold_counts = Counter(task.fold_id for task in tasks)
    if fold_counts != Counter({fold: 640 for fold in range(folds)}):
        raise ValueError(f"Unexpected frozen folds: {fold_counts}")
    if any(task.prompt_hash != sha256_json_prompt(task) for task in tasks):
        raise ValueError("A development prompt hash no longer matches the frozen prompt builder.")


def sha256_json_prompt(task: TaskExample) -> str:
    import hashlib

    return hashlib.sha256(build_prompt(task).encode("utf-8")).hexdigest()


def write_label_reports(records: Sequence[dict[str, Any]], output_dir: Path) -> None:
    rows = [
        *label_distribution(records),
        *label_distribution(records, "task_family"),
        *label_distribution(records, "dataset"),
    ]
    write_csv(output_dir / "development_label_distribution.csv", rows)
    write_json(
        output_dir / "development_label_summary.json",
        {
            "examples": len(records),
            "quality_gap_counts": dict(sorted(Counter(str(row["quality_gap"]) for row in records).items())),
            "pair_outcomes": dict(sorted(Counter(str(row["pair_outcome"]) for row in records).items())),
            "nonzero_quality_gap": sum(int(row["quality_gap"]) != 0 for row in records),
            "strong_beneficial": sum(int(row["quality_gap"]) == 1 for row in records),
            "cheap_beneficial": sum(int(row["quality_gap"]) == -1 for row in records),
        },
    )


def review_record(task: TaskExample, cheap: ModelResponse, review: ModelResponse) -> dict[str, Any]:
    cheap_correct = int(is_correct(task, cheap))
    review_correct = int(is_correct(task, review))
    outcome = (
        "beneficial"
        if not cheap_correct and review_correct
        else "harmful"
        if cheap_correct and not review_correct
        else "kept_correct"
        if cheap_correct
        else "still_wrong"
    )
    return {
        "canonical_id": task.canonical_id or task.id,
        "dataset": task.dataset,
        "task_family": task.metadata.get("category", task.task_type),
        "difficulty_group": task.difficulty_group,
        "cheap_correct": cheap_correct,
        "review_correct": review_correct,
        "review_outcome": outcome,
        "review_action": review.metadata.get("review_action"),
        "review_changed": bool(review.metadata.get("review_changed", False)),
        "cheap_answer": cheap.answer,
        "review_answer": review.answer,
        "review_raw_output": review.raw_text,
        "review_confidence": review.confidence,
        "review_metadata": review.metadata,
        "observed_latency_ms": review.latency_ms,
    }


def baseline_development_summaries(
    tasks: Sequence[TaskExample],
    cheap: Sequence[ModelResponse],
    strong: Sequence[ModelResponse],
    costs: dict[str, float],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    all_cheap = [False] * len(tasks)
    all_strong = [True] * len(tasks)
    task_aware = frozen_task_aware_selection(
        tasks, float(config["task_aware_router"]["risk_threshold"])
    )
    rows = []
    for strategy, usage in (
        ("always_cheap", all_cheap),
        ("always_strong", all_strong),
        ("task_aware", task_aware),
    ):
        selected = [
            strong_response if use_strong else cheap_response
            for cheap_response, strong_response, use_strong in zip(cheap, strong, usage)
        ]
        rows.append(
            {
                "strategy": strategy,
                **policy_summary(
                    tasks,
                    selected,
                    costs,
                    strong_usage=usage,
                    calls=[["strong"] if value else ["cheap"] for value in usage],
                ),
            }
        )
    return rows


def reflection_thresholds(config: dict[str, Any]) -> list[float]:
    start = float(config["threshold_grid_start"])
    stop = float(config["threshold_grid_stop"])
    step = float(config["threshold_grid_step"])
    count = int(round((stop - start) / step))
    return [round(start + index * step, 10) for index in range(count + 1)]


def reflection_selections(
    tasks: Sequence[TaskExample],
    cheap: Sequence[ModelResponse],
    probabilities: Sequence[float],
    threshold: float,
) -> list[bool]:
    from routerbench_mini.verifiers import verify_response

    return [
        verify_response(
            task,
            response,
            confidence_threshold=threshold,
            estimated_confidence=probability,
        ).should_escalate
        for task, response, probability in zip(tasks, cheap, probabilities)
    ]


def numeric_diagnostics(values: Sequence[float], targets: Sequence[float]) -> dict[str, Any]:
    return {
        "minimum": min(values),
        "maximum": max(values),
        "mean": statistics.fmean(values),
        "standard_deviation": statistics.pstdev(values),
        "correlation_with_target": correlation(values, targets),
        "unique_values": len(set(values)),
    }


def mean_for_label(values: Sequence[float], labels: Sequence[int], label: int) -> float:
    selected = [value for value, current in zip(values, labels) if current == label]
    return round(statistics.fmean(selected), 8) if selected else 0.0


def diagnostic_fold_threshold(
    tasks: Sequence[TaskExample],
    cheap: Sequence[ModelResponse],
    alternate: Sequence[ModelResponse],
    costs: dict[str, float],
    fold_id: int,
    mode: str,
    *,
    scores: Sequence[float] | None = None,
    reflection_threshold_grid: Sequence[float] | None = None,
) -> float | None:
    indices = [index for index, task in enumerate(tasks) if task.fold_id == fold_id]
    fold_tasks = [tasks[index] for index in indices]
    fold_cheap = [cheap[index] for index in indices]
    fold_alternate = [alternate[index] for index in indices]
    fold_scores = [scores[index] for index in indices] if scores is not None else []
    if mode == "learned":
        assert scores is not None
        try:
            return tune_learned_threshold(
                fold_tasks,
                fold_cheap,
                fold_alternate,
                [scores[index] for index in indices],
                costs,
            )[0]
        except ValueError:
            return None
    assert reflection_threshold_grid is not None
    try:
        return tune_reflection_threshold(
            fold_tasks,
            fold_cheap,
            fold_alternate,
            fold_scores,
            costs,
            reflection_threshold_grid,
        )[0]
    except ValueError:
        return None


def validate_phase1_results(
    learned_results: dict[str, dict[str, Any]],
    reflection: dict[str, Any],
    config: dict[str, Any],
) -> None:
    missing = set(LEARNED_VARIANTS) - learned_results.keys()
    if missing:
        raise ValueError(f"Missing Learned variants: {sorted(missing)}")
    rates = [float(value["strong_usage_rate"]) for value in learned_results.values()]
    rates.append(float(reflection["strong_usage_rate"]))
    minimum = float(config["learned_router"]["minimum_strong_rate"])
    maximum = float(config["learned_router"]["maximum_strong_rate"])
    if any(rate < minimum or rate > maximum for rate in rates):
        raise ValueError(
            f"A frozen router is outside the [{minimum}, {maximum}] Strong-rate range: {rates}"
        )


def write_freeze(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    output_dir: Path,
    freeze_dir: Path,
    manifest_hashes: dict[str, Any],
    tasks: Sequence[TaskExample],
    learned_results: dict[str, dict[str, Any]],
    reflection_diagnostics: dict[str, Any],
    providers: dict[str, Provider],
    cheap: Sequence[ModelResponse],
    strong: Sequence[ModelResponse],
    reviews: Sequence[ModelResponse],
    started_at: datetime,
) -> dict[str, Any]:
    prompt_spec = {
        "prompt_version": PROMPT_VERSION,
        "system_prompt": None,
        "solve_prompt_builder_source": inspect.getsource(build_prompt),
        "review_prompt_builder_source": inspect.getsource(build_review_prompt),
        "tool_choice": "auto",
        "maximum_reflection_rounds": int(config["reflection_router"]["max_reflection_rounds"]),
    }
    write_json(freeze_dir / "prompt_spec.json", prompt_spec)
    write_json(
        freeze_dir / "feature_and_scoring_spec.json",
        {
            "structured_feature_names": task_feature_names(),
            "structured_feature_count": len(task_feature_names()),
            "quality_gap": "int(strong_correct) - int(cheap_correct)",
            "scoring_module_sha256": sha256_file("src/routerbench_mini/scoring.py"),
            "feature_module_sha256": sha256_file("src/routerbench_mini/features.py"),
            "task_aware_risk_threshold": float(config["task_aware_router"]["risk_threshold"]),
            "task_aware_frozen_from_version": config["task_aware_router"]["frozen_from_version"],
        },
    )
    write_json(freeze_dir / "model_config.safe.json", safe_model_config(args.models))
    write_json(
        freeze_dir / "selected_parameters.json",
        {
            "learned": learned_results,
            "reflection": reflection_diagnostics,
            "threshold_rule": config["learned_router"]["threshold_rule"],
            "allowed_strong_rate": {
                "minimum": float(config["learned_router"]["minimum_strong_rate"]),
                "maximum": float(config["learned_router"]["maximum_strong_rate"]),
            },
            "cross_validation_folds": int(config["cross_validation_folds"]),
            "fold_ids_from_manifest": True,
        },
    )

    tracked_files = [
        args.config,
        args.models,
        args.costs,
        args.development,
        "src/routerbench_mini/providers.py",
        "src/routerbench_mini/scoring.py",
        "src/routerbench_mini/features.py",
        "src/routerbench_mini/selection.py",
        "src/routerbench_mini/calibration.py",
        "src/routerbench_mini/verifiers.py",
        "src/routerbench_mini/v5.py",
        "scripts/run_v5_phase1.py",
        "scripts/run_v5_phase2.py",
    ]
    inventory_files = [
        *tracked_files,
        *(str(path) for path in sorted(freeze_dir.glob("*.joblib"))),
        str(freeze_dir / "prompt_spec.json"),
        str(freeze_dir / "feature_and_scoring_spec.json"),
        str(freeze_dir / "model_config.safe.json"),
        str(freeze_dir / "selected_parameters.json"),
    ]
    missing = [path for path in inventory_files if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Freeze inventory cannot be completed; missing {missing}")
    inventory = {path: sha256_file(path) for path in inventory_files}
    write_json(freeze_dir / "freeze_inventory.json", inventory)
    completed_at = datetime.now(timezone.utc)
    return {
        "study_version": config["version"],
        "phase": "development_training_and_freeze_complete",
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "development_examples": len(tasks),
        "development_manifest_sha256": manifest_hashes["development_manifest"],
        "frozen_test_manifest_sha256": manifest_hashes["test_manifest"],
        "test_examples_used_in_phase1": 0,
        "test_predictions_generated_in_phase1": 0,
        "phase2_permitted": True,
        "freeze_inventory_sha256": sha256_file(freeze_dir / "freeze_inventory.json"),
        "protocol_config_sha256": sha256_file(args.config),
        "model_config_sha256": sha256_file(args.models),
        "cost_config_sha256": sha256_file(args.costs),
        "git": git_state(),
        "runtime": runtime_state(),
        "models": {
            role: {
                "model": provider.model,
                "role": role,
                "temperature": getattr(provider, "temperature", None),
                "top_p": getattr(provider, "top_p", None),
                "max_tokens": getattr(provider, "max_tokens", None),
                "timeout_s": getattr(provider, "timeout_s", None),
                "retries": getattr(provider, "retries", None),
                "base_url": getattr(provider, "base_url", None),
            }
            for role, provider in providers.items()
        },
        "development_api_totals": {
            "cheap_solve": response_api_totals(cheap),
            "strong_solve": response_api_totals(strong),
            "strong_review": response_api_totals(reviews),
        },
        "selected_thresholds": {
            **{f"learned_{mode}": result["threshold"] for mode, result in learned_results.items()},
            "reflection": reflection_diagnostics["threshold"],
        },
    }


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(args, check=True, text=True, capture_output=True).stdout.strip()

    diff = run("git", "diff", "--binary")
    return {
        "branch": run("git", "branch", "--show-current"),
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
        "working_tree_diff_sha256": sha256_json(diff),
    }


def runtime_state() -> dict[str, Any]:
    packages = {}
    for package in ("numpy", "scipy", "scikit-learn", "joblib", "requests", "pyyaml"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "missing"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


if __name__ == "__main__":
    main()
