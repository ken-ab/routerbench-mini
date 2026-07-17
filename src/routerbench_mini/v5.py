from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import load_yaml
from .features import task_risk_score
from .providers import ModelResponse
from .scoring import is_correct
from .tasks import TaskExample
from .verifiers import verify_response


LEARNED_VARIANTS = {
    "text_only": {"include_text": True, "include_structured": False},
    "structured_only": {"include_text": False, "include_structured": True},
    "combined": {"include_text": True, "include_structured": True},
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_file_hash(path: str | Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: expected {expected}, got {actual}")


def learned_estimator_kwargs(config: dict[str, Any], mode: str) -> dict[str, Any]:
    learned = config["learned_router"]
    tfidf = learned["tfidf"]
    return {
        "alpha": float(learned["alpha"]),
        "max_text_features": int(tfidf["max_features"]),
        "ngram_range": tuple(int(value) for value in tfidf["ngram_range"]),
        "min_df": int(tfidf["min_df"]),
        "max_df": float(tfidf["max_df"]),
        "sublinear_tf": bool(tfidf["sublinear_tf"]),
        "strip_accents": str(tfidf["strip_accents"]) if tfidf.get("strip_accents") else None,
        "norm": str(tfidf["norm"]),
        **LEARNED_VARIANTS[mode],
    }


def response_cost(response: ModelResponse, role: str, costs: dict[str, float]) -> float:
    if "cost" in response.metadata:
        return float(response.metadata["cost"])
    return float(costs.get(role, 0.0))


def response_tokens(response: ModelResponse) -> tuple[int, int]:
    usage = response.metadata.get("usage") or {}
    return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))


def threshold_candidates(scores: Sequence[float]) -> list[float]:
    values = sorted(set(float(value) for value in scores))
    if not values:
        raise ValueError("Threshold selection needs at least one score.")
    return [
        values[0] - 1.0,
        *((left + right) / 2 for left, right in zip(values, values[1:])),
        values[-1] + 1.0,
    ]


def tune_learned_threshold(
    tasks: Sequence[TaskExample],
    cheap: Sequence[ModelResponse],
    strong: Sequence[ModelResponse],
    scores: Sequence[float],
    costs: dict[str, float],
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for threshold in threshold_candidates(scores):
        use_strong = [score >= threshold for score in scores]
        selected = [
            strong_response if selected_strong else cheap_response
            for cheap_response, strong_response, selected_strong in zip(cheap, strong, use_strong)
        ]
        rows.append(
            policy_summary(
                tasks,
                selected,
                costs,
                strong_usage=use_strong,
                calls=[["strong"] if value else ["cheap"] for value in use_strong],
                threshold_name="advantage_threshold",
                threshold=threshold,
            )
        )
    return select_non_degenerate_threshold(rows, "advantage_threshold"), rows


def tune_reflection_threshold(
    tasks: Sequence[TaskExample],
    cheap: Sequence[ModelResponse],
    reviews: Sequence[ModelResponse],
    probabilities: Sequence[float],
    costs: dict[str, float],
    thresholds: Sequence[float],
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        selected: list[ModelResponse] = []
        use_strong: list[bool] = []
        calls: list[list[str]] = []
        for task, cheap_response, review, probability in zip(tasks, cheap, reviews, probabilities):
            verification = verify_response(
                task,
                cheap_response,
                confidence_threshold=float(threshold),
                estimated_confidence=float(probability),
            )
            use_review = verification.should_escalate
            selected.append(review if use_review else cheap_response)
            use_strong.append(use_review)
            calls.append(["cheap", "strong"] if use_review else ["cheap"])
        rows.append(
            policy_summary(
                tasks,
                selected,
                costs,
                strong_usage=use_strong,
                calls=calls,
                all_responses=[
                    [cheap_response, review] if use_review else [cheap_response]
                    for cheap_response, review, use_review in zip(cheap, reviews, use_strong)
                ],
                threshold_name="confidence_threshold",
                threshold=float(threshold),
            )
        )
    return select_non_degenerate_threshold(rows, "confidence_threshold"), rows


def select_non_degenerate_threshold(
    rows: Sequence[dict[str, Any]],
    threshold_name: str,
    *,
    minimum_strong_rate: float = 0.05,
    maximum_strong_rate: float = 0.95,
) -> float:
    candidates = [
        row
        for row in rows
        if minimum_strong_rate
        <= float(row["strong_usage_rate"])
        <= maximum_strong_rate
    ]
    if not candidates:
        raise ValueError(
            f"No threshold for {threshold_name} has Strong usage in "
            f"[{minimum_strong_rate}, {maximum_strong_rate}]."
        )
    best_accuracy = max(float(row["accuracy"]) for row in candidates)
    best = [row for row in candidates if float(row["accuracy"]) == best_accuracy]
    selected = min(
        best,
        key=lambda row: (float(row["avg_cost"]), float(row["strong_usage_rate"])),
    )
    return float(selected[threshold_name])


def policy_summary(
    tasks: Sequence[TaskExample],
    selected: Sequence[ModelResponse],
    costs: dict[str, float],
    *,
    strong_usage: Sequence[bool],
    calls: Sequence[Sequence[str]],
    all_responses: Sequence[Sequence[ModelResponse]] | None = None,
    threshold_name: str | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    responses = all_responses or [[response] for response in selected]
    total_cost = sum(
        sum(response_cost(response, response.role, costs) for response in task_responses)
        for task_responses in responses
    )
    total_latency = sum(
        sum(response.latency_ms for response in task_responses)
        for task_responses in responses
    )
    prompt_tokens = sum(response_tokens(response)[0] for values in responses for response in values)
    completion_tokens = sum(response_tokens(response)[1] for values in responses for response in values)
    correct = sum(is_correct(task, response) for task, response in zip(tasks, selected))
    row: dict[str, Any] = {
        "total": len(tasks),
        "accuracy": round(correct / len(tasks), 8),
        "total_cost": round(total_cost, 8),
        "avg_cost": round(total_cost / len(tasks), 10),
        "total_latency_ms": round(total_latency, 2),
        "avg_latency_ms": round(total_latency / len(tasks), 2),
        "strong_usage_rate": round(sum(strong_usage) / len(tasks), 8),
        "total_strong_calls": sum(sum(call == "strong" for call in task_calls) for task_calls in calls),
        "total_cheap_calls": sum(sum(call == "cheap" for call in task_calls) for task_calls in calls),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    if threshold_name is not None:
        row[threshold_name] = threshold
    return row


def task_pair_record(
    task: TaskExample, cheap: ModelResponse, strong: ModelResponse
) -> dict[str, Any]:
    cheap_correct = int(is_correct(task, cheap))
    strong_correct = int(is_correct(task, strong))
    cheap_verification = verify_response(task, cheap, confidence_threshold=0.0)
    strong_verification = verify_response(task, strong, confidence_threshold=0.0)
    return {
        "canonical_id": task.canonical_id or task.id,
        "dataset": task.dataset,
        "task_family": task.metadata.get("category", task.task_type),
        "task_type": task.task_type,
        "task_subtype": task.task_subtype,
        "difficulty_group": task.difficulty_group,
        "fold_id": task.fold_id,
        "cheap_correct": cheap_correct,
        "strong_correct": strong_correct,
        "quality_gap": strong_correct - cheap_correct,
        "pair_outcome": pair_outcome(cheap_correct, strong_correct),
        "cheap": audited_response(cheap, cheap_correct, cheap_verification.valid_format),
        "strong": audited_response(strong, strong_correct, strong_verification.valid_format),
        "grader_result": {
            "method": "task_specific_binary_scorer",
            "answer_reference": task.answer_reference if task.answer_reference is not None else task.answer,
        },
    }


def audited_response(
    response: ModelResponse, correct: int, valid_format: bool
) -> dict[str, Any]:
    prompt_tokens, completion_tokens = response_tokens(response)
    return {
        **asdict(response),
        "score": correct,
        "correct": correct,
        "parsing_result": "valid" if valid_format else "invalid",
        "error_type": "" if correct else "wrong_answer" if valid_format else "invalid_format",
        "token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
        "estimated_cost": float(response.metadata.get("cost", 0.0)),
        "observed_latency_ms": response.latency_ms,
        "tool_trace": response.metadata.get("tool_trace", []),
    }


def pair_outcome(cheap_correct: int, strong_correct: int) -> str:
    return {
        (1, 1): "cheap_correct_strong_correct",
        (0, 1): "cheap_wrong_strong_correct",
        (1, 0): "cheap_correct_strong_wrong",
        (0, 0): "cheap_wrong_strong_wrong",
    }[(cheap_correct, strong_correct)]


def label_distribution(
    records: Sequence[dict[str, Any]], field: str | None = None
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if field is None:
        grouped["all"] = list(records)
    else:
        for record in records:
            grouped[str(record[field])].append(record)
    output: list[dict[str, Any]] = []
    order = (
        "cheap_correct_strong_correct",
        "cheap_wrong_strong_correct",
        "cheap_correct_strong_wrong",
        "cheap_wrong_strong_wrong",
    )
    for group, values in sorted(grouped.items()):
        counts = Counter(str(record["pair_outcome"]) for record in values)
        for outcome in order:
            output.append(
                {
                    "group_by": field or "overall",
                    "group": group,
                    "pair_outcome": outcome,
                    "count": counts[outcome],
                    "share": round(counts[outcome] / len(values), 8),
                }
            )
    return output


def fold_policy_rows(
    tasks: Sequence[TaskExample],
    selected: Sequence[ModelResponse],
    strong_usage: Sequence[bool],
    costs: dict[str, float],
    *,
    all_responses: Sequence[Sequence[ModelResponse]] | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for fold in sorted({int(task.fold_id) for task in tasks if task.fold_id is not None}):
        indices = [index for index, task in enumerate(tasks) if task.fold_id == fold]
        fold_responses = (
            [all_responses[index] for index in indices] if all_responses is not None else None
        )
        summary = policy_summary(
            [tasks[index] for index in indices],
            [selected[index] for index in indices],
            costs,
            strong_usage=[strong_usage[index] for index in indices],
            calls=(
                [[response.role for response in values] for values in fold_responses]
                if fold_responses is not None
                else [["strong"] if strong_usage[index] else ["cheap"] for index in indices]
            ),
            all_responses=fold_responses,
        )
        output.append({"fold_id": fold, **summary})
    return output


def safe_model_config(path: str | Path) -> dict[str, Any]:
    config = load_yaml(path)
    return {
        **config,
        "providers": {
            role: {key: value for key, value in values.items() if key != "api_key"}
            for role, values in config.get("providers", {}).items()
        },
    }


def response_api_totals(responses: Iterable[ModelResponse]) -> dict[str, Any]:
    values = list(responses)
    prompt_tokens = sum(response_tokens(response)[0] for response in values)
    completion_tokens = sum(response_tokens(response)[1] for response in values)
    return {
        "logical_calls": len(values),
        "request_attempts": sum(int(response.metadata.get("request_attempts", 1)) for response in values),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "estimated_cost": round(sum(float(response.metadata.get("cost", 0.0)) for response in values), 8),
        "observed_latency_ms": round(sum(response.latency_ms for response in values), 2),
    }


def correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else 0.0


def quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Quantile needs at least one value.")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_accuracy_interval(
    outcomes: Sequence[int], *, samples: int = 4000, seed: int = 20260720
) -> tuple[float, float]:
    rng = random.Random(seed)
    draws = [sum(rng.choice(outcomes) for _ in outcomes) / len(outcomes) for _ in range(samples)]
    return quantile(draws, 0.025), quantile(draws, 0.975)


def frozen_task_aware_selection(tasks: Sequence[TaskExample], threshold: float) -> list[bool]:
    return [task_risk_score(task) >= threshold for task in tasks]
