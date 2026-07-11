from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .routers import RoutingDecision
from .scoring import is_correct
from .tasks import TaskExample


def decision_cost(decision: RoutingDecision, costs: dict[str, float]) -> float:
    if decision.responses and all("cost" in response.metadata for response in decision.responses):
        return sum(float(response.metadata["cost"]) for response in decision.responses)
    return sum(float(costs.get(call, 0.0)) for call in decision.calls)


def prediction_row(task: TaskExample, decision: RoutingDecision, costs: dict[str, float]) -> dict[str, Any]:
    correct = is_correct(task, decision.response)
    usage = [response.metadata.get("usage", {}) for response in decision.responses]
    return {
        "id": task.id,
        "dataset": task.dataset,
        "category": task.metadata.get("category", task.task_type),
        "task_type": task.task_type,
        "router": decision.router,
        "selected_role": decision.selected_role,
        "selected_model": decision.response.model,
        "correct": int(correct),
        "cost": round(decision_cost(decision, costs), 8),
        "latency_ms": round(decision.latency_ms, 2),
        "escalated": int(decision.escalated),
        "strong_used": int("strong" in decision.calls),
        "confidence": decision.response.confidence,
        "routing_correctness_probability": decision.trace.get(
            "estimated_correctness_probability",
            decision.response.confidence,
        ),
        "task_risk_score": decision.trace.get("risk_score", ""),
        "review_action": decision.trace.get("review_action", ""),
        "review_changed": int(bool(decision.trace.get("review_changed", False))),
        "prompt_tokens": sum(int(item.get("prompt_tokens", 0)) for item in usage),
        "completion_tokens": sum(int(item.get("completion_tokens", 0)) for item in usage),
        "answer": decision.response.answer,
        "verification_reason": decision.verification.reason if decision.verification else "",
        "calls": "|".join(decision.calls),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_router: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_router.setdefault(str(row["router"]), []).append(row)

    summary: list[dict[str, Any]] = []
    for router, router_rows in sorted(by_router.items()):
        total = len(router_rows)
        correct = sum(int(row["correct"]) for row in router_rows)
        total_cost = sum(float(row["cost"]) for row in router_rows)
        total_latency = sum(float(row["latency_ms"]) for row in router_rows)
        escalations = sum(int(row["escalated"]) for row in router_rows)
        strong_uses = sum(int(row.get("strong_used", 0)) for row in router_rows)
        summary.append(
            {
                "router": router,
                "total": total,
                "accuracy": round(correct / total, 4) if total else 0.0,
                "total_cost": round(total_cost, 6),
                "avg_cost": round(total_cost / total, 8) if total else 0.0,
                "avg_latency_ms": round(total_latency / total, 2) if total else 0.0,
                "escalation_rate": round(escalations / total, 4) if total else 0.0,
                "strong_usage_rate": round(strong_uses / total, 4) if total else 0.0,
            }
        )
    return summary


def summarize_rows_by_category(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    categories = sorted({str(row["category"]) for row in rows})
    for category in categories:
        category_rows = [row for row in rows if str(row["category"]) == category]
        for summary in summarize_rows(category_rows):
            output.append({"category": category, **summary})
    return output


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
