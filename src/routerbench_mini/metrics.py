from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .routers import RoutingDecision
from .scoring import is_correct
from .tasks import TaskExample


def decision_cost(decision: RoutingDecision, costs: dict[str, float]) -> float:
    if decision.router == "oracle":
        return float(costs.get(decision.selected_role, 0.0))
    return sum(float(costs.get(call, 0.0)) for call in decision.calls)


def prediction_row(task: TaskExample, decision: RoutingDecision, costs: dict[str, float]) -> dict[str, Any]:
    correct = is_correct(task, decision.response)
    return {
        "id": task.id,
        "dataset": task.dataset,
        "task_type": task.task_type,
        "router": decision.router,
        "selected_role": decision.selected_role,
        "correct": int(correct),
        "cost": round(decision_cost(decision, costs), 4),
        "latency_ms": round(decision.latency_ms, 2),
        "escalated": int(decision.escalated),
        "confidence": decision.response.confidence,
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
        summary.append(
            {
                "router": router,
                "total": total,
                "accuracy": round(correct / total, 4) if total else 0.0,
                "avg_cost": round(total_cost / total, 4) if total else 0.0,
                "avg_latency_ms": round(total_latency / total, 2) if total else 0.0,
                "escalation_rate": round(escalations / total, 4) if total else 0.0,
            }
        )
    return summary


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
