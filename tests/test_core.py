from __future__ import annotations

from pathlib import Path

from routerbench_mini.config import load_costs, load_yaml
from routerbench_mini.metrics import prediction_row, summarize_rows
from routerbench_mini.providers import provider_from_config
from routerbench_mini.routers import AlwaysCheapRouter, ReflectionRouter
from routerbench_mini.tasks import load_jsonl


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_loads() -> None:
    tasks = load_jsonl(ROOT / "data" / "mini_manifest.jsonl")
    assert len(tasks) == 12
    assert {task.task_type for task in tasks} == {"math", "vqa", "tool"}


def test_mock_router_produces_rows() -> None:
    tasks = load_jsonl(ROOT / "data" / "mini_manifest.jsonl")[:3]
    costs = load_costs(ROOT / "configs" / "costs.yaml")
    model_config = load_yaml(ROOT / "configs" / "models.mock.yaml")["providers"]
    providers = {role: provider_from_config(role, config) for role, config in model_config.items()}

    rows = []
    for router in [AlwaysCheapRouter(), ReflectionRouter()]:
        for task in tasks:
            decision = router.route(task, providers)
            rows.append(prediction_row(task, decision, costs))

    summary = summarize_rows(rows)
    assert {row["router"] for row in summary} == {"always_cheap", "reflection"}
    assert all("accuracy" in row for row in summary)
