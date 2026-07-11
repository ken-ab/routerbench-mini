from __future__ import annotations

from routerbench_mini.metrics import decision_cost
from routerbench_mini.providers import ModelResponse
from routerbench_mini.routers import RoutingDecision
from routerbench_mini.scoring import is_correct
from routerbench_mini.tasks import TaskExample


def test_tool_scoring_accepts_required_arguments_with_extra_fields() -> None:
    task = TaskExample(
        id="tool-1",
        dataset="unit",
        task_type="tool",
        question="Search papers about LLM routing.",
        answer={"name": "search_papers", "arguments": {"query": "LLM routing", "year_after": 2023}},
        tools=[
            {
                "name": "search_papers",
                "required": ["query", "year_after"],
                "properties": {"query": "string", "year_after": "integer"},
            }
        ],
    )
    response = '{"name": "search_papers", "arguments": {"query": "LLM routing", "year_after": 2023, "limit": 5}}'

    assert is_correct(task, response)


def test_decision_cost_uses_measured_api_costs() -> None:
    response = ModelResponse(
        role="strong",
        model="strong",
        answer="42",
        raw_text="42",
        confidence=1.0,
        latency_ms=1200.0,
        metadata={"cost": 0.012},
    )
    decision = RoutingDecision(
        router="always_strong",
        selected_role="strong",
        response=response,
        calls=["strong"],
        responses=[response],
    )

    assert decision_cost(decision, {"cheap": 1.0, "strong": 3.0}) == 0.012


def test_open_vqa_scoring_accepts_numeric_tolerance() -> None:
    task = TaskExample(
        id="chart-1",
        dataset="chartqa",
        task_type="vqa",
        question="What value is shown?",
        answer="100",
        image_path="chart.png",
    )

    assert is_correct(task, "104")
    assert not is_correct(task, "110")
