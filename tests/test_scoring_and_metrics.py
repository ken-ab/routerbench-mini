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


def test_oracle_decision_cost_reports_selected_model_cost_only() -> None:
    response = ModelResponse(
        role="strong_text",
        model="strong",
        answer="42",
        raw_text="42",
        confidence=1.0,
        latency_ms=1200.0,
    )
    decision = RoutingDecision(
        router="oracle",
        selected_role="strong_text",
        response=response,
        calls=["cheap_text", "strong_text"],
        escalated=True,
    )

    assert decision_cost(decision, {"cheap_text": 1.0, "strong_text": 8.0}) == 8.0
