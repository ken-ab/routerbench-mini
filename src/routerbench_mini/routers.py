from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .providers import ModelResponse, Provider
from .scoring import is_correct
from .tasks import TaskExample
from .verifiers import VerificationResult, verify_response


@dataclass
class RoutingDecision:
    router: str
    selected_role: str
    response: ModelResponse
    calls: list[str]
    responses: list[ModelResponse] = field(default_factory=list)
    escalated: bool = False
    verification: VerificationResult | None = None
    trace: dict[str, Any] = field(default_factory=dict)

    @property
    def latency_ms(self) -> float:
        if self.responses:
            return sum(response.latency_ms for response in self.responses)
        return self.response.latency_ms


class BaseRouter:
    name = "base"

    def route(self, task: TaskExample, providers: dict[str, Provider]) -> RoutingDecision:
        raise NotImplementedError


class AlwaysCheapRouter(BaseRouter):
    name = "always_cheap"

    def route(self, task: TaskExample, providers: dict[str, Provider]) -> RoutingDecision:
        response = providers["cheap"].generate(task)
        return RoutingDecision(
            router=self.name,
            selected_role="cheap",
            response=response,
            calls=["cheap"],
            responses=[response],
        )


class AlwaysStrongRouter(BaseRouter):
    name = "always_strong"

    def route(self, task: TaskExample, providers: dict[str, Provider]) -> RoutingDecision:
        response = providers["strong"].generate(task)
        return RoutingDecision(
            router=self.name,
            selected_role="strong",
            response=response,
            calls=["strong"],
            responses=[response],
        )


class TaskAwareRouter(BaseRouter):
    name = "task_aware"

    def route(self, task: TaskExample, providers: dict[str, Provider]) -> RoutingDecision:
        role = "strong" if task.metadata.get("rule_tier") == "strong" else "cheap"
        response = providers[role].generate(task)
        return RoutingDecision(
            router=self.name,
            selected_role=role,
            response=response,
            calls=[role],
            responses=[response],
            trace={"rule_tier": task.metadata.get("rule_tier", "cheap")},
        )


class ReflectionRouter(BaseRouter):
    name = "reflection"

    def __init__(self, confidence_threshold: float = 0.55) -> None:
        self.confidence_threshold = confidence_threshold

    def route(self, task: TaskExample, providers: dict[str, Provider]) -> RoutingDecision:
        cheap_response = providers["cheap"].generate(task)
        verification = verify_response(
            task,
            cheap_response,
            confidence_threshold=self.confidence_threshold,
        )
        if verification.should_escalate:
            strong_response = providers["strong"].generate(task)
            return RoutingDecision(
                router=self.name,
                selected_role="strong",
                response=strong_response,
                calls=["cheap", "verifier", "strong"],
                responses=[cheap_response, strong_response],
                escalated=True,
                verification=verification,
                trace={"cheap_answer": cheap_response.answer},
            )
        return RoutingDecision(
            router=self.name,
            selected_role="cheap",
            response=cheap_response,
            calls=["cheap", "verifier"],
            responses=[cheap_response],
            escalated=False,
            verification=verification,
        )


class OracleRouter(BaseRouter):
    """Post-hoc diagnostic upper bound; excluded from the four headline methods."""

    name = "oracle"

    def __init__(self, costs: dict[str, float]) -> None:
        self.costs = costs

    def route(self, task: TaskExample, providers: dict[str, Provider]) -> RoutingDecision:
        responses: list[ModelResponse] = []
        for role in sorted(("cheap", "strong"), key=lambda value: self.costs.get(value, 999.0)):
            response = providers[role].generate(task)
            responses.append(response)
            if is_correct(task, response):
                return RoutingDecision(
                    router=self.name,
                    selected_role=role,
                    response=response,
                    calls=[item.role for item in responses],
                    responses=responses,
                    escalated=len(responses) > 1,
                    trace={"oracle_found_correct": True},
                )
        return RoutingDecision(
            router=self.name,
            selected_role="strong",
            response=responses[-1],
            calls=[item.role for item in responses],
            responses=responses,
            escalated=True,
            trace={"oracle_found_correct": False},
        )


def default_routers(
    costs: dict[str, float] | None = None,
    confidence_threshold: float = 0.55,
) -> list[BaseRouter]:
    return [
        AlwaysCheapRouter(),
        AlwaysStrongRouter(),
        TaskAwareRouter(),
        ReflectionRouter(confidence_threshold=confidence_threshold),
    ]
