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
    escalated: bool = False
    verification: VerificationResult | None = None
    trace: dict[str, Any] = field(default_factory=dict)

    @property
    def latency_ms(self) -> float:
        return float(self.trace.get("latency_ms", self.response.latency_ms))


class BaseRouter:
    name = "base"

    def route(self, task: TaskExample, providers: dict[str, Provider]) -> RoutingDecision:
        raise NotImplementedError


class AlwaysCheapRouter(BaseRouter):
    name = "always_cheap"

    def route(self, task: TaskExample, providers: dict[str, Provider]) -> RoutingDecision:
        role = "cheap_vlm" if task.requires_vision else "cheap_text"
        response = providers[role].generate(task)
        return RoutingDecision(router=self.name, selected_role=role, response=response, calls=[role])


class AlwaysStrongRouter(BaseRouter):
    name = "always_strong"

    def route(self, task: TaskExample, providers: dict[str, Provider]) -> RoutingDecision:
        role = "strong_vlm" if task.requires_vision else "strong_text"
        response = providers[role].generate(task)
        return RoutingDecision(router=self.name, selected_role=role, response=response, calls=[role])


class RuleBasedRouter(BaseRouter):
    name = "rule_based"

    def route(self, task: TaskExample, providers: dict[str, Provider]) -> RoutingDecision:
        if task.requires_vision:
            role = "cheap_vlm"
        elif task.task_type in {"math", "tool"}:
            role = "strong_text"
        else:
            role = "cheap_text"
        response = providers[role].generate(task)
        return RoutingDecision(router=self.name, selected_role=role, response=response, calls=[role])


class SelectiveEscalationRouter(BaseRouter):
    name = "selective_escalation"

    def __init__(self, confidence_threshold: float = 0.55) -> None:
        self.confidence_threshold = confidence_threshold

    def route(self, task: TaskExample, providers: dict[str, Provider]) -> RoutingDecision:
        cheap_role = "cheap_vlm" if task.requires_vision else "cheap_text"
        strong_role = "strong_vlm" if task.requires_vision else "strong_text"

        cheap_response = providers[cheap_role].generate(task)
        verification = verify_response(task, cheap_response, confidence_threshold=self.confidence_threshold)
        calls = [cheap_role, "verifier"]
        latency_ms = cheap_response.latency_ms

        if verification.should_escalate:
            strong_response = providers[strong_role].generate(task)
            calls.append(strong_role)
            latency_ms += strong_response.latency_ms
            return RoutingDecision(
                router=self.name,
                selected_role=strong_role,
                response=strong_response,
                calls=calls,
                escalated=True,
                verification=verification,
                trace={"cheap_answer": cheap_response.answer, "latency_ms": latency_ms},
            )

        return RoutingDecision(
            router=self.name,
            selected_role=cheap_role,
            response=cheap_response,
            calls=calls,
            escalated=False,
            verification=verification,
            trace={"latency_ms": latency_ms},
        )


class OracleRouter(BaseRouter):
    name = "oracle"

    def __init__(self, costs: dict[str, float]) -> None:
        self.costs = costs

    def route(self, task: TaskExample, providers: dict[str, Provider]) -> RoutingDecision:
        candidate_roles = ["cheap_vlm", "strong_vlm"] if task.requires_vision else ["cheap_text", "strong_text"]
        candidate_roles = sorted(candidate_roles, key=lambda role: self.costs.get(role, 999.0))

        calls: list[str] = []
        first_response: ModelResponse | None = None
        latency_ms = 0.0
        for role in candidate_roles:
            response = providers[role].generate(task)
            calls.append(role)
            latency_ms += response.latency_ms
            if first_response is None:
                first_response = response
            if is_correct(task, response):
                return RoutingDecision(
                    router=self.name,
                    selected_role=role,
                    response=response,
                    calls=calls,
                    escalated=len(calls) > 1,
                    trace={
                        "latency_ms": response.latency_ms,
                        "oracle_candidate_latency_ms": latency_ms,
                        "oracle_found_correct": True,
                    },
                )

        assert first_response is not None
        return RoutingDecision(
            router=self.name,
            selected_role=candidate_roles[-1],
            response=response,
            calls=calls,
            escalated=len(calls) > 1,
            trace={
                "latency_ms": response.latency_ms,
                "oracle_candidate_latency_ms": latency_ms,
                "oracle_found_correct": False,
            },
        )


def default_routers(costs: dict[str, float]) -> list[BaseRouter]:
    return [
        AlwaysCheapRouter(),
        AlwaysStrongRouter(),
        RuleBasedRouter(),
        SelectiveEscalationRouter(),
        OracleRouter(costs=costs),
    ]
