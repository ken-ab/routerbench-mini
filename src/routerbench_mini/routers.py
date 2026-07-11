from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .calibration import ConfidenceEstimator, RawConfidenceEstimator
from .features import observable_task_features, task_risk_score
from .providers import ModelResponse, Provider
from .scoring import is_correct
from .selection import LearnedQualityGapEstimator
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

    def __init__(self, risk_threshold: float = 2.0) -> None:
        self.risk_threshold = risk_threshold

    def route(self, task: TaskExample, providers: dict[str, Provider]) -> RoutingDecision:
        features = observable_task_features(task)
        risk_score = task_risk_score(task)
        role = "strong" if risk_score >= self.risk_threshold else "cheap"
        response = providers[role].generate(task)
        return RoutingDecision(
            router=self.name,
            selected_role=role,
            response=response,
            calls=[role],
            responses=[response],
            trace={
                "observable_features": features,
                "risk_score": risk_score,
                "risk_threshold": self.risk_threshold,
            },
        )


class LearnedCostAwareRouter(BaseRouter):
    name = "learned_cost_aware"

    def __init__(
        self,
        estimator: LearnedQualityGapEstimator,
        advantage_threshold: float,
        *,
        name: str | None = None,
    ) -> None:
        self.estimator = estimator
        self.advantage_threshold = advantage_threshold
        if name is not None:
            self.name = name

    def route(self, task: TaskExample, providers: dict[str, Provider]) -> RoutingDecision:
        advantage = self.estimator.predict_advantage(task)
        role = "strong" if advantage >= self.advantage_threshold else "cheap"
        response = providers[role].generate(task)
        return RoutingDecision(
            router=self.name,
            selected_role=role,
            response=response,
            calls=[role],
            responses=[response],
            trace={
                "predicted_strong_advantage": advantage,
                "advantage_threshold": self.advantage_threshold,
            },
        )


class ReflectionRouter(BaseRouter):
    name = "reflection"

    def __init__(
        self,
        confidence_threshold: float = 0.55,
        *,
        confidence_estimator: ConfidenceEstimator | None = None,
        check_format: bool = True,
        check_confidence: bool = True,
        check_self_check: bool = True,
        name: str | None = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.confidence_estimator = confidence_estimator or RawConfidenceEstimator()
        self.check_format = check_format
        self.check_confidence = check_confidence
        self.check_self_check = check_self_check
        if name is not None:
            self.name = name

    def route(self, task: TaskExample, providers: dict[str, Provider]) -> RoutingDecision:
        cheap_response = providers["cheap"].generate(task)
        estimated_confidence = self.confidence_estimator.predict_correctness(task, cheap_response)
        verification = verify_response(
            task,
            cheap_response,
            confidence_threshold=self.confidence_threshold,
            estimated_confidence=estimated_confidence,
            check_format=self.check_format,
            check_confidence=self.check_confidence,
            check_self_check=self.check_self_check,
        )
        if verification.should_escalate:
            strong_response = providers["strong"].review_and_correct(task, cheap_response)
            return RoutingDecision(
                router=self.name,
                selected_role="strong",
                response=strong_response,
                calls=["cheap", "verifier", "strong"],
                responses=[cheap_response, strong_response],
                escalated=True,
                verification=verification,
                trace={
                    "cheap_answer": cheap_response.answer,
                    "raw_confidence": cheap_response.confidence,
                    "estimated_correctness_probability": estimated_confidence,
                    "review_action": strong_response.metadata.get("review_action", "unknown"),
                    "review_changed": bool(strong_response.metadata.get("review_changed", False)),
                },
            )
        return RoutingDecision(
            router=self.name,
            selected_role="cheap",
            response=cheap_response,
            calls=["cheap", "verifier"],
            responses=[cheap_response],
            escalated=False,
            verification=verification,
            trace={
                "raw_confidence": cheap_response.confidence,
                "estimated_correctness_probability": estimated_confidence,
            },
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
    task_risk_threshold: float = 2.0,
) -> list[BaseRouter]:
    return [
        AlwaysCheapRouter(),
        AlwaysStrongRouter(),
        TaskAwareRouter(risk_threshold=task_risk_threshold),
        ReflectionRouter(confidence_threshold=confidence_threshold),
    ]
