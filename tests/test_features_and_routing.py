from __future__ import annotations

from dataclasses import dataclass

from routerbench_mini.calibration import (
    CalibratedConfidenceEstimator,
    ConfidenceEstimator,
    cross_validated_correctness_probabilities,
)
from routerbench_mini.features import observable_task_features, task_risk_score
from routerbench_mini.providers import ModelResponse
from routerbench_mini.routers import LearnedCostAwareRouter, ReflectionRouter, TaskAwareRouter
from routerbench_mini.selection import LearnedQualityGapEstimator, cross_validated_advantages
from routerbench_mini.tasks import TaskExample


class StubProvider:
    def __init__(self, role: str) -> None:
        self.role = role
        self.review_calls = 0

    def generate(self, task: TaskExample) -> ModelResponse:
        return ModelResponse(
            role=self.role,
            model=f"{self.role}-stub",
            answer="4",
            raw_text="4",
            confidence=0.99,
            latency_ms=1.0,
            metadata={"self_check_pass": True},
        )

    def review_and_correct(self, task: TaskExample, candidate: ModelResponse) -> ModelResponse:
        del task
        self.review_calls += 1
        return ModelResponse(
            role=self.role,
            model=f"{self.role}-stub",
            answer=candidate.answer,
            raw_text=candidate.raw_text,
            confidence=0.99,
            latency_ms=1.0,
            metadata={
                "self_check_pass": True,
                "review_action": "keep",
                "review_changed": False,
            },
        )


class CorrectingProvider(StubProvider):
    def review_and_correct(self, task: TaskExample, candidate: ModelResponse) -> ModelResponse:
        del task, candidate
        self.review_calls += 1
        return ModelResponse(
            role=self.role,
            model=f"{self.role}-stub",
            answer="4",
            raw_text="4",
            confidence=0.99,
            latency_ms=1.0,
            metadata={
                "self_check_pass": True,
                "review_action": "correct",
                "review_changed": True,
            },
        )


class WrongCheapProvider(StubProvider):
    def generate(self, task: TaskExample) -> ModelResponse:
        response = super().generate(task)
        response.answer = "5"
        response.raw_text = "5"
        return response


@dataclass(frozen=True)
class FixedEstimator(ConfidenceEstimator):
    probability: float

    def predict_correctness(self, task: TaskExample, response: ModelResponse) -> float:
        del task, response
        return self.probability


def test_observable_features_do_not_include_dataset_metadata() -> None:
    task = TaskExample(
        id="math-1",
        dataset="secret-dataset-name",
        task_type="math",
        question="If 3 items each cost 4 dollars, what is the total?",
        answer="12",
        metadata={"rule_tier": "cheap", "source": "hidden-source"},
    )

    features = observable_task_features(task)

    assert "dataset" not in features
    assert "source" not in features
    assert "rule_tier" not in features
    assert features["is_math"] == 1.0


def test_task_aware_router_uses_observable_risk_not_dataset_name() -> None:
    providers = {"cheap": StubProvider("cheap"), "strong": StubProvider("strong")}
    first = TaskExample(
        id="math-a",
        dataset="easy-name",
        task_type="math",
        question="What is 9 times 7?",
        answer="63",
        metadata={"rule_tier": "cheap"},
    )
    second = TaskExample(
        id="math-b",
        dataset="hard-name",
        task_type="text",
        question="What is 9 times 7?",
        answer="63",
        metadata={"rule_tier": "strong"},
    )

    router = TaskAwareRouter(risk_threshold=2.0)

    assert task_risk_score(first) == task_risk_score(second)
    assert router.route(first, providers).selected_role == "strong"
    assert router.route(second, providers).selected_role == "strong"


def test_reflection_uses_estimated_correctness_probability() -> None:
    providers = {"cheap": StubProvider("cheap"), "strong": StubProvider("strong")}
    task = TaskExample(id="math-1", dataset="unit", task_type="math", question="2 + 2?", answer="4")
    router = ReflectionRouter(
        confidence_threshold=0.6,
        confidence_estimator=FixedEstimator(0.2),
    )

    decision = router.route(task, providers)

    assert decision.escalated
    assert decision.selected_role == "strong"
    assert decision.trace["raw_confidence"] == 0.99
    assert decision.trace["estimated_correctness_probability"] == 0.2
    assert decision.trace["review_action"] == "keep"


def test_reflection_review_and_correct_fixes_candidate() -> None:
    strong = CorrectingProvider("strong")
    providers = {"cheap": WrongCheapProvider("cheap"), "strong": strong}
    task = TaskExample(id="math-2", dataset="unit", task_type="math", question="2 + 2?", answer="4")
    router = ReflectionRouter(
        confidence_threshold=0.6,
        confidence_estimator=FixedEstimator(0.2),
    )

    decision = router.route(task, providers)

    assert decision.response.answer == "4"
    assert decision.trace["cheap_answer"] == "5"
    assert decision.trace["review_action"] == "correct"
    assert strong.review_calls == 1


def test_calibrated_confidence_returns_probability() -> None:
    tasks = [
        TaskExample(
            id=f"math-{index}",
            dataset="unit",
            task_type="math",
            question=f"What is {index} plus 1?",
            answer=str(index + 1),
        )
        for index in range(12)
    ]
    responses = [
        ModelResponse(
            role="cheap",
            model="cheap-stub",
            answer=str(index + 1 if index < 8 else -1),
            raw_text="",
            confidence=0.9 if index < 8 else 0.4,
            latency_ms=1.0,
            metadata={"self_check_pass": index < 8},
        )
        for index in range(12)
    ]
    estimator = CalibratedConfidenceEstimator(include_task_features=True).fit(tasks, responses)

    probability = estimator.predict_correctness(tasks[0], responses[0])

    assert 0.0 <= probability <= 1.0
    assert estimator.diagnostics["method"] == "cross_validated_platt_scaling"


def test_learned_quality_gap_uses_request_features_and_routes() -> None:
    tasks = [
        TaskExample(
            id=f"task-{index}",
            dataset="unit",
            task_type="math" if index % 2 else "mcq",
            question=f"Shared reasoning question number {index % 3}",
            answer="4",
            metadata={"category": "text"},
        )
        for index in range(15)
    ]
    cheap = [
        ModelResponse("cheap", "cheap", "5" if index % 3 == 0 else "4", "", 0.5, 1.0)
        for index in range(15)
    ]
    strong = [ModelResponse("strong", "strong", "4", "", 0.8, 2.0) for _ in tasks]
    estimator = LearnedQualityGapEstimator().fit(tasks, cheap, strong)
    router = LearnedCostAwareRouter(estimator, advantage_threshold=-1.0)

    decision = router.route(tasks[0], {"cheap": StubProvider("cheap"), "strong": StubProvider("strong")})

    assert decision.selected_role == "strong"
    assert isinstance(decision.trace["predicted_strong_advantage"], float)
    assert estimator.diagnostics["strong_beneficial"] == 5


def test_outer_fold_predictions_cover_every_example() -> None:
    tasks = [
        TaskExample(
            id=f"task-{index}",
            dataset="unit",
            task_type="math",
            question=f"What is {index} plus one?",
            answer=str(index + 1),
            metadata={"category": "text" if index % 2 else "vision"},
        )
        for index in range(30)
    ]
    cheap = [
        ModelResponse(
            "cheap",
            "cheap",
            str(index + 1 if index % 4 else -1),
            "",
            0.8 if index % 4 else 0.3,
            1.0,
            {"self_check_pass": bool(index % 4)},
        )
        for index in range(30)
    ]
    strong = [ModelResponse("strong", "strong", str(index + 1), "", 0.9, 2.0) for index in range(30)]

    advantages = cross_validated_advantages(tasks, cheap, strong, folds=3)
    probabilities = cross_validated_correctness_probabilities(tasks, cheap, folds=3)

    assert len(advantages) == len(tasks)
    assert len(probabilities) == len(tasks)
    assert all(0.0 <= probability <= 1.0 for probability in probabilities)
