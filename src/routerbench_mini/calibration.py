from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .features import task_feature_vector
from .providers import ModelResponse
from .scoring import is_correct
from .tasks import TaskExample
from .verifiers import verify_response


class ConfidenceEstimator(Protocol):
    def predict_correctness(self, task: TaskExample, response: ModelResponse) -> float: ...


@dataclass(frozen=True)
class RawConfidenceEstimator:
    """Ablation baseline that trusts the model's self-reported confidence."""

    def predict_correctness(self, task: TaskExample, response: ModelResponse) -> float:
        del task
        return response.confidence


class CalibratedConfidenceEstimator:
    """Estimate P(cheap answer is correct) from validation-only evidence."""

    def __init__(self, *, include_task_features: bool = True) -> None:
        self.include_task_features = include_task_features
        self._model: object | None = None
        self._fallback_probability = 0.5
        self.diagnostics: dict[str, float | int | str] = {
            "method": "unfitted",
            "examples": 0,
        }

    def fit(
        self,
        tasks: Sequence[TaskExample],
        responses: Sequence[ModelResponse],
    ) -> "CalibratedConfidenceEstimator":
        if len(tasks) != len(responses) or not tasks:
            raise ValueError("Calibration needs equally sized, non-empty task and response lists.")

        labels = [int(is_correct(task, response)) for task, response in zip(tasks, responses)]
        self._fallback_probability = (sum(labels) + 1) / (len(labels) + 2)
        minority_count = min(sum(labels), len(labels) - sum(labels))
        if minority_count < 2:
            self.diagnostics = {
                "method": "laplace_fallback",
                "examples": len(labels),
                "empirical_accuracy": round(sum(labels) / len(labels), 6),
                "fallback_probability": round(self._fallback_probability, 6),
            }
            return self

        try:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:  # pragma: no cover - depends on study extra
            raise RuntimeError(
                'Probability calibration requires scikit-learn; install with pip install -e ".[study]".'
            ) from exc

        features = [self._feature_vector(task, response) for task, response in zip(tasks, responses)]
        folds = min(3, minority_count)
        base_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, max_iter=1000, random_state=42),
        )
        model = CalibratedClassifierCV(base_model, method="sigmoid", cv=folds)
        model.fit(features, labels)
        self._model = model

        probabilities = [float(value[1]) for value in model.predict_proba(features)]
        brier = sum((probability - label) ** 2 for probability, label in zip(probabilities, labels)) / len(labels)
        self.diagnostics = {
            "method": "cross_validated_platt_scaling",
            "examples": len(labels),
            "folds": folds,
            "empirical_accuracy": round(sum(labels) / len(labels), 6),
            "brier_on_calibration_set": round(brier, 6),
            "include_task_features": str(self.include_task_features).lower(),
        }
        return self

    def predict_correctness(self, task: TaskExample, response: ModelResponse) -> float:
        if self._model is None:
            return self._fallback_probability
        probabilities = self._model.predict_proba([self._feature_vector(task, response)])
        return max(0.0, min(1.0, float(probabilities[0][1])))

    def _feature_vector(self, task: TaskExample, response: ModelResponse) -> list[float]:
        verification = verify_response(task, response, confidence_threshold=0.0)
        features = [
            float(response.confidence),
            float(verification.valid_format),
            float(response.metadata.get("self_check_pass", True)),
        ]
        if self.include_task_features:
            features.extend(task_feature_vector(task))
        return features
