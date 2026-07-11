from __future__ import annotations

from typing import Sequence

from .features import task_feature_names, task_feature_vector
from .providers import ModelResponse
from .scoring import is_correct
from .tasks import TaskExample


class LearnedQualityGapEstimator:
    """Predict Strong-minus-Cheap accuracy from request-time features only."""

    def __init__(
        self,
        *,
        alpha: float = 0.1,
        max_text_features: int = 1500,
        include_text: bool = True,
        include_structured: bool = True,
    ) -> None:
        if not include_text and not include_structured:
            raise ValueError("Quality-gap estimation needs text, structured features, or both.")
        self.alpha = alpha
        self.max_text_features = max_text_features
        self.include_text = include_text
        self.include_structured = include_structured
        self._vectorizer: object | None = None
        self._scaler: object | None = None
        self._model: object | None = None
        self.diagnostics: dict[str, object] = {"method": "unfitted", "examples": 0}

    def fit(
        self,
        tasks: Sequence[TaskExample],
        cheap_responses: Sequence[ModelResponse],
        strong_responses: Sequence[ModelResponse],
    ) -> "LearnedQualityGapEstimator":
        if not tasks or len(tasks) != len(cheap_responses) or len(tasks) != len(strong_responses):
            raise ValueError("Quality-gap fitting needs equally sized, non-empty task and response lists.")

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        targets = [
            float(is_correct(task, strong) - is_correct(task, cheap))
            for task, cheap, strong in zip(tasks, cheap_responses, strong_responses)
        ]
        vectorizer = None
        text_matrix = None
        if self.include_text:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=self.max_text_features,
                sublinear_tf=True,
                strip_accents="unicode",
            )
            text_matrix = vectorizer.fit_transform(task.question for task in tasks)
        scaler = None
        structured = None
        if self.include_structured:
            scaler = StandardScaler()
            structured = scaler.fit_transform([task_feature_vector(task) for task in tasks])
        matrix = _join_features(text_matrix, structured)
        model = Ridge(alpha=self.alpha)
        model.fit(matrix, targets)

        self._vectorizer = vectorizer
        self._scaler = scaler
        self._model = model
        self.diagnostics = {
            "method": "tfidf_structured_ridge_quality_gap",
            "examples": len(tasks),
            "alpha": self.alpha,
            "text_features": len(vectorizer.get_feature_names_out()) if vectorizer is not None else 0,
            "structured_features": task_feature_names() if scaler is not None else [],
            "strong_beneficial": sum(target > 0 for target in targets),
            "cheap_beneficial": sum(target < 0 for target in targets),
            "ties": sum(target == 0 for target in targets),
        }
        return self

    def predict_advantage(self, task: TaskExample) -> float:
        if self._model is None:
            raise RuntimeError("Quality-gap estimator must be fitted before prediction.")
        text_matrix = self._vectorizer.transform([task.question]) if self._vectorizer is not None else None
        structured = (
            self._scaler.transform([task_feature_vector(task)]) if self._scaler is not None else None
        )
        matrix = _join_features(text_matrix, structured)
        return float(self._model.predict(matrix)[0])


def cross_validated_advantages(
    tasks: Sequence[TaskExample],
    cheap_responses: Sequence[ModelResponse],
    strong_responses: Sequence[ModelResponse],
    *,
    folds: int = 5,
    alpha: float = 0.1,
    max_text_features: int = 1500,
    include_text: bool = True,
    include_structured: bool = True,
) -> list[float]:
    """Return leakage-resistant out-of-fold quality-gap predictions."""

    if len(tasks) < folds or len(tasks) != len(cheap_responses) or len(tasks) != len(strong_responses):
        raise ValueError("Cross-validation needs aligned responses and at least one example per fold.")

    from sklearn.model_selection import StratifiedKFold

    strata = [str(task.metadata.get("category", task.task_type)) for task in tasks]
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    predictions = [0.0] * len(tasks)
    indices = list(range(len(tasks)))
    for train_indices, validation_indices in splitter.split(indices, strata):
        estimator = LearnedQualityGapEstimator(
            alpha=alpha,
            max_text_features=max_text_features,
            include_text=include_text,
            include_structured=include_structured,
        ).fit(
            [tasks[index] for index in train_indices],
            [cheap_responses[index] for index in train_indices],
            [strong_responses[index] for index in train_indices],
        )
        for index in validation_indices:
            predictions[index] = estimator.predict_advantage(tasks[index])
    return predictions


def _join_features(text_matrix: object | None, structured: object | None) -> object:
    from scipy.sparse import csr_matrix, hstack

    if text_matrix is None:
        assert structured is not None
        return csr_matrix(structured)
    if structured is None:
        return text_matrix
    return hstack([text_matrix, csr_matrix(structured)], format="csr")
