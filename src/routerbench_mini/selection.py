from __future__ import annotations

from typing import Any, Sequence

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
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 1.0,
        sublinear_tf: bool = True,
        strip_accents: str | None = "unicode",
        norm: str = "l2",
        include_text: bool = True,
        include_structured: bool = True,
    ) -> None:
        if not include_text and not include_structured:
            raise ValueError("Quality-gap estimation needs text, structured features, or both.")
        self.alpha = alpha
        self.max_text_features = max_text_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self.strip_accents = strip_accents
        self.norm = norm
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
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                max_features=self.max_text_features,
                sublinear_tf=self.sublinear_tf,
                strip_accents=self.strip_accents,
                norm=self.norm,
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
            "tfidf": {
                "ngram_range": list(self.ngram_range),
                "min_df": self.min_df,
                "max_df": self.max_df,
                "max_features": self.max_text_features,
                "sublinear_tf": self.sublinear_tf,
                "strip_accents": self.strip_accents,
                "norm": self.norm,
            },
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

    def predict_advantages(self, tasks: Sequence[TaskExample]) -> list[float]:
        return [self.predict_advantage(task) for task in tasks]


def cross_validated_advantages(
    tasks: Sequence[TaskExample],
    cheap_responses: Sequence[ModelResponse],
    strong_responses: Sequence[ModelResponse],
    *,
    folds: int = 5,
    alpha: float = 0.1,
    max_text_features: int = 1500,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 1.0,
    sublinear_tf: bool = True,
    strip_accents: str | None = "unicode",
    norm: str = "l2",
    include_text: bool = True,
    include_structured: bool = True,
) -> list[float]:
    """Return leakage-resistant out-of-fold quality-gap predictions."""

    if len(tasks) < folds or len(tasks) != len(cheap_responses) or len(tasks) != len(strong_responses):
        raise ValueError("Cross-validation needs aligned responses and at least one example per fold.")

    predictions = [0.0] * len(tasks)
    for train_indices, validation_indices in _fold_splits(tasks, folds):
        estimator = LearnedQualityGapEstimator(
            alpha=alpha,
            max_text_features=max_text_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            strip_accents=strip_accents,
            norm=norm,
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


def fold_splits(tasks: Sequence[TaskExample], folds: int) -> list[tuple[list[int], list[int]]]:
    """Expose the exact CV split used by estimators for audit reports."""

    return _fold_splits(tasks, folds)


def _fold_splits(tasks: Sequence[TaskExample], folds: int) -> list[tuple[list[int], list[int]]]:
    frozen = [task.fold_id for task in tasks]
    if all(value is not None for value in frozen):
        fold_values = sorted({int(value) for value in frozen if value is not None})
        if fold_values != list(range(folds)):
            raise ValueError(f"Frozen fold IDs must be 0..{folds - 1}; got {fold_values}")
        return [
            (
                [index for index, value in enumerate(frozen) if int(value) != fold],
                [index for index, value in enumerate(frozen) if int(value) == fold],
            )
            for fold in fold_values
        ]

    from sklearn.model_selection import StratifiedKFold

    strata = [str(task.metadata.get("category", task.task_type)) for task in tasks]
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    indices = list(range(len(tasks)))
    return [
        (list(train_indices), list(validation_indices))
        for train_indices, validation_indices in splitter.split(indices, strata)
    ]


def _join_features(text_matrix: object | None, structured: object | None) -> object:
    from scipy.sparse import csr_matrix, hstack

    if text_matrix is None:
        assert structured is not None
        return csr_matrix(structured)
    if structured is None:
        return text_matrix
    return hstack([text_matrix, csr_matrix(structured)], format="csr")
