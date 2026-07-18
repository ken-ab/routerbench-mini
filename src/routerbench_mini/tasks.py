from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskExample:
    id: str
    dataset: str
    task_type: str
    question: str
    answer: Any
    choices: list[str] = field(default_factory=list)
    image_path: str | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    canonical_id: str | None = None
    source_dataset: str | None = None
    source_version: str | None = None
    source_split: str | None = None
    source_id: str | None = None
    task_subtype: str | None = None
    difficulty_group: str | None = None
    difficulty_reason: str | None = None
    selection_seed: int | None = None
    fold_id: int | None = None
    image_id: str | None = None
    image_reference: str | None = None
    tool_category: str | None = None
    tool_schema_id: str | None = None
    prompt_hash: str | None = None
    answer_reference: Any = None
    license_note: str | None = None

    @property
    def requires_vision(self) -> bool:
        return bool(self.image_path) or bool(self.metadata.get("has_image"))

    @property
    def is_multiple_choice(self) -> bool:
        return bool(self.choices)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "TaskExample":
        return cls(
            id=str(row["id"]),
            dataset=str(row["dataset"]),
            task_type=str(row["task_type"]),
            question=str(row["question"]),
            answer=row["answer"],
            choices=list(row.get("choices") or []),
            image_path=row.get("image_path"),
            tools=list(row.get("tools") or []),
            metadata=dict(row.get("metadata") or {}),
            canonical_id=row.get("canonical_id"),
            source_dataset=row.get("source_dataset"),
            source_version=row.get("source_version"),
            source_split=row.get("source_split"),
            source_id=row.get("source_id"),
            task_subtype=row.get("task_subtype"),
            difficulty_group=row.get("difficulty_group"),
            difficulty_reason=row.get("difficulty_reason"),
            selection_seed=row.get("selection_seed"),
            fold_id=row.get("fold_id"),
            image_id=row.get("image_id"),
            image_reference=row.get("image_reference"),
            tool_category=row.get("tool_category"),
            tool_schema_id=row.get("tool_schema_id"),
            prompt_hash=row.get("prompt_hash"),
            answer_reference=row.get("answer_reference"),
            license_note=row.get("license_note"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dataset": self.dataset,
            "task_type": self.task_type,
            "question": self.question,
            "answer": self.answer,
            "choices": self.choices,
            "image_path": self.image_path,
            "tools": self.tools,
            "metadata": self.metadata,
            "canonical_id": self.canonical_id,
            "source_dataset": self.source_dataset,
            "source_version": self.source_version,
            "source_split": self.source_split,
            "source_id": self.source_id,
            "task_subtype": self.task_subtype,
            "difficulty_group": self.difficulty_group,
            "difficulty_reason": self.difficulty_reason,
            "selection_seed": self.selection_seed,
            "fold_id": self.fold_id,
            "image_id": self.image_id,
            "image_reference": self.image_reference,
            "tool_category": self.tool_category,
            "tool_schema_id": self.tool_schema_id,
            "prompt_hash": self.prompt_hash,
            "answer_reference": self.answer_reference,
            "license_note": self.license_note,
        }


def load_jsonl(path: str | Path) -> list[TaskExample]:
    examples: list[TaskExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(TaskExample.from_dict(json.loads(line)))
            except Exception as exc:  # pragma: no cover - keeps bad data debuggable
                raise ValueError(f"Failed to parse {path}:{line_number}: {exc}") from exc
    return examples


def write_jsonl(path: str | Path, examples: list[TaskExample]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
