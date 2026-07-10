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

