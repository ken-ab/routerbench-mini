from __future__ import annotations

from routerbench_mini.selection import fold_splits
from routerbench_mini.tasks import TaskExample
from routerbench_mini.v5 import select_non_degenerate_threshold


def test_frozen_manifest_fold_ids_are_used_exactly() -> None:
    tasks = [
        TaskExample(
            id=f"task-{index}",
            dataset="unit",
            task_type="text",
            question=f"Question {index}",
            answer="answer",
            fold_id=index % 2,
        )
        for index in range(10)
    ]

    splits = fold_splits(tasks, folds=2)

    assert splits[0][1] == [0, 2, 4, 6, 8]
    assert splits[1][1] == [1, 3, 5, 7, 9]
    assert set(splits[0][0]).isdisjoint(splits[0][1])


def test_threshold_selection_rejects_near_all_strong_solution() -> None:
    rows = [
        {
            "advantage_threshold": -1.0,
            "accuracy": 1.0,
            "avg_cost": 3.0,
            "strong_usage_rate": 0.999,
        },
        {
            "advantage_threshold": 0.1,
            "accuracy": 0.9,
            "avg_cost": 2.0,
            "strong_usage_rate": 0.5,
        },
    ]

    selected = select_non_degenerate_threshold(rows, "advantage_threshold")

    assert selected == 0.1
