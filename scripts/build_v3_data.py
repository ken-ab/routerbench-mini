from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Callable

from routerbench_mini.tasks import TaskExample, load_jsonl, write_jsonl

try:
    from scripts.build_manifest import (
        load_bbh_logic,
        load_bfcl,
        load_chartqa,
        load_commonsense_qa,
        load_gsm8k,
        load_mmmu,
        load_ocr_vqa,
        load_scienceqa,
    )
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repository root.
    from build_manifest import (
        load_bbh_logic,
        load_bfcl,
        load_chartqa,
        load_commonsense_qa,
        load_gsm8k,
        load_mmmu,
        load_ocr_vqa,
        load_scienceqa,
    )


TARGET_COUNTS = {
    "gsm8k": 20,
    "commonsenseqa": 15,
    "bbh-logical-deduction": 15,
    "scienceqa-img": 20,
    "chartqa": 10,
    "ocr-vqa": 10,
    "mmmu": 10,
    "bfcl-simple_python": 25,
    "bfcl-multiple": 25,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the unseen 150-example V3 test set.")
    parser.add_argument("--development", nargs="+", default=["data/manifest.jsonl"])
    parser.add_argument("--out", default="data/v3_test.jsonl")
    parser.add_argument("--image-dir", default="data/v3_images")
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--version", default="v3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    development = [task for path in args.development for task in load_jsonl(path)]
    excluded = {task_fingerprint(task) for task in development}
    development_fingerprints = set(excluded)
    selected: list[TaskExample] = []

    with tempfile.TemporaryDirectory(prefix="routerbench-v3-") as temporary:
        root = Path(temporary)
        text_loaders: list[tuple[str, int, Callable[[], list[TaskExample]]]] = [
            ("gsm8k", 20, lambda: load_gsm8k(160, args.seed)),
            ("commonsenseqa", 15, lambda: load_commonsense_qa(120, args.seed)),
            ("bbh-logical-deduction", 15, lambda: load_bbh_logic(120, args.seed)),
        ]
        for dataset, count, loader in text_loaders:
            selected.extend(select_unseen(loader(), count, excluded, dataset))

        for directory in (root / "scienceqa", root / "chartqa", root / "ocr"):
            directory.mkdir(parents=True, exist_ok=True)
        vision_loaders: list[tuple[str, int, Callable[[], list[TaskExample]]]] = [
            ("scienceqa-img", 20, lambda: load_scienceqa(120, args.seed, root / "scienceqa")),
            ("chartqa", 10, lambda: load_chartqa(80, args.seed, root / "chartqa")),
            ("ocr-vqa", 10, lambda: load_ocr_vqa(80, root / "ocr")),
        ]
        for dataset, count, loader in vision_loaders:
            selected.extend(select_unseen(loader(), count, excluded, dataset))

        mmmu_pool: list[TaskExample] = []
        for offset in range(5):
            mmmu_root = root / f"mmmu-{offset}"
            mmmu_root.mkdir(parents=True, exist_ok=True)
            mmmu_pool.extend(load_mmmu(20, args.seed + offset, mmmu_root))
        selected.extend(select_unseen(mmmu_pool, 10, excluded, "mmmu"))

        selected.extend(
            select_unseen(load_bfcl("simple_python", 150, args.seed), 25, excluded, "bfcl-simple_python")
        )
        selected.extend(select_unseen(load_bfcl("multiple", 150, args.seed), 25, excluded, "bfcl-multiple"))

        finalized = finalize_examples(selected, Path(args.image_dir), args.seed, args.version)

    random.Random(args.seed).shuffle(finalized)
    validate_v3(finalized, development_fingerprints)
    write_jsonl(args.out, finalized)
    print(f"Wrote {len(finalized)} unseen {args.version.upper()} test examples to {args.out}")


def task_fingerprint(task: TaskExample) -> str:
    image_digest = None
    if task.image_path and Path(task.image_path).exists():
        image_digest = hashlib.sha256(Path(task.image_path).read_bytes()).hexdigest()
    payload = {
        "question": " ".join(task.question.lower().split()),
        "choices": task.choices,
        "tools": task.tools,
        "image_sha256": image_digest,
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def select_unseen(
    candidates: list[TaskExample],
    count: int,
    excluded: set[str],
    dataset: str,
) -> list[TaskExample]:
    output: list[TaskExample] = []
    seen = set(excluded)
    for task in candidates:
        fingerprint = task_fingerprint(task)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        output.append(task)
        if len(output) == count:
            break
    if len(output) != count:
        raise ValueError(f"{dataset} yielded {len(output)} unseen examples; expected {count}")
    excluded.update(task_fingerprint(task) for task in output)
    return output


def finalize_examples(
    tasks: list[TaskExample], image_dir: Path, seed: int, version: str
) -> list[TaskExample]:
    image_dir.mkdir(parents=True, exist_ok=True)
    counters: dict[str, int] = {}
    output: list[TaskExample] = []
    for task in tasks:
        index = counters.get(task.dataset, 0)
        counters[task.dataset] = index + 1
        slug = task.dataset.replace("_", "-")
        task_id = f"{version}-{slug}-{index:04d}"
        image_path: str | None = None
        if task.image_path:
            source = Path(task.image_path)
            suffix = source.suffix.lower() or ".png"
            destination = image_dir / f"{task_id}{suffix}"
            shutil.copyfile(source, destination)
            image_path = str(destination)
        output.append(
            replace(
                task,
                id=task_id,
                image_path=image_path,
                metadata={
                    **task.metadata,
                    "benchmark_version": version,
                    "held_out": True,
                    "selection_seed": seed,
                },
            )
        )
    return output


def validate_v3(tasks: list[TaskExample], development_fingerprints: set[str]) -> None:
    counts: dict[str, int] = {}
    fingerprints: set[str] = set()
    for task in tasks:
        counts[task.dataset] = counts.get(task.dataset, 0) + 1
        fingerprint = task_fingerprint(task)
        if fingerprint in development_fingerprints:
            raise ValueError(f"Development leakage detected for {task.id}")
        if fingerprint in fingerprints:
            raise ValueError(f"Duplicate V3 task detected for {task.id}")
        fingerprints.add(fingerprint)
        if task.requires_vision and (not task.image_path or not Path(task.image_path).exists()):
            raise ValueError(f"Missing V3 image for {task.id}")
    if counts != TARGET_COUNTS:
        raise ValueError(f"Unexpected V3 dataset counts: {counts}; expected {TARGET_COUNTS}")


if __name__ == "__main__":
    main()
