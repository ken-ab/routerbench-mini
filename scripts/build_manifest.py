from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from routerbench_mini.tasks import TaskExample, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RouterBench-Mini manifests from public datasets.")
    parser.add_argument("--out", default="data/manifest.jsonl")
    parser.add_argument("--gsm8k", type=int, default=100, help="Number of GSM8K test examples.")
    parser.add_argument("--scienceqa", type=int, default=100, help="Number of image-grounded ScienceQA examples.")
    parser.add_argument("--bfcl-file", default=None, help="Optional local BFCL JSONL file for tool-use examples.")
    parser.add_argument("--bfcl", type=int, default=100, help="Number of BFCL examples if --bfcl-file is provided.")
    parser.add_argument("--save-images", action="store_true", help="Save ScienceQA images under data/images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples: list[TaskExample] = []
    examples.extend(load_gsm8k(args.gsm8k))
    examples.extend(load_scienceqa(args.scienceqa, save_images=args.save_images))
    if args.bfcl_file:
        examples.extend(load_bfcl_jsonl(args.bfcl_file, args.bfcl))
    write_jsonl(args.out, examples)
    print(f"Wrote {len(examples)} examples to {args.out}")


def load_gsm8k(limit: int) -> list[TaskExample]:
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    examples: list[TaskExample] = []
    for idx, row in enumerate(dataset.select(range(min(limit, len(dataset))))):
        answer = _extract_gsm8k_answer(row["answer"])
        examples.append(
            TaskExample(
                id=f"gsm8k-{idx:04d}",
                dataset="gsm8k",
                task_type="math",
                question=row["question"],
                answer=answer,
                metadata={"source": "openai/gsm8k"},
            )
        )
    return examples


def load_scienceqa(limit: int, save_images: bool = False) -> list[TaskExample]:
    from datasets import load_dataset

    dataset = load_dataset("derek-thomas/ScienceQA", split="test")
    examples: list[TaskExample] = []
    image_dir = Path("data/images")
    if save_images:
        image_dir.mkdir(parents=True, exist_ok=True)

    for row in dataset:
        image = row.get("image")
        if image is None:
            continue
        choices = list(row.get("choices") or [])
        answer_idx = int(row.get("answer"))
        if not choices or answer_idx >= len(choices):
            continue

        image_path = None
        if save_images:
            image_path = str(image_dir / f"scienceqa-{len(examples):04d}.png")
            image.save(image_path)

        examples.append(
            TaskExample(
                id=f"scienceqa-{len(examples):04d}",
                dataset="scienceqa-img",
                task_type="vqa",
                question=row.get("question", ""),
                answer=chr(65 + answer_idx),
                choices=choices,
                image_path=image_path,
                metadata={"has_image": True, "source": "derek-thomas/ScienceQA"},
            )
        )
        if len(examples) >= limit:
            break
    return examples


def load_bfcl_jsonl(path: str, limit: int) -> list[TaskExample]:
    import json

    examples: list[TaskExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(examples) >= limit:
                break
            row = json.loads(line)
            converted = _convert_bfcl_row(row, len(examples))
            if converted:
                examples.append(converted)
    return examples


def _convert_bfcl_row(row: dict[str, Any], idx: int) -> TaskExample | None:
    question = row.get("question") or row.get("prompt") or row.get("user_prompt")
    tools = row.get("function") or row.get("tools") or row.get("functions")
    answer = row.get("ground_truth") or row.get("answer") or row.get("reference")
    if not question or not tools or not answer:
        return None
    if isinstance(tools, dict):
        tools = [tools]
    return TaskExample(
        id=f"bfcl-{idx:04d}",
        dataset="bfcl",
        task_type="tool",
        question=str(question),
        answer=answer,
        tools=list(tools),
        metadata={"source": "bfcl_local_jsonl"},
    )


def _extract_gsm8k_answer(answer_text: str) -> str:
    if "####" in answer_text:
        answer_text = answer_text.rsplit("####", 1)[1]
    matches = re.findall(r"-?\d+(?:\.\d+)?", answer_text.replace(",", ""))
    return matches[-1] if matches else answer_text.strip()


if __name__ == "__main__":
    main()

