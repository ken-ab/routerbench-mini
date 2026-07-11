from __future__ import annotations

import argparse
import ast
import io
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable

import requests

from routerbench_mini.tasks import TaskExample, write_jsonl


BBH_LOGIC_URL = (
    "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/"
    "main/bbh/logical_deduction_three_objects.json"
)
BFCL_DATA_ROOT = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/"
    "berkeley-function-call-leaderboard/bfcl_eval/data"
)
MMMU_SUBJECTS = (
    "Accounting",
    "Agriculture",
    "Architecture_and_Engineering",
    "Art",
    "Biology",
    "Chemistry",
    "Computer_Science",
    "Design",
    "Economics",
    "Electronics",
    "Energy_and_Power",
    "Finance",
    "Geography",
    "History",
    "Literature",
    "Materials",
    "Math",
    "Mechanical_Engineering",
    "Music",
    "Physics",
)
PROVIDER_BLOCKED_PHRASES = ("dalai lama",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the 300-example RouterBench-Mini study dataset.")
    parser.add_argument("--out", default="data/manifest.jsonl")
    parser.add_argument("--validation-out", default="data/validation.jsonl")
    parser.add_argument("--test-out", default="data/test.jsonl")
    parser.add_argument("--image-dir", default="data/images")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    examples = [
        *load_gsm8k(40, args.seed),
        *load_commonsense_qa(30, args.seed),
        *load_bbh_logic(30, args.seed),
        *load_scienceqa(40, args.seed, image_dir),
        *load_chartqa(20, args.seed, image_dir),
        *load_ocr_vqa(20, image_dir),
        *load_mmmu(20, args.seed, image_dir),
        *load_bfcl("simple_python", 50, args.seed),
        *load_bfcl("multiple", 50, args.seed),
    ]
    _validate_counts(examples)
    validation, test = stratified_split(examples, args.validation_ratio, args.seed)

    write_jsonl(args.out, examples)
    write_jsonl(args.validation_out, validation)
    write_jsonl(args.test_out, test)
    print(f"Wrote {len(examples)} examples to {args.out}")
    print(f"Wrote {len(validation)} validation examples to {args.validation_out}")
    print(f"Wrote {len(test)} test examples to {args.test_out}")


def load_gsm8k(limit: int, seed: int) -> list[TaskExample]:
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=seed)
    return [
        TaskExample(
            id=f"gsm8k-{idx:04d}",
            dataset="gsm8k",
            task_type="math",
            question=row["question"],
            answer=_extract_gsm8k_answer(row["answer"]),
            metadata={
                "source": "openai/gsm8k",
                "category": "text",
            },
        )
        for idx, row in enumerate(dataset.select(range(limit)))
    ]


def load_commonsense_qa(limit: int, seed: int) -> list[TaskExample]:
    from datasets import load_dataset

    dataset = load_dataset("tau/commonsense_qa", split="validation").shuffle(seed=seed)
    examples: list[TaskExample] = []
    for idx, row in enumerate(dataset.select(range(limit))):
        labels = list(row["choices"]["label"])
        choices = list(row["choices"]["text"])
        examples.append(
            TaskExample(
                id=f"commonsenseqa-{idx:04d}",
                dataset="commonsenseqa",
                task_type="mcq",
                question=row["question"],
                answer=labels.index(row["answerKey"]),
                choices=choices,
                metadata={
                    "source": "tau/commonsense_qa",
                    "category": "text",
                },
            )
        )
    return examples


def load_bbh_logic(limit: int, seed: int) -> list[TaskExample]:
    payload = _get_json(BBH_LOGIC_URL)
    rows = list(payload["examples"])
    random.Random(seed).shuffle(rows)
    examples: list[TaskExample] = []
    for idx, row in enumerate(rows[:limit]):
        question, choices = _split_bbh_question(row["input"])
        answer = ord(row["target"].strip("()")) - ord("A")
        examples.append(
            TaskExample(
                id=f"bbh-logical-{idx:04d}",
                dataset="bbh-logical-deduction",
                task_type="mcq",
                question=question,
                answer=answer,
                choices=choices,
                metadata={
                    "source": "BIG-Bench-Hard/logical_deduction_three_objects",
                    "category": "text",
                },
            )
        )
    return examples


def load_scienceqa(limit: int, seed: int, image_dir: Path) -> list[TaskExample]:
    from datasets import load_dataset

    dataset = load_dataset("derek-thomas/ScienceQA", split="test").shuffle(seed=seed)
    examples: list[TaskExample] = []
    for row in dataset:
        image = row.get("image")
        choices = list(row.get("choices") or [])
        if image is None or not choices:
            continue
        image_path = image_dir / f"scienceqa-{len(examples):04d}.png"
        image.convert("RGB").save(image_path, optimize=True)
        hint = str(row.get("hint") or "").strip()
        question = row.get("question", "")
        if hint:
            question = f"Context: {hint}\nQuestion: {question}"
        examples.append(
            TaskExample(
                id=f"scienceqa-{len(examples):04d}",
                dataset="scienceqa-img",
                task_type="vqa",
                question=question,
                answer=int(row["answer"]),
                choices=choices,
                image_path=str(image_path),
                metadata={
                    "source": "derek-thomas/ScienceQA",
                    "category": "vision",
                    "vision_subtype": "scienceqa",
                    "grade": row.get("grade"),
                    "subject": row.get("subject"),
                },
            )
        )
        if len(examples) >= limit:
            break
    return examples


def load_chartqa(limit: int, seed: int, image_dir: Path) -> list[TaskExample]:
    from datasets import load_dataset

    dataset = load_dataset("docintel/ChartQA", split="test").shuffle(seed=seed)
    examples: list[TaskExample] = []
    for idx, row in enumerate(dataset.select(range(limit))):
        image_path = image_dir / f"chartqa-{idx:04d}.png"
        row["image"].convert("RGB").save(image_path, optimize=True)
        examples.append(
            TaskExample(
                id=f"chartqa-{idx:04d}",
                dataset="chartqa",
                task_type="vqa",
                question=row["question"],
                answer=str(row["answer"]),
                image_path=str(image_path),
                metadata={
                    "source": "docintel/ChartQA",
                    "category": "vision",
                    "vision_subtype": "chart",
                },
            )
        )
    return examples


def load_ocr_vqa(limit: int, image_dir: Path) -> list[TaskExample]:
    from datasets import load_dataset

    dataset = load_dataset("pppop7/OCR-VQA", split="train", streaming=True)
    examples: list[TaskExample] = []
    for row in dataset:
        questions = list(row.get("questions") or [])
        answers = list(row.get("answers") or [])
        image = row.get("image")
        if image is None or not questions or not answers:
            continue
        question_and_answer = f"{questions[0]} {answers[0]}".lower()
        if any(phrase in question_and_answer for phrase in PROVIDER_BLOCKED_PHRASES):
            continue
        image_path = image_dir / f"ocr-vqa-{len(examples):04d}.jpg"
        image.convert("RGB").save(image_path, quality=88, optimize=True)
        examples.append(
            TaskExample(
                id=f"ocr-vqa-{len(examples):04d}",
                dataset="ocr-vqa",
                task_type="vqa",
                question=str(questions[0]),
                answer=str(answers[0]),
                image_path=str(image_path),
                metadata={
                    "source": "pppop7/OCR-VQA",
                    "category": "vision",
                    "vision_subtype": "ocr",
                },
            )
        )
        if len(examples) >= limit:
            break
    return examples


def load_mmmu(limit: int, seed: int, image_dir: Path) -> list[TaskExample]:
    from datasets import load_dataset

    examples: list[TaskExample] = []
    cached_rows = _cached_mmmu_validation_rows()
    for subject in MMMU_SUBJECTS:
        if cached_rows:
            rows = [row for row in cached_rows if str(row.get("id", "")).startswith(f"validation_{subject}_")]
        else:
            rows = list(load_dataset("MMMU/MMMU", subject, split="validation"))
        random.Random(f"{seed}:{subject}").shuffle(rows)
        for row in rows:
            images = [row.get(f"image_{index}") for index in range(1, 8)]
            present_images = [image for image in images if image is not None]
            choices = _parse_mmmu_options(row.get("options"))
            answer = str(row.get("answer") or "").strip().upper()
            if len(present_images) != 1 or not choices or len(answer) != 1 or answer not in "ABCDEFG":
                continue
            answer_index = ord(answer) - ord("A")
            if answer_index >= len(choices):
                continue

            image_path = image_dir / f"mmmu-{len(examples):04d}.png"
            _save_mmmu_image(present_images[0], image_path)
            question = str(row.get("question") or "").strip()
            question = re.sub(r"<image[_ ]?\d+>", "the image", question, flags=re.IGNORECASE)
            examples.append(
                TaskExample(
                    id=f"mmmu-{len(examples):04d}",
                    dataset="mmmu",
                    task_type="vqa",
                    question=question,
                    answer=answer_index,
                    choices=choices,
                    image_path=str(image_path),
                    metadata={
                        "source": "MMMU/MMMU",
                        "category": "vision",
                        "vision_subtype": "multidiscipline",
                        "subject": subject,
                        "topic_difficulty": row.get("topic_difficulty"),
                    },
                )
            )
            break
        if len(examples) >= limit:
            break
    if len(examples) != limit:
        raise ValueError(f"MMMU yielded {len(examples)} usable single-image MCQs; expected {limit}")
    return examples


def _cached_mmmu_validation_rows() -> list[dict[str, Any]]:
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--lmms-lab--MMMU" / "snapshots"
    paths = sorted(cache_root.glob("*/data/validation-*.parquet"))
    if not paths:
        return []
    import pyarrow.parquet as pq

    return pq.read_table(paths[-1]).to_pylist()


def _save_mmmu_image(value: Any, path: Path) -> None:
    if hasattr(value, "convert"):
        image = value
    elif isinstance(value, dict) and value.get("bytes"):
        from PIL import Image

        image = Image.open(io.BytesIO(value["bytes"]))
    else:
        raise ValueError("Unsupported MMMU image payload")
    image.convert("RGB").save(path, optimize=True)


def load_bfcl(category: str, limit: int, seed: int) -> list[TaskExample]:
    data_name = f"BFCL_v4_{category}.json"
    rows = _get_jsonl(f"{BFCL_DATA_ROOT}/{data_name}")
    answers = {
        row["id"]: row["ground_truth"]
        for row in _get_jsonl(f"{BFCL_DATA_ROOT}/possible_answer/{data_name}")
    }
    random.Random(seed).shuffle(rows)
    examples: list[TaskExample] = []
    for row in rows:
        ground_truth = answers.get(row["id"])
        converted = _convert_bfcl_row(row, ground_truth, category, len(examples))
        if converted is not None:
            examples.append(converted)
        if len(examples) >= limit:
            break
    return examples


def _convert_bfcl_row(
    row: dict[str, Any],
    ground_truth: Any,
    category: str,
    idx: int,
) -> TaskExample | None:
    tools = list(row.get("function") or [])
    question = _bfcl_question(row.get("question"))
    answer = _canonical_bfcl_answer(ground_truth, tools)
    if not question or not tools or answer is None:
        return None
    return TaskExample(
        id=f"bfcl-{category}-{idx:04d}",
        dataset=f"bfcl-{category}",
        task_type="tool",
        question=question,
        answer=answer,
        tools=tools,
        metadata={
            "source": f"BFCL_v4_{category}",
            "category": "tool",
            "tool_subtype": "multiple" if category == "multiple" else "simple",
        },
    )


def stratified_split(
    examples: list[TaskExample], validation_ratio: float, seed: int
) -> tuple[list[TaskExample], list[TaskExample]]:
    grouped: dict[str, list[TaskExample]] = {}
    for example in examples:
        grouped.setdefault(str(example.metadata["category"]), []).append(example)
    validation: list[TaskExample] = []
    test: list[TaskExample] = []
    rng = random.Random(seed)
    for category in sorted(grouped):
        rows = grouped[category]
        rng.shuffle(rows)
        split_at = round(len(rows) * validation_ratio)
        validation.extend(rows[:split_at])
        test.extend(rows[split_at:])
    rng.shuffle(validation)
    rng.shuffle(test)
    return validation, test


def _validate_counts(examples: list[TaskExample]) -> None:
    counts: dict[str, int] = {}
    for example in examples:
        category = str(example.metadata.get("category"))
        counts[category] = counts.get(category, 0) + 1
        if example.requires_vision and not example.image_path:
            raise ValueError(f"Vision example {example.id} has no image path")
        if example.image_path and not Path(example.image_path).exists():
            raise ValueError(f"Missing image for {example.id}: {example.image_path}")
    expected = {"text": 100, "vision": 100, "tool": 100}
    if counts != expected:
        raise ValueError(f"Unexpected category counts: {counts}; expected {expected}")


def _get_json(url: str) -> dict[str, Any]:
    return _download(url).json()


def _get_jsonl(url: str) -> list[dict[str, Any]]:
    response = _download(url)
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def _download(url: str) -> requests.Response:
    error: requests.RequestException | None = None
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            error = exc
            if attempt < 3:
                time.sleep(2**attempt)
    assert error is not None
    raise error


def _extract_gsm8k_answer(answer_text: str) -> str:
    if "####" in answer_text:
        answer_text = answer_text.rsplit("####", 1)[1]
    matches = re.findall(r"-?\d+(?:\.\d+)?", answer_text.replace(",", ""))
    return matches[-1] if matches else answer_text.strip()


def _split_bbh_question(value: str) -> tuple[str, list[str]]:
    question, option_block = value.rsplit("\nOptions:\n", 1)
    choices = [re.sub(r"^\([A-Z]\)\s*", "", line) for line in option_block.splitlines() if line.strip()]
    return question, choices


def _parse_mmmu_options(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _bfcl_question(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    for turn in _walk(value):
        if isinstance(turn, dict) and turn.get("role") == "user" and turn.get("content"):
            return str(turn["content"])
    return ""


def _walk(value: Iterable[Any]) -> Iterable[Any]:
    for item in value:
        if isinstance(item, list):
            yield from _walk(item)
        else:
            yield item


def _canonical_bfcl_answer(ground_truth: Any, tools: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not isinstance(ground_truth, list) or not ground_truth:
        return None
    call = ground_truth[0]
    if not isinstance(call, dict) or not call:
        return None
    name, accepted_args = next(iter(call.items()))
    tool = next((tool for tool in tools if tool.get("name") == name), None)
    if tool is None or not isinstance(accepted_args, dict):
        return None
    required = set((tool.get("parameters") or {}).get("required") or [])
    arguments: dict[str, Any] = {}
    for key in required:
        values = accepted_args.get(key)
        if not isinstance(values, list) or not values:
            return None
        arguments[key] = next((value for value in values if value != ""), values[0])
    return {"name": name, "arguments": arguments}


if __name__ == "__main__":
    main()
