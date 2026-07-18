from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import random
import re
import shutil
import urllib.request
from collections import Counter
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Sequence

from routerbench_mini.config import load_yaml
from routerbench_mini.providers import PROMPT_VERSION, build_prompt
from routerbench_mini.tasks import TaskExample, load_jsonl, write_jsonl


BBH_TASKS = (
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
)
HARD_BBH_TASKS = {
    "dyck_languages",
    "formal_fallacies",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "multistep_arithmetic_two",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "web_of_lies",
}
BFCL_SIMPLE_FILES = (
    "BFCL_v4_simple_python",
    "BFCL_v4_simple_java",
    "BFCL_v4_simple_javascript",
    "BFCL_v4_live_simple",
)
BFCL_MULTIPLE_FILES = ("BFCL_v4_multiple", "BFCL_v4_live_multiple")
PROVIDER_BLOCKED_PHRASES = ("dalai lama",)


@dataclass(frozen=True)
class Candidate:
    source_key: str
    task: TaskExample
    hard_score: float
    hard_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build frozen 3200/800 manifests for RouterBench-Mini V5.")
    parser.add_argument("--config", default="configs/v5_large_scale.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--refresh-source-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    paths = config["paths"]
    development_path = Path(paths["development_manifest"])
    test_path = Path(paths["test_manifest"])
    if not args.force and (development_path.exists() or test_path.exists()):
        raise FileExistsError("V5 manifests already exist. Pass --force only before the protocol is frozen.")

    selection_seed = int(config["selection_seed"])
    image_dir = Path(paths["image_dir"])
    image_dir.mkdir(parents=True, exist_ok=True)
    legacy = load_legacy_tasks()

    candidate_root = Path(paths["root"]) / "candidate_cache"
    candidate_index = candidate_root / "cache_index.json"
    config_digest = sha256_file(Path(args.config))
    if (
        candidate_index.exists()
        and not args.refresh_source_cache
        and json.loads(candidate_index.read_text(encoding="utf-8")).get("config_sha256") == config_digest
    ):
        pools = read_candidate_pools(candidate_root, config)
    else:
        if candidate_root.exists():
            shutil.rmtree(candidate_root)
        candidate_root.mkdir(parents=True, exist_ok=True)
        pools = load_candidate_pools(config, candidate_root)
        write_candidate_pools(candidate_root, pools, config_digest)

    development, test, allocation = allocate_all(config, pools, legacy)
    development = assign_folds(
        development,
        folds=int(config["cross_validation_folds"]),
        seed=int(config["fold_seed"]),
    )
    development = finalize_tasks(development, image_dir / "development", selection_seed)
    test = finalize_tasks(test, image_dir / "test", selection_seed)

    validate_manifest_fields(development, require_fold=True)
    validate_manifest_fields(test, require_fold=False)
    overlap_report = build_overlap_report(development, test, legacy, config)
    write_json(Path(paths["overlap_report"]), overlap_report)
    if overlap_report["blocking_overlap_count"]:
        write_jsonl(Path(paths["root"]) / "development_manifest.draft.jsonl", development)
        write_jsonl(Path(paths["root"]) / "test_manifest.draft.jsonl", test)
        raise ValueError(
            "Near-duplicate leakage remains after allocation; inspect candidates before freezing: "
            f"{overlap_report['blocking_overlap_count']} blocking overlaps"
        )

    (Path(paths["root"]) / "development_manifest.draft.jsonl").unlink(missing_ok=True)
    (Path(paths["root"]) / "test_manifest.draft.jsonl").unlink(missing_ok=True)
    write_jsonl(development_path, development)
    write_jsonl(test_path, test)
    data_report = build_data_report(development, test, allocation, config)
    write_json(Path(paths["data_report"]), data_report)
    hashes = {
        "protocol_config": sha256_file(Path(args.config)),
        "development_manifest": sha256_file(development_path),
        "test_manifest": sha256_file(test_path),
        "prompt_version": PROMPT_VERSION,
        "development_examples": len(development),
        "test_examples": len(test),
    }
    write_json(Path(paths["manifest_hashes"]), hashes)
    print(f"Wrote {len(development)} development tasks to {development_path}")
    print(f"Wrote {len(test)} frozen test tasks to {test_path}")
    print(f"Development SHA-256: {hashes['development_manifest']}")
    print(f"Test SHA-256: {hashes['test_manifest']}")


def write_candidate_pools(
    root: Path, pools: dict[str, list[Candidate]], config_digest: str
) -> None:
    pool_dir = root / "pools"
    pool_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for source_key, candidates in pools.items():
        path = pool_dir / f"{source_key}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for candidate in candidates:
                handle.write(
                    json.dumps(
                        {
                            "source_key": candidate.source_key,
                            "hard_score": candidate.hard_score,
                            "hard_reason": candidate.hard_reason,
                            "task": candidate.task.to_dict(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        counts[source_key] = len(candidates)
    write_json(
        root / "cache_index.json",
        {"config_sha256": config_digest, "pool_counts": counts},
    )


def read_candidate_pools(
    root: Path, config: dict[str, Any]
) -> dict[str, list[Candidate]]:
    output: dict[str, list[Candidate]] = {}
    for source_key in config["development"]["standard"]:
        path = root / "pools" / f"{source_key}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Candidate cache is incomplete: {path}")
        candidates: list[Candidate] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            candidates.append(
                Candidate(
                    source_key=str(row["source_key"]),
                    task=TaskExample.from_dict(row["task"]),
                    hard_score=float(row["hard_score"]),
                    hard_reason=str(row["hard_reason"]),
                )
            )
        output[source_key] = candidates
    return output


def load_candidate_pools(config: dict[str, Any], root: Path) -> dict[str, list[Candidate]]:
    seed = int(config["selection_seed"])
    needed = {
        key: sum(int(config[split][group].get(key, 0)) for split in ("development", "test") for group in ("standard", "hard"))
        for key in config["development"]["standard"]
    }
    target = {key: max(count + 80, min(count * 2, count + 600)) for key, count in needed.items()}
    sources = config["sources"]
    return {
        "gsm8k": load_gsm8k(target["gsm8k"], seed, sources["gsm8k"]),
        "commonsenseqa": load_commonsenseqa(target["commonsenseqa"], seed, sources["commonsenseqa"]),
        "bbh": load_bbh(target["bbh"], seed, sources["bbh"]),
        "scienceqa": load_scienceqa(target["scienceqa"], seed, sources["scienceqa"], root / "scienceqa"),
        "mmmu": load_mmmu(target["mmmu"], seed, sources["mmmu"], root / "mmmu"),
        "chartqa": load_chartqa(target["chartqa"], seed, sources["chartqa"], root / "chartqa"),
        "ocr_vqa": load_ocr_vqa(target["ocr_vqa"], seed, sources["ocr_vqa"], root / "ocr_vqa"),
        "bfcl_simple": load_bfcl(BFCL_SIMPLE_FILES, "bfcl_simple", seed, sources["bfcl"]),
        "bfcl_multiple": load_bfcl(BFCL_MULTIPLE_FILES, "bfcl_multiple", seed, sources["bfcl"]),
    }


def load_gsm8k(limit: int, seed: int, source: dict[str, Any]) -> list[Candidate]:
    from datasets import load_dataset

    dataset = load_dataset(
        source["repository"], "main", split=source["split"], revision=source["version"]
    )
    indices = list(range(len(dataset)))
    random.Random(f"{seed}:gsm8k-pool").shuffle(indices)
    output: list[Candidate] = []
    for index in indices[:limit]:
        row = dataset[index]
        steps = str(row["answer"]).count("<<")
        question = str(row["question"])
        score = steps * 2 + word_count(question) / 20 + numeric_count(question)
        output.append(
            make_candidate(
                "gsm8k",
                source,
                source_id=f"{source['split']}:{index}",
                task_type="math",
                task_subtype="mathematical_reasoning",
                question=question,
                answer=extract_gsm8k_answer(str(row["answer"])),
                hard_score=score,
                hard_reason=f"pre_model_solution_steps={steps};question_words={word_count(question)}",
                metadata={"category": "text", "solution_step_count": steps},
            )
        )
    return output


def load_commonsenseqa(limit: int, seed: int, source: dict[str, Any]) -> list[Candidate]:
    from datasets import load_dataset

    dataset = load_dataset(source["repository"], split=source["split"], revision=source["version"])
    indices = list(range(len(dataset)))
    random.Random(f"{seed}:commonsenseqa-pool").shuffle(indices)
    output: list[Candidate] = []
    for index in indices[:limit]:
        row = dataset[index]
        choices = list(row["choices"]["text"])
        labels = list(row["choices"]["label"])
        question = str(row["question"])
        option_words = sum(word_count(value) for value in choices)
        score = word_count(question) + option_words * 0.5
        output.append(
            make_candidate(
                "commonsenseqa",
                source,
                source_id=str(row.get("id") or f"{source['split']}:{index}"),
                task_type="mcq",
                task_subtype="text_multiple_choice",
                question=question,
                answer=labels.index(str(row["answerKey"])),
                choices=choices,
                hard_score=score,
                hard_reason=f"question_words={word_count(question)};option_words={option_words}",
                metadata={"category": "text", "question_concept": row.get("question_concept")},
            )
        )
    return output


def load_bbh(limit: int, seed: int, source: dict[str, Any]) -> list[Candidate]:
    root = (
        f"https://raw.githubusercontent.com/{source['repository']}/{source['version']}/bbh"
    )
    output: list[Candidate] = []
    for task_name in BBH_TASKS:
        payload = get_json(f"{root}/{task_name}.json")
        for index, row in enumerate(payload["examples"]):
            raw_question = str(row["input"])
            target = str(row["target"])
            question, choices = split_bbh_question(raw_question)
            if choices and re.fullmatch(r"\([A-Z]\)", target.strip()):
                task_type = "mcq"
                answer: Any = ord(target.strip("()")) - ord("A")
                subtype = "text_multiple_choice"
            elif task_name in {"multistep_arithmetic_two", "object_counting"}:
                task_type = "math"
                answer = target
                subtype = "mathematical_reasoning"
            else:
                task_type = "text"
                answer = target
                subtype = "short_exact_reasoning"
            difficult_task = task_name in HARD_BBH_TASKS
            score = (8 if difficult_task else 0) + word_count(question) / 15 + len(choices)
            output.append(
                make_candidate(
                    "bbh",
                    source,
                    source_id=f"{task_name}:{index}",
                    source_split=task_name,
                    task_type=task_type,
                    task_subtype=subtype,
                    question=question,
                    answer=answer,
                    choices=choices,
                    hard_score=score,
                    hard_reason=(
                        f"bbh_task={task_name};predefined_hard_task={str(difficult_task).lower()};"
                        f"question_words={word_count(question)}"
                    ),
                    metadata={"category": "text", "bbh_task": task_name},
                )
            )
    random.Random(f"{seed}:bbh-pool").shuffle(output)
    if len(output) < limit:
        raise ValueError(f"BBH yielded {len(output)} candidates; need {limit}")
    return output[:limit]


def load_scienceqa(
    limit: int, seed: int, source: dict[str, Any], image_dir: Path
) -> list[Candidate]:
    from datasets import load_dataset

    image_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(source["repository"], split=source["split"], revision=source["version"])
    dataset = dataset.add_column("_source_index", list(range(len(dataset)))).shuffle(
        seed=stable_int(f"{seed}:scienceqa")
    )
    output: list[Candidate] = []
    for row in dataset:
        image = row.get("image")
        choices = list(row.get("choices") or [])
        if image is None or not choices:
            continue
        index = int(row["_source_index"])
        path = image_dir / f"scienceqa-{index}.png"
        image.convert("RGB").save(path, optimize=True)
        hint = str(row.get("hint") or "").strip()
        question = str(row.get("question") or "").strip()
        if hint:
            question = f"Context: {hint}\nQuestion: {question}"
        grade_number = numeric_count(str(row.get("grade") or ""))
        score = word_count(question) / 12 + len(choices) + grade_number * 0.2
        output.append(
            make_candidate(
                "scienceqa",
                source,
                source_id=f"{source['split']}:{index}",
                task_type="vqa",
                task_subtype="visual_multiple_choice",
                question=question,
                answer=int(row["answer"]),
                choices=choices,
                image_path=str(path),
                image_id=f"scienceqa:{source['split']}:{index}",
                hard_score=score,
                hard_reason=(
                    f"question_context_words={word_count(question)};choice_count={len(choices)};"
                    f"grade={row.get('grade')}"
                ),
                metadata={
                    "category": "vision",
                    "grade": row.get("grade"),
                    "subject": row.get("subject"),
                    "topic": row.get("topic"),
                },
            )
        )
        if len(output) >= limit:
            break
    require_count("ScienceQA", output, limit)
    return output


def load_mmmu(limit: int, seed: int, source: dict[str, Any], image_dir: Path) -> list[Candidate]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    image_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = hf_hub_download(
        repo_id=source["repository"],
        filename="data/validation-00000-of-00001.parquet",
        repo_type="dataset",
        revision=source["version"],
    )
    rows = pq.read_table(parquet_path).to_pylist()
    random.Random(f"{seed}:mmmu-pool").shuffle(rows)
    output: list[Candidate] = []
    for row in rows:
        images = [row.get(f"image_{index}") for index in range(1, 8)]
        present = [value for value in images if value is not None]
        choices = parse_options(row.get("options"))
        answer = str(row.get("answer") or "").strip().upper()
        if len(present) != 1 or not choices or answer not in "ABCDEFG":
            continue
        answer_index = ord(answer) - ord("A")
        if answer_index >= len(choices):
            continue
        source_id = str(row["id"])
        path = image_dir / f"{safe_slug(source_id)}.png"
        save_image_value(present[0], path)
        question = re.sub(
            r"<image[_ ]?\d+>", "the image", str(row.get("question") or ""), flags=re.I
        ).strip()
        difficulty = str(row.get("topic_difficulty") or "").lower()
        score = {"hard": 10, "medium": 5, "easy": 0}.get(difficulty, 2)
        score += word_count(question) / 20 + len(choices)
        output.append(
            make_candidate(
                "mmmu",
                source,
                source_id=source_id,
                task_type="vqa",
                task_subtype="visual_multiple_choice",
                question=question,
                answer=answer_index,
                choices=choices,
                image_path=str(path),
                image_id=f"mmmu:{source_id}",
                hard_score=score,
                hard_reason=(
                    f"topic_difficulty={row.get('topic_difficulty')};question_words={word_count(question)};"
                    f"subfield={row.get('subfield')}"
                ),
                metadata={
                    "category": "vision",
                    "subject": source_id.split("_")[1] if "_" in source_id else "unknown",
                    "subfield": row.get("subfield"),
                    "topic_difficulty": row.get("topic_difficulty"),
                },
            )
        )
        if len(output) >= limit:
            break
    require_count("MMMU", output, limit)
    return output


def load_chartqa(
    limit: int, seed: int, source: dict[str, Any], image_dir: Path
) -> list[Candidate]:
    from datasets import load_dataset

    image_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(source["repository"], split=source["split"], revision=source["version"])
    dataset = dataset.add_column("_source_index", list(range(len(dataset)))).shuffle(
        seed=stable_int(f"{seed}:chartqa")
    )
    output: list[Candidate] = []
    for row in dataset:
        index = int(row["_source_index"])
        source_id = str(row.get("id", index))
        path = image_dir / f"chartqa-{safe_slug(source_id)}.png"
        row["image"].convert("RGB").save(path, optimize=True)
        question = str(row["question"])
        comparison_cues = sum(
            cue in question.lower() for cue in ("difference", "between", "highest", "lowest", "more", "less", "average", "total")
        )
        score = numeric_count(question) * 2 + comparison_cues * 3 + word_count(question) / 15
        output.append(
            make_candidate(
                "chartqa",
                source,
                source_id=f"{source['split']}:{source_id}",
                task_type="vqa",
                task_subtype="open_chart_qa",
                question=question,
                answer=str(row["answer"]),
                image_path=str(path),
                image_id=f"chartqa:{source_id}",
                hard_score=score,
                hard_reason=(
                    f"numeric_mentions={numeric_count(question)};comparison_cues={comparison_cues};"
                    f"question_words={word_count(question)}"
                ),
                metadata={"category": "vision", "chartqa_type": row.get("type")},
            )
        )
        if len(output) >= limit:
            break
    require_count("ChartQA", output, limit)
    return output


def load_ocr_vqa(
    limit: int, seed: int, source: dict[str, Any], image_dir: Path
) -> list[Candidate]:
    from datasets import load_dataset

    image_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        source["repository"], split=source["split"], revision=source["version"], streaming=True
    ).shuffle(seed=stable_int(f"{seed}:ocr-vqa"), buffer_size=10_000)
    output: list[Candidate] = []
    seen_images: set[str] = set()
    for row in dataset:
        image_id = str(row.get("image_id") or "")
        questions = list(row.get("questions") or [])
        answers = list(row.get("answers") or [])
        image = row.get("image")
        if not image_id or image_id in seen_images or image is None or not questions or not answers:
            continue
        question_and_answer = f"{questions[0]} {answers[0]}".lower()
        if any(phrase in question_and_answer for phrase in PROVIDER_BLOCKED_PHRASES):
            continue
        seen_images.add(image_id)
        path = image_dir / f"ocr-vqa-{safe_slug(image_id)}.jpg"
        image.convert("RGB").save(path, quality=88, optimize=True)
        question = str(questions[0])
        ocr_tokens = list(row.get("ocr_tokens") or [])
        score = len(ocr_tokens) / 8 + word_count(question) / 8 + len(str(answers[0])) / 10
        output.append(
            make_candidate(
                "ocr_vqa",
                source,
                source_id=f"{source['split']}:{image_id}:q0",
                task_type="vqa",
                task_subtype="open_ocr_qa",
                question=question,
                answer=str(answers[0]),
                image_path=str(path),
                image_id=f"ocr-vqa:{image_id}",
                hard_score=score,
                hard_reason=(
                    f"ocr_token_count={len(ocr_tokens)};question_words={word_count(question)};"
                    f"answer_chars={len(str(answers[0]))}"
                ),
                metadata={
                    "category": "vision",
                    "ocr_token_count": len(ocr_tokens),
                    "image_width": row.get("image_width"),
                    "image_height": row.get("image_height"),
                    "set_name": row.get("set_name"),
                },
            )
        )
        if len(output) >= limit:
            break
    require_count("OCR-VQA", output, limit)
    return output


def load_bfcl(
    files: Sequence[str], source_key: str, seed: int, source: dict[str, Any]
) -> list[Candidate]:
    root = (
        f"https://raw.githubusercontent.com/{source['repository']}/{source['version']}/"
        "berkeley-function-call-leaderboard/bfcl_eval/data"
    )
    output: list[Candidate] = []
    category = "simple" if source_key == "bfcl_simple" else "multiple"
    for filename in files:
        rows = get_jsonl(f"{root}/{filename}.json")
        answers = {str(row["id"]): row["ground_truth"] for row in get_jsonl(f"{root}/possible_answer/{filename}.json")}
        for index, row in enumerate(rows):
            tools = list(row.get("function") or [])
            question = bfcl_question(row.get("question"))
            answer = canonical_bfcl_answer(answers.get(str(row.get("id"))), tools)
            if word_count(question) < 3 or not tools or answer is None:
                continue
            required = sum(len((tool.get("parameters") or {}).get("required") or []) for tool in tools)
            depth = max((schema_depth(tool.get("parameters") or {}) for tool in tools), default=0)
            similar_names = len(tools) - len({str(tool.get("name", "")).split(".")[-1].split("_")[0] for tool in tools})
            score = len(tools) * 3 + required * 2 + depth + similar_names + word_count(question) / 20
            output.append(
                make_candidate(
                    source_key,
                    source,
                    source_dataset=filename,
                    source_id=f"{filename}:{row.get('id', index)}",
                    source_split=filename,
                    task_type="tool",
                    task_subtype=f"tool_{category}",
                    question=question,
                    answer=answer,
                    tools=tools,
                    hard_score=score,
                    hard_reason=(
                        f"tool_count={len(tools)};required_arg_count={required};"
                        f"schema_depth={depth};similar_name_count={similar_names}"
                    ),
                    tool_category=category,
                    metadata={"category": "tool", "bfcl_source_file": filename},
                )
            )
    random.Random(f"{seed}:{source_key}-pool").shuffle(output)
    return output


def make_candidate(
    source_key: str,
    source: dict[str, Any],
    *,
    source_id: str,
    task_type: str,
    task_subtype: str,
    question: str,
    answer: Any,
    hard_score: float,
    hard_reason: str,
    metadata: dict[str, Any],
    source_dataset: str | None = None,
    source_split: str | None = None,
    choices: list[str] | None = None,
    image_path: str | None = None,
    image_id: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_category: str | None = None,
) -> Candidate:
    canonical = f"v5:{source_key}:{source_id}"
    tool_values = list(tools or [])
    task = TaskExample(
        id=canonical,
        canonical_id=canonical,
        dataset=source_key,
        source_dataset=source_dataset or str(source["repository"]),
        source_version=str(source["version"]),
        source_split=source_split or str(source["split"]),
        source_id=source_id,
        task_type=task_type,
        task_subtype=task_subtype,
        question=question,
        answer=answer,
        answer_reference=answer,
        choices=list(choices or []),
        image_path=image_path,
        image_id=image_id,
        image_reference=(
            f"hf://{source['repository']}@{source['version']}/{source_split or source['split']}/{source_id}"
            if image_id
            else None
        ),
        tools=tool_values,
        tool_category=tool_category,
        tool_schema_id=sha256_json(tool_values) if tool_values else None,
        license_note=str(source["license_note"]),
        metadata={**metadata, "source_repository": source["repository"]},
    )
    return Candidate(source_key, task, float(hard_score), hard_reason)


def allocate_all(
    config: dict[str, Any], pools: dict[str, list[Candidate]], legacy: list[TaskExample]
) -> tuple[list[TaskExample], list[TaskExample], dict[str, Any]]:
    development: list[TaskExample] = []
    test: list[TaskExample] = []
    report: dict[str, Any] = {}
    seed = int(config["selection_seed"])
    legacy_by_dataset: dict[str, list[TaskExample]] = {}
    for task in legacy:
        legacy_by_dataset.setdefault(legacy_dataset_key(task.dataset), []).append(task)

    for source_key, candidates in pools.items():
        candidates = unique_candidates(candidates)
        if source_key.startswith("bfcl_"):
            candidates = [item for item in candidates if word_count(item.task.question) >= 3]
        candidates = [
            item
            for item in candidates
            if not any(tasks_are_near_duplicate(item.task, prior) for prior in test)
        ]
        dev_standard = int(config["development"]["standard"][source_key])
        dev_hard = int(config["development"]["hard"][source_key])
        test_standard = int(config["test"]["standard"][source_key])
        test_hard = int(config["test"]["hard"][source_key])
        required = dev_standard + dev_hard + test_standard + test_hard
        if len(candidates) < required:
            raise ValueError(f"{source_key} has {len(candidates)} unique candidates; need {required}")

        if source_key in {"bbh", "bfcl_simple", "bfcl_multiple"}:
            (
                dev_hard_rows,
                dev_standard_rows,
                test_hard_rows,
                test_standard_rows,
                group_report,
            ) = allocate_group_holdout(
                source_key=source_key,
                candidates=candidates,
                dev_hard=dev_hard,
                dev_standard=dev_standard,
                test_hard=test_hard,
                test_standard=test_standard,
                prior_development=development,
                prior_test=test,
                legacy=legacy_by_dataset.get(legacy_dataset_key(source_key), []),
                seed=seed,
            )
            development.extend(mark_difficulty(item, "hard") for item in dev_hard_rows)
            development.extend(mark_difficulty(item, "standard") for item in dev_standard_rows)
            test.extend(mark_difficulty(item, "hard") for item in test_hard_rows)
            test.extend(mark_difficulty(item, "standard") for item in test_standard_rows)
            report[source_key] = {
                "candidate_pool": len(candidates),
                "development_standard": len(dev_standard_rows),
                "development_hard": len(dev_hard_rows),
                "test_standard": len(test_standard_rows),
                "test_hard": len(test_hard_rows),
                **group_report,
            }
            continue

        hard_order = sorted(
            candidates,
            key=lambda item: (-item.hard_score, stable_int(f"hard:{seed}:{item.task.canonical_id}")),
        )
        dev_hard_rows = hard_order[:dev_hard]
        used = {item.task.canonical_id for item in dev_hard_rows}
        try:
            test_hard_rows = choose_test_rows(
                [item for item in hard_order if item.task.canonical_id not in used],
                test_hard,
                [*development, *(item.task for item in dev_hard_rows)],
                [*legacy_by_dataset.get(legacy_dataset_key(source_key), []), *test],
            )
        except ValueError as exc:
            raise ValueError(f"{source_key} hard allocation failed: {exc}") from exc
        used.update(item.task.canonical_id for item in test_hard_rows)

        standard_pool = [
            item
            for item in candidates
            if item.task.canonical_id not in used
            and not any(tasks_are_near_duplicate(item.task, selected.task) for selected in test_hard_rows)
        ]
        random.Random(f"{seed}:{source_key}:standard").shuffle(standard_pool)
        dev_standard_rows = standard_pool[:dev_standard]
        used.update(item.task.canonical_id for item in dev_standard_rows)
        try:
            test_standard_rows = choose_test_rows(
                [item for item in standard_pool if item.task.canonical_id not in used],
                test_standard,
                [
                    *development,
                    *(item.task for item in dev_hard_rows),
                    *(item.task for item in dev_standard_rows),
                ],
                [*legacy_by_dataset.get(legacy_dataset_key(source_key), []), *test],
            )
        except ValueError as exc:
            raise ValueError(f"{source_key} standard allocation failed: {exc}") from exc

        development.extend(
            mark_difficulty(item, "hard") for item in dev_hard_rows
        )
        development.extend(
            mark_difficulty(item, "standard") for item in dev_standard_rows
        )
        test.extend(mark_difficulty(item, "hard") for item in test_hard_rows)
        test.extend(mark_difficulty(item, "standard") for item in test_standard_rows)
        report[source_key] = {
            "candidate_pool": len(candidates),
            "development_standard": len(dev_standard_rows),
            "development_hard": len(dev_hard_rows),
            "test_standard": len(test_standard_rows),
            "test_hard": len(test_hard_rows),
        }

    random.Random(f"{seed}:development-order").shuffle(development)
    random.Random(f"{seed}:test-order").shuffle(test)
    if len(development) != 3200 or len(test) != 800:
        raise ValueError(f"Unexpected allocation: development={len(development)}, test={len(test)}")
    return development, test, report


def allocate_group_holdout(
    *,
    source_key: str,
    candidates: Sequence[Candidate],
    dev_hard: int,
    dev_standard: int,
    test_hard: int,
    test_standard: int,
    prior_development: Sequence[TaskExample],
    prior_test: Sequence[TaskExample],
    legacy: Sequence[TaskExample],
    seed: int,
) -> tuple[list[Candidate], list[Candidate], list[Candidate], list[Candidate], dict[str, Any]]:
    if source_key.startswith("bfcl_"):
        candidates = annotate_bfcl_template_components(candidates)
    groups: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        group = task_template_group(candidate.task)
        if group is None:
            raise ValueError(f"{source_key} candidate lacks a template group: {candidate.task.canonical_id}")
        groups.setdefault(group, []).append(candidate)

    if source_key == "bbh":
        reserved_names = {
            "boolean_expressions",
            "date_understanding",
            "dyck_languages",
            "navigate",
            "tracking_shuffled_objects_seven_objects",
        }
        test_groups = {f"bbh:{name}" for name in reserved_names}
    else:
        test_groups: set[str] = set()
        reserved_rows = 0
        test_rows = test_hard + test_standard
        target_rows = test_rows + max(40, test_rows // 4)
        for group in sorted(groups, key=lambda value: stable_int(f"group:{seed}:{source_key}:{value}")):
            test_groups.add(group)
            reserved_rows += len(groups[group])
            if reserved_rows >= target_rows:
                break

    missing_groups = test_groups - groups.keys()
    if missing_groups:
        raise ValueError(f"{source_key} missing configured test groups: {sorted(missing_groups)}")

    test_pool = [item for group in test_groups for item in groups[group]]
    development_pool = [
        item
        for group, rows in groups.items()
        if group not in test_groups
        for item in rows
        if not any(tasks_are_near_duplicate(item.task, prior) for prior in prior_test)
    ]
    if len(development_pool) < dev_hard + dev_standard:
        raise ValueError(
            f"{source_key} group holdout leaves {len(development_pool)} development rows; "
            f"need {dev_hard + dev_standard}"
        )

    dev_hard_rows = sorted(
        development_pool,
        key=lambda item: (-item.hard_score, stable_int(f"hard:{seed}:{item.task.canonical_id}")),
    )[:dev_hard]
    used = {item.task.canonical_id for item in dev_hard_rows}
    dev_standard_pool = [item for item in development_pool if item.task.canonical_id not in used]
    random.Random(f"{seed}:{source_key}:grouped-standard").shuffle(dev_standard_pool)
    dev_standard_rows = dev_standard_pool[:dev_standard]
    selected_development = [
        *prior_development,
        *(item.task for item in dev_hard_rows),
        *(item.task for item in dev_standard_rows),
    ]

    test_hard_order = sorted(
        test_pool,
        key=lambda item: (-item.hard_score, stable_int(f"test-hard:{seed}:{item.task.canonical_id}")),
    )
    test_hard_rows = choose_test_rows(
        test_hard_order,
        test_hard,
        selected_development,
        [*legacy, *prior_test],
        deduplicate_output=False,
    )
    used = {item.task.canonical_id for item in test_hard_rows}
    test_standard_pool = [item for item in test_pool if item.task.canonical_id not in used]
    random.Random(f"{seed}:{source_key}:grouped-test-standard").shuffle(test_standard_pool)
    test_standard_rows = choose_test_rows(
        test_standard_pool,
        test_standard,
        selected_development,
        [*legacy, *prior_test],
        deduplicate_output=False,
    )
    return (
        dev_hard_rows,
        dev_standard_rows,
        test_hard_rows,
        test_standard_rows,
        {
            "group_holdout": True,
            "candidate_groups": len(groups),
            "reserved_test_groups": len(test_groups),
            "reserved_test_pool": len(test_pool),
            "test_group_ids": sorted(test_groups),
        },
    )


def choose_test_rows(
    candidates: Sequence[Candidate],
    count: int,
    development: Sequence[TaskExample],
    legacy: Sequence[TaskExample],
    *,
    deduplicate_output: bool = True,
) -> list[Candidate]:
    output: list[Candidate] = []
    comparisons = [*development, *legacy]
    for candidate in candidates:
        if any(tasks_are_near_duplicate(candidate.task, other) for other in comparisons):
            continue
        if deduplicate_output and any(
            tasks_are_near_duplicate(candidate.task, selected.task) for selected in output
        ):
            continue
        output.append(candidate)
        if len(output) == count:
            return output
    raise ValueError(f"Only {len(output)} non-overlapping test candidates available; need {count}")


def mark_difficulty(candidate: Candidate, group: str) -> TaskExample:
    reason = candidate.hard_reason if group == "hard" else "seeded_standard_sample_before_model_inference"
    return replace(
        candidate.task,
        difficulty_group=group,
        difficulty_reason=reason,
        metadata={
            **candidate.task.metadata,
            "difficulty_group": group,
            "difficulty_score": round(candidate.hard_score, 6),
        },
    )


def assign_folds(tasks: list[TaskExample], *, folds: int, seed: int) -> list[TaskExample]:
    from sklearn.model_selection import StratifiedKFold

    strata = [
        f"{task.metadata['category']}|{task.dataset}|{task.difficulty_group}"
        for task in tasks
    ]
    counts = Counter(strata)
    if min(counts.values()) < folds:
        raise ValueError(f"A development stratum has fewer than {folds} samples: {counts}")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_ids = [-1] * len(tasks)
    indices = list(range(len(tasks)))
    for fold_id, (_, validation_indices) in enumerate(splitter.split(indices, strata)):
        for index in validation_indices:
            fold_ids[index] = fold_id
    if any(value < 0 for value in fold_ids):
        raise RuntimeError("Not every development sample received a fold ID")
    return [replace(task, fold_id=fold_ids[index]) for index, task in enumerate(tasks)]


def finalize_tasks(tasks: list[TaskExample], image_dir: Path, seed: int) -> list[TaskExample]:
    image_dir.mkdir(parents=True, exist_ok=True)
    output: list[TaskExample] = []
    for task in tasks:
        image_path = None
        if task.image_path:
            source = Path(task.image_path)
            suffix = source.suffix.lower() or ".png"
            destination = image_dir / f"{safe_slug(task.canonical_id or task.id)}{suffix}"
            shutil.copyfile(source, destination)
            image_path = str(destination)
        updated = replace(task, image_path=image_path, selection_seed=seed)
        prompt_hash = hashlib.sha256(build_prompt(updated).encode("utf-8")).hexdigest()
        output.append(replace(updated, prompt_hash=prompt_hash))
    return output


def build_overlap_report(
    development: list[TaskExample],
    test: list[TaskExample],
    legacy: list[TaskExample],
    config: dict[str, Any],
) -> dict[str, Any]:
    development_fp = {task_fingerprint(task) for task in development}
    test_fp = {task_fingerprint(task) for task in test}
    legacy_fp = {task_fingerprint(task) for task in legacy}
    source_overlap = source_identity_set(development) & source_identity_set(test)
    image_overlap = {task.image_id for task in development if task.image_id} & {
        task.image_id for task in test if task.image_id
    }
    development_image_hashes = {
        sha256_file(Path(task.image_path))
        for task in development
        if task.image_path and Path(task.image_path).exists()
    }
    test_image_hashes = {
        sha256_file(Path(task.image_path))
        for task in test
        if task.image_path and Path(task.image_path).exists()
    }
    development_template_groups = {
        group for task in development if (group := task_template_group(task)) is not None
    }
    test_template_groups = {
        group for task in test if (group := task_template_group(task)) is not None
    }
    template_group_overlap = development_template_groups & test_template_groups
    question_threshold = float(config["near_duplicate"]["question_cosine_threshold"])
    combined_threshold = float(config["near_duplicate"]["combined_cosine_threshold"])
    question_matches = nearest_cross_matches(development, test, combined=False)
    combined_matches = nearest_cross_matches(development, test, combined=True)
    blocking: dict[str, dict[str, Any]] = {}
    for match in question_matches:
        if match["similarity"] >= question_threshold and not match["distinct_visual_pair"]:
            blocking[f"question:{match['test_id']}"] = match
    for match in combined_matches:
        if match["similarity"] >= combined_threshold and not match["distinct_visual_pair"]:
            blocking[f"combined:{match['test_id']}"] = match
    invariant_overlap_count = (
        len(development_fp & test_fp)
        + len(test_fp & legacy_fp)
        + len(source_overlap)
        + len(image_overlap)
        + len(development_image_hashes & test_image_hashes)
        + len(template_group_overlap)
    )
    return {
        "development_examples": len(development),
        "test_examples": len(test),
        "exact_fingerprint_overlap": len(development_fp & test_fp),
        "test_exact_overlap_with_legacy_600": len(test_fp & legacy_fp),
        "source_identity_overlap": len(source_overlap),
        "image_id_overlap": len(image_overlap),
        "image_sha256_overlap": len(development_image_hashes & test_image_hashes),
        "template_group_overlap": len(template_group_overlap),
        "overlapping_template_groups": sorted(template_group_overlap),
        "question_cosine_threshold": question_threshold,
        "combined_cosine_threshold": combined_threshold,
        "blocking_near_duplicate_count": len(blocking),
        "blocking_invariant_overlap_count": invariant_overlap_count,
        "blocking_overlap_count": len(blocking) + invariant_overlap_count,
        "blocking_near_duplicates": list(blocking.values())[:100],
        "highest_question_matches": sorted(question_matches, key=lambda row: -row["similarity"])[:20],
        "highest_combined_matches": sorted(combined_matches, key=lambda row: -row["similarity"])[:20],
        "bfcl_schema_overlap_count": len(
            {task.tool_schema_id for task in development if task.tool_schema_id}
            & {task.tool_schema_id for task in test if task.tool_schema_id}
        ),
    }


def nearest_cross_matches(
    development: Sequence[TaskExample], test: Sequence[TaskExample], *, combined: bool
) -> list[dict[str, Any]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    def value(task: TaskExample) -> str:
        parts = [task.question]
        if combined:
            parts.extend(task.choices)
            if task.tools:
                parts.append(json.dumps(task.tools, ensure_ascii=False, sort_keys=True))
        return "\n".join(parts)

    dev_text = [value(task) for task in development]
    test_text = [value(task) for task in test]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=100_000, sublinear_tf=True
    )
    matrix = vectorizer.fit_transform([*dev_text, *test_text])
    dev_matrix = matrix[: len(development)]
    test_matrix = matrix[len(development) :]
    neighbors = NearestNeighbors(n_neighbors=1, metric="cosine").fit(dev_matrix)
    distances, indices = neighbors.kneighbors(test_matrix)
    return [
        {
            "test_id": test[index].canonical_id,
            "development_id": development[int(indices[index][0])].canonical_id,
            "similarity": round(1.0 - float(distances[index][0]), 6),
            "distinct_visual_pair": images_are_distinct(
                test[index], development[int(indices[index][0])]
            ),
        }
        for index in range(len(test))
    ]


def build_data_report(
    development: list[TaskExample],
    test: list[TaskExample],
    allocation: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": config["version"],
        "selection_seed": config["selection_seed"],
        "fold_seed": config["fold_seed"],
        "allocation": allocation,
        "development": distribution_report(development, include_folds=True),
        "test": distribution_report(test, include_folds=False),
        "source_versions": {
            key: {"repository": value["repository"], "version": value["version"], "split": value["split"]}
            for key, value in config["sources"].items()
        },
        "difficulty_selection": "Intrinsic source metadata and observable complexity only; no Cheap/Strong output was generated before selection.",
        "bfcl_quota_note": (
            "Static BFCL V4 simple_python/multiple contain only 400/200 rows. The official V4 "
            "simple language variants and live_simple/live_multiple files supply the remaining unique rows."
        ),
    }


def distribution_report(tasks: Sequence[TaskExample], *, include_folds: bool) -> dict[str, Any]:
    report: dict[str, Any] = {
        "total": len(tasks),
        "task_family": dict(sorted(Counter(str(task.metadata["category"]) for task in tasks).items())),
        "dataset": dict(sorted(Counter(task.dataset for task in tasks).items())),
        "source_dataset": dict(sorted(Counter(str(task.source_dataset) for task in tasks).items())),
        "difficulty": dict(sorted(Counter(str(task.difficulty_group) for task in tasks).items())),
        "source_split": dict(sorted(Counter(f"{task.source_dataset}:{task.source_split}" for task in tasks).items())),
    }
    if include_folds:
        report["folds"] = {
            str(fold): {
                "total": sum(task.fold_id == fold for task in tasks),
                "task_family": dict(
                    sorted(Counter(str(task.metadata["category"]) for task in tasks if task.fold_id == fold).items())
                ),
                "difficulty": dict(
                    sorted(Counter(str(task.difficulty_group) for task in tasks if task.fold_id == fold).items())
                ),
                "dataset": dict(
                    sorted(Counter(task.dataset for task in tasks if task.fold_id == fold).items())
                ),
            }
            for fold in sorted({int(task.fold_id) for task in tasks if task.fold_id is not None})
        }
    return report


def validate_manifest_fields(tasks: Sequence[TaskExample], *, require_fold: bool) -> None:
    required = (
        "canonical_id",
        "source_dataset",
        "source_version",
        "source_split",
        "source_id",
        "task_subtype",
        "difficulty_group",
        "difficulty_reason",
        "selection_seed",
        "prompt_hash",
        "answer_reference",
        "license_note",
    )
    ids: set[str] = set()
    for task in tasks:
        missing = [name for name in required if getattr(task, name) is None]
        if require_fold and task.fold_id is None:
            missing.append("fold_id")
        if missing:
            raise ValueError(f"{task.id} is missing manifest fields: {missing}")
        if task.canonical_id in ids:
            raise ValueError(f"Duplicate canonical ID: {task.canonical_id}")
        ids.add(str(task.canonical_id))
        if task.requires_vision and (not task.image_path or not Path(task.image_path).exists()):
            raise ValueError(f"Missing image for {task.canonical_id}")


def unique_candidates(candidates: Sequence[Candidate]) -> list[Candidate]:
    output: list[Candidate] = []
    identities: set[tuple[str | None, str | None, str | None]] = set()
    fingerprints: set[str] = set()
    for candidate in candidates:
        task = candidate.task
        identity = (task.source_dataset, task.source_split, task.source_id)
        fingerprint = task_fingerprint(task)
        if identity in identities or fingerprint in fingerprints:
            continue
        identities.add(identity)
        fingerprints.add(fingerprint)
        output.append(candidate)
    return output


def tasks_are_near_duplicate(left: TaskExample, right: TaskExample) -> bool:
    if legacy_dataset_key(left.dataset) != legacy_dataset_key(right.dataset):
        return False
    left_group = task_template_group(left)
    right_group = task_template_group(right)
    if left_group is not None and left_group == right_group:
        return True
    if left.requires_vision and right.requires_vision:
        if left.image_path and right.image_path:
            left_path = Path(left.image_path)
            right_path = Path(right.image_path)
            if left_path.exists() and right_path.exists():
                return sha256_file(left_path) == sha256_file(right_path)
        elif left.image_id and right.image_id and left.image_id != right.image_id:
            return False
    left_question = normalize_text(left.question)
    right_question = normalize_text(right.question)
    if left_question == right_question:
        return True
    ratio = SequenceMatcher(None, left_question, right_question, autojunk=False).ratio()
    if ratio >= 0.90:
        return True
    if left.tools and right.tools and sha256_json(left.tools) == sha256_json(right.tools) and ratio >= 0.75:
        return True
    return False


def task_template_group(task: TaskExample) -> str | None:
    annotated = task.metadata.get("template_group")
    if annotated:
        return str(annotated)
    if task.dataset == "bbh":
        task_name = task.metadata.get("bbh_task")
        return f"bbh:{task_name}" if task_name else None
    if task.dataset not in {"bfcl_simple", "bfcl_multiple"}:
        return None
    generation_group = bfcl_generation_group(task)
    if generation_group:
        return generation_group
    return None


def annotate_bfcl_template_components(
    candidates: Sequence[Candidate],
) -> list[Candidate]:
    parents = list(range(len(candidates)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    first_by_identity: dict[str, int] = {}
    for index, candidate in enumerate(candidates):
        task = candidate.task
        identities = [bfcl_generation_group(task)]
        if task.tool_schema_id:
            identities.append(f"bfcl-multiple:schema:{task.tool_schema_id}")
        for identity in (value for value in identities if value):
            if identity in first_by_identity:
                union(index, first_by_identity[identity])
            else:
                first_by_identity[identity] = index

    component_members: dict[int, list[str]] = {}
    for index, candidate in enumerate(candidates):
        component_members.setdefault(find(index), []).append(
            str(candidate.task.canonical_id)
        )
    component_ids = {
        root: f"{candidates[root].source_key}:component:{sha256_json(sorted(members))[:16]}"
        for root, members in component_members.items()
    }
    return [
        replace(
            candidate,
            task=replace(
                candidate.task,
                metadata={
                    **candidate.task.metadata,
                    "template_group": component_ids[find(index)],
                    "template_group_rule": "connected_component_of_official_generation_group_or_tool_schema",
                },
            ),
        )
        for index, candidate in enumerate(candidates)
    ]


def bfcl_generation_group(task: TaskExample) -> str | None:
    if task.dataset not in {"bfcl_simple", "bfcl_multiple"}:
        return None
    filename = str(task.metadata.get("bfcl_source_file") or task.source_dataset or "")
    source_id = str(task.source_id or "")
    raw_id = source_id.split(":", 1)[-1]
    if filename in {"BFCL_v4_live_simple", "BFCL_v4_live_multiple"}:
        parts = raw_id.rsplit("-", 2)
        if len(parts) == 3:
            return f"{task.dataset}:generation:{filename}:{parts[1]}"
    return f"{task.dataset}:generation:{filename}:{raw_id}"


def images_are_distinct(left: TaskExample, right: TaskExample) -> bool:
    if not left.requires_vision or not right.requires_vision:
        return False
    if left.image_path and right.image_path:
        left_path = Path(left.image_path)
        right_path = Path(right.image_path)
        if left_path.exists() and right_path.exists():
            return sha256_file(left_path) != sha256_file(right_path)
    return bool(left.image_id and right.image_id and left.image_id != right.image_id)


def task_fingerprint(task: TaskExample) -> str:
    image_digest = None
    if task.image_path and Path(task.image_path).exists():
        image_digest = sha256_file(Path(task.image_path))
    return sha256_json(
        {
            "question": normalize_text(task.question),
            "choices": task.choices,
            "tools": task.tools,
            "image_sha256": image_digest,
        }
    )


def source_identity_set(tasks: Sequence[TaskExample]) -> set[tuple[str | None, str | None, str | None]]:
    return {(task.source_dataset, task.source_split, task.source_id) for task in tasks}


def load_legacy_tasks() -> list[TaskExample]:
    paths = [Path("data/manifest.jsonl"), Path("data/v3_test.jsonl"), Path("data/v4_test.jsonl")]
    return [task for path in paths if path.exists() for task in load_jsonl(path)]


def legacy_dataset_key(value: str) -> str:
    aliases = {
        "bbh-logical-deduction": "bbh",
        "scienceqa-img": "scienceqa",
        "ocr-vqa": "ocr_vqa",
        "bfcl-simple_python": "bfcl",
        "bfcl-multiple": "bfcl",
        "bfcl_simple": "bfcl",
        "bfcl_multiple": "bfcl",
    }
    return aliases.get(value, value)


def split_bbh_question(value: str) -> tuple[str, list[str]]:
    if "\nOptions:\n" not in value:
        return value, []
    question, option_block = value.rsplit("\nOptions:\n", 1)
    choices = [
        re.sub(r"^\([A-Z]\)\s*", "", line)
        for line in option_block.splitlines()
        if line.strip()
    ]
    return question, choices


def parse_options(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def save_image_value(value: Any, path: Path) -> None:
    if hasattr(value, "convert"):
        image = value
    elif isinstance(value, dict) and value.get("bytes"):
        from PIL import Image

        image = Image.open(io.BytesIO(value["bytes"]))
    else:
        raise ValueError("Unsupported MMMU image payload")
    image.convert("RGB").save(path, optimize=True)


def bfcl_question(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    for item in walk(value):
        if isinstance(item, dict) and item.get("role") == "user" and item.get("content"):
            return str(item["content"])
    return ""


def walk(value: Iterable[Any]) -> Iterable[Any]:
    for item in value:
        if isinstance(item, list):
            yield from walk(item)
        else:
            yield item


def canonical_bfcl_answer(ground_truth: Any, tools: list[dict[str, Any]]) -> dict[str, Any] | None:
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


def schema_depth(value: Any) -> int:
    if isinstance(value, dict):
        return 1 + max((schema_depth(item) for item in value.values()), default=0)
    if isinstance(value, list):
        return 1 + max((schema_depth(item) for item in value), default=0)
    return 0


def extract_gsm8k_answer(value: str) -> str:
    if "####" in value:
        value = value.rsplit("####", 1)[1]
    values = re.findall(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
    return values[-1] if values else value.strip()


def get_json(url: str) -> dict[str, Any]:
    return json.loads(download(url))


def get_jsonl(url: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in download(url).splitlines() if line.strip()]


def download(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "routerbench-mini-v5"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read().decode("utf-8")


def normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def word_count(value: str) -> int:
    return len(value.split())


def numeric_count(value: str) -> int:
    return len(re.findall(r"-?\d+(?:\.\d+)?", value))


def stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")[:180]


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def require_count(name: str, values: Sequence[Any], expected: int) -> None:
    if len(values) < expected:
        raise ValueError(f"{name} yielded {len(values)} candidates; need {expected}")


if __name__ == "__main__":
    main()
