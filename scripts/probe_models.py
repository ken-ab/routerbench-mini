from __future__ import annotations

import argparse

from routerbench_mini.cli import build_providers
from routerbench_mini.tasks import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe configured models on one task per category.")
    parser.add_argument("--manifest", default="data/validation.jsonl")
    parser.add_argument("--models", default="configs/models.qwen_api.yaml")
    args = parser.parse_args()

    tasks = load_jsonl(args.manifest)
    samples = [
        next(task for task in tasks if task.metadata.get("category") == category)
        for category in ("text", "vision", "tool")
    ]
    providers = build_providers(args.models)
    for role, provider in providers.items():
        for task in samples:
            response = provider.generate(task)
            usage = response.metadata.get("usage", {})
            print(
                f"{role} {task.metadata['category']} {task.id}: "
                f"answer={response.answer[:100]!r} confidence={response.confidence:.2f} "
                f"tokens={usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)}"
            )


if __name__ == "__main__":
    main()
