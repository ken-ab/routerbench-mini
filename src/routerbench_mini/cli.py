from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from .config import load_costs, load_yaml
from .metrics import prediction_row, summarize_rows, summarize_rows_by_category, write_csv
from .providers import Provider, provider_from_config
from .routers import BaseRouter, default_routers
from .tasks import TaskExample, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RouterBench-Mini evaluation.")
    parser.add_argument("--manifest", default="data/mini_manifest.jsonl")
    parser.add_argument("--models", default="configs/models.mock.yaml")
    parser.add_argument("--costs", default="configs/costs.yaml")
    parser.add_argument("--out", default="results/mock")
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-precompute", action="store_true")
    return parser.parse_args()


def build_providers(model_path: str) -> dict[str, Provider]:
    model_config = load_yaml(model_path).get("providers", {})
    return {
        role: provider_from_config(role, config)
        for role, config in model_config.items()
    }


def precompute_responses(
    tasks: Iterable[TaskExample],
    providers: dict[str, Provider],
    workers: int,
) -> None:
    calls = [(task, provider) for task in tasks for provider in providers.values()]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(provider.generate, task) for task, provider in calls]
        for future in tqdm(as_completed(futures), total=len(futures), desc="model responses"):
            future.result()


def evaluate(
    tasks: list[TaskExample],
    providers: dict[str, Provider],
    routers: list[BaseRouter],
    costs: dict[str, float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for router in routers:
        for task in tqdm(tasks, desc=router.name):
            decision = router.route(task, providers)
            rows.append(prediction_row(task, decision, costs))
    return rows


def write_results(output_dir: str | Path, rows: list[dict[str, object]]) -> None:
    output = Path(output_dir)
    write_csv(output / "predictions.csv", rows)
    write_csv(output / "summary.csv", summarize_rows(rows))
    write_csv(output / "summary_by_category.csv", summarize_rows_by_category(rows))


def main() -> None:
    args = parse_args()
    tasks = load_jsonl(args.manifest)
    costs = load_costs(args.costs)
    providers = build_providers(args.models)
    if not args.no_precompute:
        precompute_responses(tasks, providers, workers=args.workers)
    routers = default_routers(costs, confidence_threshold=args.confidence_threshold)
    rows = evaluate(tasks, providers, routers, costs)
    write_results(args.out, rows)

    summary = summarize_rows(rows)
    for row in summary:
        print(
            f"{row['router']}: accuracy={row['accuracy']:.3f}, "
            f"avg_cost={row['avg_cost']:.6f}, strong_usage={row['strong_usage_rate']:.2f}"
        )


if __name__ == "__main__":
    main()
