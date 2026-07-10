from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from .config import load_costs, load_yaml
from .metrics import prediction_row, summarize_rows, write_csv
from .providers import provider_from_config
from .routers import default_routers
from .tasks import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RouterBench-Mini evaluation.")
    parser.add_argument("--manifest", default="data/mini_manifest.jsonl", help="Task manifest JSONL path.")
    parser.add_argument("--models", default="configs/models.mock.yaml", help="Model provider YAML path.")
    parser.add_argument("--costs", default="configs/costs.yaml", help="Relative cost YAML path.")
    parser.add_argument("--out", default="results/mock", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_jsonl(args.manifest)
    costs = load_costs(args.costs)
    model_config = load_yaml(args.models).get("providers", {})
    providers = {
        role: provider_from_config(role, config)
        for role, config in model_config.items()
    }

    rows = []
    routers = default_routers(costs)
    for router in routers:
        for task in tqdm(tasks, desc=router.name):
            decision = router.route(task, providers)
            rows.append(prediction_row(task, decision, costs))

    output_dir = Path(args.out)
    write_csv(output_dir / "predictions.csv", rows)
    summary = summarize_rows(rows)
    write_csv(output_dir / "summary.csv", summary)

    print(f"Wrote {output_dir / 'predictions.csv'}")
    print(f"Wrote {output_dir / 'summary.csv'}")
    for row in summary:
        print(
            f"{row['router']}: accuracy={row['accuracy']:.3f}, "
            f"avg_cost={row['avg_cost']:.2f}, escalation={row['escalation_rate']:.2f}"
        )


if __name__ == "__main__":
    main()

