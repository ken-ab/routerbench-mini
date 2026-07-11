from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from routerbench_mini.metrics import summarize_rows, write_csv

from run_v3_study import add_bootstrap_intervals, paired_bootstrap_comparisons


FROZEN_METHODS = {"always_cheap", "always_strong", "task_aware"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the V3 and V4 held-out replications.")
    parser.add_argument("--v3", default="results/qwen3.5-v3-study/test_predictions.csv")
    parser.add_argument("--v4", default="results/qwen3.5-v4-study/test_predictions.csv")
    parser.add_argument("--out", default="results/qwen3.5-confirmatory")
    args = parser.parse_args()

    replications: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for name, path in (("V3", args.v3), ("V4", args.v4)):
        rows = list(csv.DictReader(Path(path).open(encoding="utf-8")))
        all_rows.extend({**row, "replication": name} for row in rows)
        for summary in summarize_rows(rows):
            replications.append({"replication": name, **summary})

    frozen_rows = [row for row in all_rows if str(row["router"]) in FROZEN_METHODS]
    pooled = add_bootstrap_intervals(frozen_rows, summarize_rows(frozen_rows))
    output_dir = Path(args.out)
    write_csv(output_dir / "replication_summary.csv", replications)
    write_csv(output_dir / "pooled_frozen_summary.csv", pooled)
    write_csv(output_dir / "pooled_paired_comparisons.csv", paired_bootstrap_comparisons(frozen_rows))


if __name__ == "__main__":
    main()
