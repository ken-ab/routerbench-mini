from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def load_costs(path: str | Path) -> dict[str, float]:
    data = load_yaml(path)
    roles = data.get("roles", {})
    return {str(key): float(value) for key, value in roles.items()}

