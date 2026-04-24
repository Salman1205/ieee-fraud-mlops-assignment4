import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dump_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
