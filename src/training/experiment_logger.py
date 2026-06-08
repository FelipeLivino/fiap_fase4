"""Small persistence helpers for experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_path


def save_history(history: list[dict[str, Any]], path: str | Path) -> Path:
    output_path = Path(path)
    ensure_parent_dir(output_path)
    pd.DataFrame(history).to_csv(output_path, index=False)
    return output_path


def append_history_row(history: list[dict[str, Any]], row: dict[str, Any]) -> list[dict[str, Any]]:
    history.append(row)
    return history
