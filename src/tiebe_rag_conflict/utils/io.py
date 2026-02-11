from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


__all__ = [
    "read_jsonl",
    "write_jsonl",
    "read_config",
    "ensure_dir",
    "set_seed",
]


def read_jsonl(path: str | os.PathLike) -> list[dict[str, Any]]:
    """Read a JSON Lines file into a list of dictionaries."""
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: str | os.PathLike, records: Iterable[dict[str, Any]]) -> None:
    """Write an iterable of dictionaries to a JSON Lines file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def read_config(path: str | os.PathLike) -> dict[str, Any]:
    """Load a YAML configuration file into a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

