"""Loading of committed model parameters (weights and biases).

Parameters are stored as JSON so that examples need no binary dependencies.
Two layouts are accepted::

    {"shape": [out, in], "data": [[...], ...]}   # or flattened data
    [[...], [...]]                                # bare nested list
"""

from __future__ import annotations

import json
import os
from typing import Any


def _flatten(seq: Any) -> list[float]:
    out: list[float] = []
    stack = [seq]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(reversed(item))
        else:
            out.append(float(item))
    return out


def load_array(path: str) -> tuple[list[float], tuple[int, ...]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"parameter file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        obj = json.load(fh)

    if isinstance(obj, dict):
        data = obj.get("data", [])
        shape = tuple(obj.get("shape", ()))
        flat = _flatten(data)
        if not shape:
            shape = (len(flat),)
        return flat, shape

    flat = _flatten(obj)
    return flat, (len(flat),)


def load_matrix(path: str, rows: int, cols: int) -> list[list[float]]:
    flat, shape = load_array(path)
    if len(flat) != rows * cols:
        raise ValueError(
            f"{path}: expected {rows * cols} values but found {len(flat)}"
        )
    return [flat[r * cols:(r + 1) * cols] for r in range(rows)]


def load_vector(path: str, size: int) -> list[float]:
    flat, _ = load_array(path)
    if len(flat) != size:
        raise ValueError(f"{path}: expected {size} values but found {len(flat)}")
    return flat
