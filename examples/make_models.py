"""Generate deterministic example parameters and sample inputs.

Run from the repository root:

    python examples/make_models.py

Creates ``examples/weights/<model>/w*.json`` and ``examples/inputs/<model>.json``.
All values are produced from a fixed seed so results are reproducible.
"""

from __future__ import annotations

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))


def write_matrix(path: str, rows: int, cols: int, rng: random.Random, scale: float = 0.5):
    data = [[round(rng.uniform(-scale, scale), 6) for _ in range(cols)] for _ in range(rows)]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"shape": [rows, cols], "data": data}, fh)


def write_vector(path: str, size: int, rng: random.Random, scale: float = 0.2):
    data = [round(rng.uniform(-scale, scale), 6) for _ in range(size)]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"shape": [size], "data": data}, fh)


def write_sample_input(path: str, size: int, rng: random.Random):
    data = [round(rng.uniform(-1.0, 1.0), 6) for _ in range(size)]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)


def make_small():
    rng = random.Random(20240001)
    base = os.path.join(HERE, "weights", "mlp_small")
    write_matrix(os.path.join(base, "w1.json"), 8, 4, rng)
    write_vector(os.path.join(base, "b1.json"), 8, rng)
    write_matrix(os.path.join(base, "w2.json"), 3, 8, rng)
    write_vector(os.path.join(base, "b2.json"), 3, rng)
    write_sample_input(os.path.join(HERE, "inputs", "mlp_small.json"), 4, rng)


def make_large():
    rng = random.Random(20240002)
    base = os.path.join(HERE, "weights", "mlp_large")
    write_matrix(os.path.join(base, "w1.json"), 16, 8, rng)
    write_vector(os.path.join(base, "b1.json"), 16, rng)
    # Pruned layer: all-zero weights and bias, so constant propagation can
    # collapse it (and the following ReLU) entirely.
    write_matrix(os.path.join(base, "w2.json"), 16, 16, rng, scale=0.0)
    write_vector(os.path.join(base, "b2.json"), 16, rng, scale=0.0)
    write_matrix(os.path.join(base, "w3.json"), 10, 16, rng)
    write_vector(os.path.join(base, "b3.json"), 10, rng)
    write_sample_input(os.path.join(HERE, "inputs", "mlp_large.json"), 8, rng)


def main():
    make_small()
    make_large()
    print("wrote example parameters under", os.path.join(HERE, "weights"))
    print("wrote sample inputs under", os.path.join(HERE, "inputs"))


if __name__ == "__main__":
    main()
