"""Reference (non-ZK) evaluation of the ML IR.

This is the *semantic oracle*: it computes what the network actually does, and
the witness generator uses it to fill in every signal of the circuit.
"""

from __future__ import annotations

import math
import os

from ..ml_ir import MLModule
from ..weights import load_matrix, load_vector


def _resolve(basedir: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(basedir, path)


def forward(ml: MLModule, x: list[float]) -> dict:
    """Evaluate the network and return every signal value by name."""
    basedir = ml.meta.get("basedir", os.getcwd())
    values: dict[str, object] = {}
    if ml.inputs:
        values[ml.inputs[0][0]] = list(x)

    for op in ml.ops:
        if op.op == "dense":
            n_in = int(op.attrs.get("in", 0))
            n_out = int(op.attrs.get("out", 0))
            src = values[op.inputs[0]]
            if op.attrs.get("weights"):
                W = load_matrix(_resolve(basedir, op.attrs["weights"]), n_out, n_in)
            else:
                W = [[1.0 if r == c else 0.0 for c in range(n_in)] for r in range(n_out)]
            if op.attrs.get("bias"):
                b = load_vector(_resolve(basedir, op.attrs["bias"]), n_out)
            else:
                b = [0.0] * n_out
            y = [
                sum(W[r][c] * src[c] for c in range(n_in)) + b[r]
                for r in range(n_out)
            ]
            values[op.output] = y
        elif op.op == "relu":
            src = values[op.inputs[0]]
            values[op.output] = [max(0.0, float(v)) for v in src]
        elif op.op == "argmax":
            src = values[op.inputs[0]]
            values[op.output] = int(max(range(len(src)), key=lambda i: src[i]))
        elif op.op == "const":
            values[op.output] = op.attrs.get("value", 0.0)
    return values


def logits_signal(ml: MLModule) -> str | None:
    for op in ml.ops:
        if op.op == "argmax":
            return op.inputs[0]
    for op in reversed(ml.ops):
        if op.op in ("dense", "relu"):
            return op.output
    return None


def softmax(vec: list[float]) -> list[float]:
    if not vec:
        return []
    m = max(vec)
    exps = [math.exp(v - m) for v in vec]
    s = sum(exps)
    return [e / s for e in exps]


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def quantize(vec, bits: int = 16, frac_bits: int = 8):
    scale = 1 << frac_bits
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    out = []
    for v in vec:
        q = int(round(float(v) * scale))
        q = max(lo, min(hi, q))
        out.append(q)
    return out


def dequantize(vec, frac_bits: int = 8):
    scale = 1 << frac_bits
    return [float(v) / scale for v in vec]
