"""Witness generation and constraint evaluation.

The witness assigns a concrete value to every signal of the optimized circuit
using the reference semantics. The same routine evaluates each constraint so
that the prover can attest that the circuit is satisfied.
"""

from __future__ import annotations

import hashlib
import os

from ..compiler import CompilationResult
from ..ml_ir import MLModule
from ..weights import load_matrix, load_vector
from .inference import forward, logits_signal, softmax


def _resolve(basedir: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(basedir, path)


def parameter_files(ml: MLModule) -> list[str]:
    basedir = ml.meta.get("basedir", os.getcwd())
    files: list[str] = []
    for op in ml.ops:
        if op.op == "dense":
            for key in ("weights", "bias"):
                p = op.attrs.get(key)
                if p:
                    files.append(_resolve(basedir, p))
    return files


def model_commitment(ml: MLModule) -> str:
    """A SHA-256 commitment over the concatenated parameter files."""
    h = hashlib.sha256()
    for path in parameter_files(ml):
        h.update(os.path.basename(path).encode("utf-8"))
        with open(path, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def build_witness(result: CompilationResult, x: list[float]):
    ml = result.ml
    basedir = ml.meta.get("basedir", os.getcwd())
    values = forward(ml, x)

    witness: dict[str, object] = {}
    for name, _size in ml.inputs:
        witness[name] = values.get(name)

    # parameters
    for op in ml.ops:
        if op.op == "dense":
            n_in = int(op.attrs.get("in", 0))
            n_out = int(op.attrs.get("out", 0))
            if op.attrs.get("weights"):
                witness[f"{op.output}_W"] = load_matrix(
                    _resolve(basedir, op.attrs["weights"]), n_out, n_in
                )
            if op.attrs.get("bias"):
                witness[f"{op.output}_b"] = load_vector(
                    _resolve(basedir, op.attrs["bias"]), n_out
                )

    # intermediates
    for op in ml.ops:
        if op.output in values:
            witness[op.output] = values[op.output]

    # bind the declared output variable to the final network value
    out_var = next(
        (i.detail.get("output") for i in result.optimized.intents if i.kind == "inference"),
        None,
    )
    if out_var is not None:
        final = ml.ops[-1].output if ml.ops else out_var
        witness[out_var] = values.get(final, values.get(out_var))

    # confidence head
    logits_name = logits_signal(ml)
    if logits_name is not None and logits_name in values:
        witness["conf"] = softmax(list(values[logits_name]))

    # range checks
    for name, _size in ml.inputs:
        decl = result.spec.inputs.get(name)
        if decl and decl.range_low is not None and decl.range_high is not None:
            arr = values.get(name, [])
            ok = all(decl.range_low <= float(v) <= decl.range_high for v in arr)
            witness[f"{name}_range_ok"] = 1 if ok else 0

    witness["model_commit"] = model_commitment(ml)

    checks = evaluate_constraints(result, witness)
    return witness, checks


def _approx(a, b, tol: float = 1e-6) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(abs(float(u) - float(v)) <= tol for u, v in zip(a, b))
    if isinstance(a, str) or isinstance(b, str):
        return str(a) == str(b)
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return a == b


def evaluate_constraints(result: CompilationResult, witness: dict):
    checks = []
    for c in result.optimized.constraints:
        ok = True
        detail = c.kind
        try:
            if c.kind == "dense":
                W = witness.get(f"{c.out}_W", [])
                b = witness.get(f"{c.out}_b", [])
                src = witness.get(c.inputs[0], [])
                exp = [
                    sum(W[r][col] * src[col] for col in range(len(src))) + b[r]
                    for r in range(len(b))
                ]
                ok = _approx(exp, witness.get(c.out))
            elif c.kind == "relu":
                src = witness.get(c.inputs[0], [])
                exp = [max(0.0, float(v)) for v in src]
                ok = _approx(exp, witness.get(c.out))
            elif c.kind == "argmax":
                src = witness.get(c.inputs[0], [])
                if src:
                    exp = int(max(range(len(src)), key=lambda i: src[i]))
                    ok = int(witness.get(c.out, -1)) == exp
            elif c.kind == "softmax":
                src = witness.get(c.inputs[0], [])
                ok = _approx(softmax(list(src)), witness.get("conf"))
            elif c.kind == "alias":
                ok = _approx(witness.get(c.inputs[0]), witness.get(c.out))
            elif c.kind == "range":
                ok = witness.get(c.out, 0) in (1, True)
            elif c.kind == "model_commit":
                ok = bool(witness.get("model_commit"))
        except Exception as exc:  # pragma: no cover - defensive
            ok = False
            detail = f"{c.kind}:{exc}"
        checks.append({"cid": c.cid, "kind": detail, "ok": bool(ok)})
    return checks
