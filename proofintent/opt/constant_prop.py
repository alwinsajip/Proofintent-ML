"""Constant propagation over the ML IR.

Weights and biases are committed constants that are known at compile time in
this prototype (they are files on disk). A dense layer whose weights *and*
bias are all zero produces a constant zero vector, so it needs no
multiplication constraints at all. That constant then propagates through
subsequent ReLU and Argmax operators, which may fold away in turn.

The pass returns a new :class:`MLModule` and a dictionary of statistics.
"""

from __future__ import annotations

import os

from ..ml_ir import MLModule, MLOp
from ..weights import load_vector
from . import PassResult


def _is_zero_layer(op: MLOp, basedir: str) -> bool:
    w = op.attrs.get("weights")
    b = op.attrs.get("bias")
    if not w:
        return False
    wpath = os.path.join(basedir, w) if not os.path.isabs(w) else w
    if not os.path.exists(wpath):
        return False
    try:
        flat, _ = _load_flat(wpath)
    except Exception:
        return False
    if any(abs(v) > 1e-12 for v in flat):
        return False
    if b:
        bpath = os.path.join(basedir, b) if not os.path.isabs(b) else b
        if os.path.exists(bpath):
            bias = load_vector(bpath, int(op.attrs.get("out", len(flat))))
            if any(abs(v) > 1e-12 for v in bias):
                return False
        else:
            return False
    else:
        return False
    return True


def _load_flat(path: str):
    from ..weights import load_array

    return load_array(path)


def constant_prop(ml: MLModule) -> tuple[MLModule, PassResult]:
    basedir = ml.meta.get("basedir", os.getcwd())
    before_ops = len(ml.ops)

    const_value: dict[str, object] = {}
    new_ops: list[MLOp] = []
    folded = 0

    for op in ml.ops:
        if op.op == "const":
            new_ops.append(op)
            const_value[op.output] = op.attrs.get("value", 0.0)
            continue

        if op.op == "dense":
            if _is_zero_layer(op, basedir):
                new_ops.append(MLOp("const", op.output, [], {"value": 0.0, "source": "zero_dense"}))
                const_value[op.output] = 0.0
                folded += 1
            else:
                new_ops.append(op)
                const_value.pop(op.output, None)
            continue

        if op.op == "relu":
            src = op.inputs[0]
            if src in const_value and const_value[src] == 0.0:
                new_ops.append(MLOp("const", op.output, [], {"value": 0.0, "source": "relu_zero"}))
                const_value[op.output] = 0.0
                folded += 1
            else:
                new_ops.append(op)
                const_value.pop(op.output, None)
            continue

        if op.op == "argmax":
            src = op.inputs[0]
            if src in const_value:
                new_ops.append(MLOp("const", op.output, [], {"value": 0, "source": "argmax_const"}))
                const_value[op.output] = 0
                folded += 1
            else:
                new_ops.append(op)
                const_value.pop(op.output, None)
            continue

        new_ops.append(op)

    result = MLModule(
        name=ml.name,
        ops=new_ops,
        inputs=list(ml.inputs),
        outputs=list(ml.outputs),
        meta=dict(ml.meta),
    )
    stats = PassResult(
        name="constant_propagation",
        before_rows=before_ops,
        after_rows=len(new_ops),
        before_cost=0,
        after_cost=0,
        details={"folded_ops": folded},
    )
    return result, stats
