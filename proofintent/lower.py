"""Lowering from the checked AST down to the ML IR and the Proof IR."""

from __future__ import annotations

import os

from . import ast_nodes as ast
from .ml_ir import MLModule, MLOp
from .proof_ir import (
    Constraint,
    Intent,
    ProofModule,
    Signal,
    constraint_cost,
)
from .semantic import CheckedSpec


def _basedir(spec: CheckedSpec) -> str:
    path = spec.program.source_path
    if path:
        return os.path.dirname(os.path.abspath(path))
    return os.getcwd()


# ---------------------------------------------------------------------------
# AST -> ML IR
# ---------------------------------------------------------------------------
def lower_to_ml(spec: CheckedSpec) -> MLModule:
    model = spec.model
    ml = MLModule(name=model.name, meta={"basedir": _basedir(spec)})

    input_names: list[str] = []
    for name, decl in spec.inputs.items():
        ml.inputs.append((name, decl.size))
        input_names.append(name)
    for name, decl in spec.outputs.items():
        ml.outputs.append((name, decl.size))

    current = input_names[0] if input_names else "x"

    index = 0
    for layer in model.layers:
        out_name = f"l{index}"
        if layer.kind in ("dense", "linear"):
            attrs = {
                "in": int(layer.params.get("in", 0)),
                "out": int(layer.params.get("out", 0)),
            }
            if "weights" in layer.params:
                attrs["weights"] = layer.params["weights"]
            if "bias" in layer.params:
                attrs["bias"] = layer.params["bias"]
            ml.ops.append(MLOp("dense", out_name, [current], attrs))
        elif layer.kind == "relu":
            size = _size_of(ml, current)
            ml.ops.append(MLOp("relu", out_name, [current], {"n_out": size}))
        elif layer.kind == "argmax":
            size = _size_of(ml, current)
            ml.ops.append(MLOp("argmax", out_name, [current], {"n_out": size}))
        current = out_name
        index += 1

    # If the output is a class but the network does not end in argmax, add one
    # so that the published value really is a class label.
    class_out = None
    for name, decl in spec.outputs.items():
        if decl.kind == "class":
            class_out = decl
            break
    if class_out is not None and (not ml.ops or ml.ops[-1].op != "argmax"):
        size = _size_of(ml, current)
        ml.ops.append(
            MLOp("argmax", f"l{index}", [current], {"n_out": size})
        )
        current = f"l{index}"

    return ml


def _size_of(ml: MLModule, signal: str) -> int:
    for name, size in ml.inputs:
        if name == signal:
            return size
    op = ml.op_by_output(signal)
    if op is not None:
        return int(op.attrs.get("out", op.attrs.get("n_out", 1)))
    return 1


# ---------------------------------------------------------------------------
# ML IR -> Proof IR
# ---------------------------------------------------------------------------
def lower_to_proof(spec: CheckedSpec, ml: MLModule, include_all: bool = True) -> ProofModule:
    module = ProofModule(
        model_name=ml.name,
        metadata={
            "source": spec.program.source_path,
            "basedir": ml.meta.get("basedir"),
        },
    )

    # -- signals -----------------------------------------------------------
    for name, decl in spec.inputs.items():
        module.signals.append(Signal(name, decl.visibility, "input", (decl.size,)))
    for name, decl in spec.outputs.items():
        module.signals.append(Signal(name, decl.visibility, "output", (decl.size,)))

    param_signals: list[str] = []
    for op in ml.ops:
        if op.op == "dense":
            wname = f"{op.output}_W"
            bname = f"{op.output}_b"
            module.signals.append(
                Signal(wname, "public", "param", (op.attrs.get("out", 0), op.attrs.get("in", 0)))
            )
            module.signals.append(Signal(bname, "public", "param", (op.attrs.get("out", 0),)))
            param_signals.extend([wname, bname])

    # -- constraints -------------------------------------------------------
    cid = 0
    signal_of: dict[str, str] = {}
    const_signals: set[str] = set()
    logits_signal: str | None = None

    for op in ml.ops:
        if op.op == "dense":
            wname, bname = f"{op.output}_W", f"{op.output}_b"
            module.signals.append(
                Signal(op.output, "private", "intermediate", (op.attrs.get("out", 0),))
            )
            inputs = [op.inputs[0], wname, bname]
            module.constraints.append(
                Constraint(
                    cid,
                    "dense",
                    op.output,
                    inputs,
                    constraint_cost("dense", {"n_out": op.attrs.get("out", 0)}),
                    "inference",
                    {"n_out": op.attrs.get("out", 0), "n_in": op.attrs.get("in", 0)},
                )
            )
            cid += 1
            logits_signal = op.output
            signal_of[op.output] = op.output
        elif op.op == "relu":
            module.signals.append(
                Signal(op.output, "private", "intermediate", (op.attrs.get("n_out", 0),))
            )
            module.constraints.append(
                Constraint(
                    cid,
                    "relu",
                    op.output,
                    [op.inputs[0]],
                    constraint_cost("relu", {"n_out": op.attrs.get("n_out", 0)}),
                    "inference",
                    {"n_out": op.attrs.get("n_out", 0)},
                )
            )
            cid += 1
        elif op.op == "argmax":
            module.signals.append(
                Signal(op.output, "private", "intermediate", (1,))
            )
            module.constraints.append(
                Constraint(
                    cid,
                    "argmax",
                    op.output,
                    [op.inputs[0]],
                    constraint_cost("argmax", op.attrs),
                    "inference",
                    dict(op.attrs),
                )
            )
            cid += 1
        elif op.op == "const":
            module.signals.append(
                Signal(op.output, "public", "const", (1,))
            )
            const_signals.add(op.output)
        elif op.op == "softmax":
            pass

    final_signal = ml.ops[-1].output if ml.ops else (next(iter(spec.inputs), "x"))

    # Alias the final value to the declared output variable (zero cost).
    out_var = None
    if spec.inference is not None:
        out_var = spec.inference.output_name
    elif spec.assignments:
        out_var = spec.assignments[0].target
    if out_var is None and spec.outputs:
        out_var = next(iter(spec.outputs))
    if out_var is None:
        out_var = "y"

    module.constraints.append(
        Constraint(cid, "alias", out_var, [final_signal], 0, "inference", {})
    )
    cid += 1

    # Confidence head (softmax over the logits). Included in the full circuit
    # regardless of whether the intent asks for it; dead-constraint
    # elimination removes it when unneeded.
    class_out = next(
        (d for d in spec.outputs.values() if d.kind == "class"), None
    )
    if include_all and class_out is not None and logits_signal is not None:
        n_classes = class_out.size
        module.signals.append(
            Signal("conf", "private", "intermediate", (n_classes,))
        )
        module.constraints.append(
            Constraint(
                cid,
                "softmax",
                "conf",
                [logits_signal],
                constraint_cost("softmax", {"n_out": n_classes}),
                "confidence",
                {"n_out": n_classes},
            )
        )
        cid += 1

    # Input range checks (full circuit always emits them).
    if include_all:
        for name, decl in spec.inputs.items():
            if decl.range_low is None or decl.range_high is None:
                continue
            module.signals.append(
                Signal(f"{name}_range_ok", "public", "intermediate", (1,))
            )
            module.constraints.append(
                Constraint(
                    cid,
                    "range",
                    f"{name}_range_ok",
                    [name],
                    constraint_cost("range", {"n_elems": decl.size}),
                    "range",
                    {"low": decl.range_low, "high": decl.range_high, "n_elems": decl.size},
                )
            )
            cid += 1

    # Model commitment (binds the committed parameters).
    module.signals.append(
        Signal("model_commit", "public", "intermediate", (1,))
    )
    module.constraints.append(
        Constraint(
            cid,
            "model_commit",
            "model_commit",
            param_signals,
            constraint_cost("model_commit", {}),
            "model_is",
            {"n_params": len(param_signals)},
        )
    )
    cid += 1

    # -- intents -----------------------------------------------------------
    module.intents.append(
        Intent("model_is", {"model": spec.model_is.name if spec.model_is else ml.name})
    )
    module.intents.append(
        Intent("inference", {"input": _primary_input(spec), "output": out_var})
    )
    if spec.confidence is not None:
        module.intents.append(
            Intent(
                "confidence",
                {
                    "input": spec.confidence.input_name,
                    "op": spec.confidence.op,
                    "threshold": spec.confidence.threshold,
                },
            )
        )
    for name, decl in spec.inputs.items():
        if decl.range_low is not None and decl.range_high is not None:
            module.intents.append(
                Intent(
                    "range",
                    {
                        "input": name,
                        "low": decl.range_low,
                        "high": decl.range_high,
                    },
                )
            )

    return module


def _primary_input(spec: CheckedSpec) -> str:
    if spec.inference is not None:
        return spec.inference.input_name
    return next(iter(spec.inputs), "x")
