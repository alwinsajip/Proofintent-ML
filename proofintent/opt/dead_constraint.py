"""Intent-driven dead-constraint elimination.

Starting from the proof intents the developer declared, we walk *backwards*
through the constraint dependency graph and keep only the constraints that the
declared claims actually require. Everything else (for example the softmax head
when no confidence bound was requested, or an input range check when no range
was declared) is removed.

This is the central optimization of ProofIntent-ML.
"""

from __future__ import annotations

from ..proof_ir import ProofModule, Signal
from . import PassResult


def _intent_targets(module: ProofModule) -> set[str]:
    targets: set[str] = set()
    for intent in module.intents:
        if intent.kind == "model_is":
            targets.add("model_commit")
        elif intent.kind == "inference":
            out = intent.detail.get("output")
            if out:
                targets.add(out)
        elif intent.kind == "confidence":
            targets.add("conf")
        elif intent.kind == "range":
            name = intent.detail.get("input")
            if name:
                targets.add(f"{name}_range_ok")
    return targets


def eliminate_dead_constraints(module: ProofModule) -> tuple[ProofModule, PassResult]:
    before_rows = module.constraint_rows()
    before_cost = module.total_cost()

    targets = _intent_targets(module)
    producer = {c.out: c for c in module.constraints}

    needed = set(targets)
    kept: set[int] = set()
    changed = True
    while changed:
        changed = False
        for sig in list(needed):
            c = producer.get(sig)
            if c is not None and c.cid not in kept:
                kept.add(c.cid)
                needed.update(c.inputs)
                changed = True

    new_constraints = [c for c in module.constraints if c.cid in kept]

    referenced: set[str] = set()
    for c in new_constraints:
        referenced.update(c.inputs)
        referenced.add(c.out)

    keep_kinds = {"input", "output", "param", "const"}
    new_signals: list[Signal] = []
    for s in module.signals:
        if s.name in referenced or s.kind in keep_kinds:
            new_signals.append(s)

    result = ProofModule(
        model_name=module.model_name,
        signals=new_signals,
        constraints=new_constraints,
        intents=[i for i in module.intents],
        metadata=dict(module.metadata),
    )
    stats = PassResult(
        name="dead_constraint_elimination",
        before_rows=before_rows,
        after_rows=result.constraint_rows(),
        before_cost=before_cost,
        after_cost=result.total_cost(),
        details={
            "intents": [i.kind for i in module.intents],
            "removed_rows": before_rows - result.constraint_rows(),
        },
    )
    return result, stats
