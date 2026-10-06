"""Proof IR: the representation that the backend actually consumes.

The Proof IR carries, for a given model:

* the set of signals, each tagged ``public`` or ``private`` (this is where the
  privacy analysis becomes concrete);
* the constraint system that the circuit must satisfy;
* the proof intents that the developer declared, each linked to the
  constraints that implement it.

Constraint cost is expressed in R1CS constraints using a documented model so
that baseline and optimized circuits can be compared honestly.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Constraint cost model (R1CS constraints per logical constraint).
#
#   dense / linear : n_out          one linear combination per output neuron
#   relu           : n_out          one booleanization + range check per unit
#   argmax         : 2 * (n - 1)    comparison network over n logits
#   softmax        : n_out          one division/exp approx per class
#   range          : n_elems        one bounded-integer decomposition per element
#   model_commit   : 1              a single hash/commitment check
# ---------------------------------------------------------------------------
def constraint_cost(kind: str, attrs: dict) -> int:
    if kind in ("dense", "linear", "relu", "softmax"):
        return int(attrs.get("n_out", 1))
    if kind == "argmax":
        n = int(attrs.get("n_out", 1))
        return 2 * max(0, n - 1)
    if kind == "range":
        return int(attrs.get("n_elems", 1))
    if kind == "model_commit":
        return 1
    if kind == "comparison":
        return 1
    return 1


@dataclass
class Signal:
    name: str
    visibility: str  # public | private
    kind: str  # input | param | intermediate | output | const
    shape: tuple = ()
    dtype: str = "field"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "visibility": self.visibility,
            "kind": self.kind,
            "shape": list(self.shape),
            "dtype": self.dtype,
        }


@dataclass
class Constraint:
    cid: int
    kind: str
    out: str
    inputs: list[str]
    cost: int
    intent: str  # model_is | inference | confidence | range
    meta: dict = field(default_factory=dict)

    def describe(self) -> str:
        args = ", ".join(self.inputs)
        return (
            f"c{self.cid}: {self.kind}({args}) -> %{self.out} "
            f"[cost={self.cost}, intent={self.intent}]"
        )

    def to_dict(self) -> dict:
        return {
            "cid": self.cid,
            "kind": self.kind,
            "out": self.out,
            "inputs": self.inputs,
            "cost": self.cost,
            "intent": self.intent,
            "meta": self.meta,
        }


@dataclass
class Intent:
    kind: str  # model_is | inference | confidence | range
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "detail": self.detail}


@dataclass
class ProofModule:
    model_name: str
    signals: list[Signal] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    intents: list[Intent] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # -- inspection --------------------------------------------------------
    def signal(self, name: str) -> Signal | None:
        for s in self.signals:
            if s.name == name:
                return s
        return None

    def public_signals(self) -> list[Signal]:
        return [s for s in self.signals if s.visibility == "public"]

    def private_signals(self) -> list[Signal]:
        return [s for s in self.signals if s.visibility == "private"]

    def total_cost(self) -> int:
        return sum(c.cost for c in self.constraints)

    def constraint_rows(self) -> int:
        return len(self.constraints)

    def copy(self) -> "ProofModule":
        return ProofModule(
            model_name=self.model_name,
            signals=[Signal(**vars(s)) for s in self.signals],
            constraints=[
                Constraint(
                    cid=c.cid,
                    kind=c.kind,
                    out=c.out,
                    inputs=list(c.inputs),
                    cost=c.cost,
                    intent=c.intent,
                    meta=dict(c.meta),
                )
                for c in self.constraints
            ],
            intents=[Intent(kind=i.kind, detail=dict(i.detail)) for i in self.intents],
            metadata=dict(self.metadata),
        )

    # -- rendering ---------------------------------------------------------
    def to_lines(self) -> list[str]:
        lines = [f"proof.module @{self.model_name} {{"]
        lines.append("  signals:")
        for s in self.signals:
            lines.append(f"    %{s.name} : {s.visibility} {s.kind} {list(s.shape)}")
        lines.append("  intents:")
        for i in self.intents:
            lines.append(f"    {i.kind} {i.detail}")
        lines.append("  constraints:")
        for c in self.constraints:
            lines.append("    " + c.describe())
        lines.append(
            f"  total: {self.constraint_rows()} constraints, "
            f"r1cs cost {self.total_cost()}"
        )
        lines.append("}")
        return lines

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "signals": [s.to_dict() for s in self.signals],
            "constraints": [c.to_dict() for c in self.constraints],
            "intents": [i.to_dict() for i in self.intents],
            "metrics": {
                "constraint_rows": self.constraint_rows(),
                "r1cs_cost": self.total_cost(),
            },
            "metadata": self.metadata,
        }


def summarize(module: ProofModule) -> dict:
    return {
        "constraint_rows": module.constraint_rows(),
        "r1cs_cost": module.total_cost(),
        "public_signals": len(module.public_signals()),
        "private_signals": len(module.private_signals()),
    }
