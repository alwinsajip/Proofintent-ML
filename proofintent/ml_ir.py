"""ML IR: a small, backend-independent intermediate representation of the model.

The ML IR is a linear SSA-style list of operations. It only knows about the
operator set the prototype supports (Linear / Dense / ReLU / Argmax / Softmax)
and says nothing about zero-knowledge or proving. This separation is what lets
the same model be retargeted to a different backend later.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MLOp:
    op: str  # input | dense | linear | relu | argmax | softmax
    output: str
    inputs: list[str] = field(default_factory=list)
    attrs: dict = field(default_factory=dict)

    def render(self) -> str:
        args = ", ".join(self.inputs)
        attr = ""
        if self.attrs:
            inner = ", ".join(f"{k}={v!r}" for k, v in self.attrs.items())
            attr = f" {{{inner}}}"
        if args:
            return f"%{self.output} = {self.op}({args}){attr}"
        return f"%{self.output} = {self.op}(){attr}"


@dataclass
class MLModule:
    name: str
    ops: list[MLOp] = field(default_factory=list)
    inputs: list[tuple[str, int]] = field(default_factory=list)
    outputs: list[tuple[str, int]] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def op_by_output(self, name: str) -> MLOp | None:
        for op in self.ops:
            if op.output == name:
                return op
        return None

    def last_activation(self) -> MLOp | None:
        for op in reversed(self.ops):
            if op.op in ("argmax", "dense", "linear", "relu", "softmax"):
                return op
        return None

    def to_lines(self) -> list[str]:
        lines = [f"ml.module @{self.name} {{"]
        for name, size in self.inputs:
            lines.append(f"  %{name} = input() {{size={size}}}")
        for op in self.ops:
            lines.append("  " + op.render())
        for name, size in self.outputs:
            lines.append(f"  output %{name} {{size={size}}}")
        lines.append("}")
        return lines

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "inputs": [{"name": n, "size": s} for n, s in self.inputs],
            "outputs": [{"name": n, "size": s} for n, s in self.outputs],
            "ops": [
                {"op": o.op, "output": o.output, "inputs": o.inputs, "attrs": o.attrs}
                for o in self.ops
            ],
        }
