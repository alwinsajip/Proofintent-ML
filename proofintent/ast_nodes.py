"""Abstract syntax tree for the ProofIntent-ML DSL."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Node:
    line: int = 0
    col: int = 0


# ---------------------------------------------------------------------------
# Expressions used on the right hand side of prove assignments.
# ---------------------------------------------------------------------------
@dataclass
class Expr(Node):
    pass


@dataclass
class NumberExpr(Expr):
    value: float = 0.0


@dataclass
class VarExpr(Expr):
    name: str = ""


@dataclass
class CallExpr(Expr):
    func: str = ""  # "inference" | "argmax" | "confidence"
    arg: Expr | None = None


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------
@dataclass
class Layer(Node):
    kind: str = ""  # dense | linear | relu | argmax
    params: dict = field(default_factory=dict)


@dataclass
class ModelDecl(Node):
    name: str = ""
    layers: list[Layer] = field(default_factory=list)


@dataclass
class IODecl(Node):
    name: str = ""
    direction: str = ""  # input | output
    visibility: str = ""  # private | public
    kind: str = ""  # tensor | class
    size: int = 0
    range_low: float | None = None
    range_high: float | None = None


# ---------------------------------------------------------------------------
# Prove block statements
# ---------------------------------------------------------------------------
@dataclass
class ProveStmt(Node):
    pass


@dataclass
class ModelIsStmt(ProveStmt):
    name: str = ""


@dataclass
class InferenceEqStmt(ProveStmt):
    """`inference(x) == y`"""

    input_name: str = ""
    output_name: str = ""


@dataclass
class ConfidenceStmt(ProveStmt):
    """`confidence(x) >= 0.8`"""

    input_name: str = ""
    op: str = ">="
    threshold: float = 0.0


@dataclass
class AssignStmt(ProveStmt):
    """`y = inference(x)` or `y = x`"""

    target: str = ""
    value: Expr | None = None


@dataclass
class Program(Node):
    model: ModelDecl | None = None
    ios: list[IODecl] = field(default_factory=list)
    prove: list[ProveStmt] = field(default_factory=list)
    source_path: str | None = None
