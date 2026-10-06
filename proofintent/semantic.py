"""Semantic analysis and privacy checking for ProofIntent-ML.

This module performs everything that must happen *before* any IR is built:

* name resolution for models, inputs and outputs;
* structural checks on the declared network;
* the privacy check (a private value may not flow to a public one without a
  declassifying inference step);
* consistency checks on ranges, thresholds and proof intent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import ast_nodes as ast
from .diagnostics import Diagnostic


@dataclass
class CheckedSpec:
    """A fully resolved specification, ready for lowering."""

    program: ast.Program
    model: ast.ModelDecl
    inputs: dict[str, ast.IODecl]
    outputs: dict[str, ast.IODecl]
    model_is: ast.ModelIsStmt | None = None
    inference: ast.InferenceEqStmt | None = None
    confidence: ast.ConfidenceStmt | None = None
    assignments: list[ast.AssignStmt] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == "error"]

    @property
    def has_errors(self) -> bool:
        return any(d.severity == "error" for d in self.diagnostics)


class SemanticChecker:
    def __init__(self, program: ast.Program):
        self.program = program
        self.diagnostics: list[Diagnostic] = []

    # -- reporting ---------------------------------------------------------
    def _error(self, code: str, message: str, node: ast.Node) -> None:
        self.diagnostics.append(
            Diagnostic("error", code, message, node.line, node.col)
        )

    def _warning(self, code: str, message: str, node: ast.Node) -> None:
        self.diagnostics.append(
            Diagnostic("warning", code, message, node.line, node.col)
        )

    # -- driver ------------------------------------------------------------
    def check(self) -> CheckedSpec:
        program = self.program
        model = program.model
        assert model is not None

        inputs: dict[str, ast.IODecl] = {}
        outputs: dict[str, ast.IODecl] = {}

        for decl in program.ios:
            table = inputs if decl.direction == "input" else outputs
            if decl.name in inputs or decl.name in outputs:
                self._error(
                    "E007",
                    f"duplicate declaration of '{decl.name}'",
                    decl,
                )
                continue
            table[decl.name] = decl
            self._check_io_decl(decl)

        self._check_model_layers(model)

        spec = CheckedSpec(
            program=program,
            model=model,
            inputs=inputs,
            outputs=outputs,
        )

        self._check_prove(spec)
        self._check_privacy(spec)
        self._check_outputs_bound(spec)

        spec.diagnostics = self.diagnostics
        return spec

    # -- individual checks -------------------------------------------------
    def _check_io_decl(self, decl: ast.IODecl) -> None:
        if decl.size <= 0:
            self._error(
                "E008", f"'{decl.name}' must have a positive size", decl
            )
        if decl.direction == "output" and decl.visibility == "private":
            self._warning(
                "W001",
                f"output '{decl.name}' is private; the verifier cannot learn it",
                decl,
            )
        if (
            decl.range_low is not None
            and decl.range_high is not None
            and decl.range_low > decl.range_high
        ):
            self._error(
                "E004",
                f"invalid range for '{decl.name}': "
                f"min {decl.range_low} is greater than max {decl.range_high}",
                decl,
            )

    def _check_model_layers(self, model: ast.ModelDecl) -> None:
        prev_out: int | None = None
        seen_argmax = False
        for idx, layer in enumerate(model.layers):
            if seen_argmax:
                self._error(
                    "E006",
                    "no layer may follow 'argmax'",
                    layer,
                )
            params = layer.params
            if layer.kind in ("dense", "linear"):
                if "in" not in params or "out" not in params:
                    self._error(
                        "E006",
                        f"'{layer.kind}' layer requires 'in' and 'out' parameters",
                        layer,
                    )
                    continue
                try:
                    n_in = int(params["in"])
                    n_out = int(params["out"])
                except (TypeError, ValueError):
                    self._error(
                        "E006", f"'{layer.kind}' dimensions must be numbers", layer
                    )
                    continue
                if n_in <= 0 or n_out <= 0:
                    self._error(
                        "E006",
                        f"'{layer.kind}' dimensions must be positive", layer
                    )
                if prev_out is not None and n_in != prev_out:
                    self._error(
                        "E006",
                        f"dimension mismatch: layer expects in={n_in} "
                        f"but previous layer produced {prev_out}",
                        layer,
                    )
                if "weights" not in params:
                    self._warning(
                        "W002",
                        f"'{layer.kind}' layer has no weights file; "
                        "identity weights assumed",
                        layer,
                    )
                prev_out = n_out
            elif layer.kind == "relu":
                if prev_out is None:
                    self._error("E006", "'relu' requires a preceding layer", layer)
            elif layer.kind == "argmax":
                if prev_out is None:
                    self._error("E006", "'argmax' requires a preceding layer", layer)
                seen_argmax = True

    def _check_prove(self, spec: CheckedSpec) -> None:
        declared_models = {spec.model.name}
        for stmt in spec.program.prove:
            if isinstance(stmt, ast.ModelIsStmt):
                if spec.model_is is not None:
                    self._warning("W003", "duplicate 'model_is' statement", stmt)
                if stmt.name not in declared_models:
                    self._error(
                        "E001",
                        f"model_is references undefined model '{stmt.name}'; "
                        f"declared model is '{spec.model.name}'",
                        stmt,
                    )
                spec.model_is = stmt
            elif isinstance(stmt, ast.InferenceEqStmt):
                if stmt.input_name not in spec.inputs:
                    self._error(
                        "E005",
                        f"'{stmt.input_name}' used in inference is not a "
                        "declared input",
                        stmt,
                    )
                if stmt.output_name not in spec.outputs:
                    self._error(
                        "E005",
                        f"'{stmt.output_name}' used in inference is not a "
                        "declared output",
                        stmt,
                    )
                if spec.inference is not None:
                    self._warning("W003", "duplicate 'inference' statement", stmt)
                spec.inference = stmt
            elif isinstance(stmt, ast.ConfidenceStmt):
                if stmt.input_name not in spec.inputs:
                    self._error(
                        "E005",
                        f"'{stmt.input_name}' used in confidence is not a "
                        "declared input",
                        stmt,
                    )
                if not (0.0 <= stmt.threshold <= 1.0):
                    self._error(
                        "E005",
                        f"confidence threshold {stmt.threshold} is outside [0, 1]",
                        stmt,
                    )
                if spec.confidence is not None:
                    self._warning("W003", "duplicate 'confidence' statement", stmt)
                spec.confidence = stmt
            elif isinstance(stmt, ast.AssignStmt):
                if stmt.target not in spec.outputs:
                    self._error(
                        "E005",
                        f"assignment target '{stmt.target}' is not a declared output",
                        stmt,
                    )
                self._check_expr_vars(stmt.value, spec)
                spec.assignments.append(stmt)

        if spec.model_is is None:
            self._error(
                "E001",
                "missing 'model_is' statement: the compiler cannot tell which "
                "model must be proven",
                spec.program,
            )

    def _check_expr_vars(self, expr: ast.Expr | None, spec: CheckedSpec) -> None:
        if expr is None:
            return
        if isinstance(expr, ast.VarExpr):
            if expr.name not in spec.inputs and expr.name not in spec.outputs:
                self._error(
                    "E005", f"'{expr.name}' is not a declared variable", expr
                )
        elif isinstance(expr, ast.CallExpr):
            # The innermost argument must be a declared input.
            self._check_expr_vars(expr.arg, spec)

    # -- privacy -----------------------------------------------------------
    def _private_vars(self, spec: CheckedSpec) -> set[str]:
        return {name for name, d in spec.inputs.items() if d.visibility == "private"}

    def _taint(self, expr: ast.Expr | None, spec: CheckedSpec) -> set[str]:
        """Return the set of private variables an expression depends on.

        Calls to ``inference``, ``argmax`` and ``confidence`` are
        *declassifiers*: their result is safe to publish because the whole
        point of the proof is to reveal only the inference outcome.
        """
        private = self._private_vars(spec)
        if expr is None:
            return set()
        if isinstance(expr, ast.NumberExpr):
            return set()
        if isinstance(expr, ast.VarExpr):
            return {expr.name} if expr.name in private else set()
        if isinstance(expr, ast.CallExpr):
            if expr.func in ("inference", "argmax", "confidence"):
                return set()
            return self._taint(expr.arg, spec)
        return set()

    def _check_privacy(self, spec: CheckedSpec) -> None:
        for stmt in spec.assignments:
            target = spec.outputs.get(stmt.target)
            if target is None or target.visibility != "public":
                continue
            tainted = self._taint(stmt.value, spec)
            if tainted:
                offenders = ", ".join(sorted(tainted))
                self._error(
                    "E002",
                    f"privacy leak: public output '{stmt.target}' is assigned a "
                    f"value derived from private input(s) {offenders} without an "
                    "inference step",
                    stmt,
                )

        # An inference statement deliberately publishes a private input's
        # result, but only if the input really is the private one and the
        # output is public. If the output is public and the input public this
        # is just redundant, not a leak.
        if spec.inference is not None:
            out = spec.outputs.get(spec.inference.output_name)
            if out is not None and out.visibility == "public":
                pass  # declassified by construction

    # -- output binding ----------------------------------------------------
    def _check_outputs_bound(self, spec: CheckedSpec) -> None:
        bound: set[str] = set()
        if spec.inference is not None:
            bound.add(spec.inference.output_name)
        for stmt in spec.assignments:
            bound.add(stmt.target)
        for name, decl in spec.outputs.items():
            if name not in bound:
                self._error(
                    "E003",
                    f"output '{name}' is never bound by an inference or "
                    "assignment; the verifier could not check it",
                    decl,
                )


def analyze(program: ast.Program) -> CheckedSpec:
    return SemanticChecker(program).check()
