"""End-to-end compilation pipeline.

``compile_source`` ties the stages together:

    source -> lexer -> parser -> AST -> semantic check
           -> ML IR -> Proof IR (baseline)
           -> [constant propagation, dead-constraint elimination,
               fixed-point quantization] -> optimized Proof IR
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import ast_nodes as ast
from .diagnostics import Diagnostic, SemanticError
from .lower import lower_to_ml, lower_to_proof
from .ml_ir import MLModule
from .opt import (
    PassResult,
    constant_prop,
    eliminate_dead_constraints,
    quantize_fixed_point,
)
from .parser import parse
from .proof_ir import ProofModule, summarize
from .semantic import CheckedSpec, analyze


@dataclass
class CompilationResult:
    spec: CheckedSpec
    ml: MLModule
    baseline: ProofModule
    optimized: ProofModule
    passes: list[PassResult] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def metrics(self) -> dict:
        b = summarize(self.baseline)
        o = summarize(self.optimized)
        row_delta = b["constraint_rows"] - o["constraint_rows"]
        cost_delta = b["r1cs_cost"] - o["r1cs_cost"]
        return {
            "baseline": b,
            "optimized": o,
            "constraint_rows_removed": row_delta,
            "r1cs_cost_removed": cost_delta,
            "r1cs_cost_reduction_pct": (
                round(100.0 * cost_delta / b["r1cs_cost"], 2)
                if b["r1cs_cost"]
                else 0.0
            ),
            "passes": [p.to_dict() for p in self.passes],
        }


def compile_source(
    source: str,
    source_path: str | None = None,
    optimize: bool = True,
) -> CompilationResult:
    program = parse(source, source_path)
    spec = analyze(program)
    if spec.has_errors:
        raise SemanticError(spec.errors)

    ml = lower_to_ml(spec)
    baseline = lower_to_proof(spec, ml, include_all=True)

    passes: list[PassResult] = []
    if optimize:
        ml_opt, cp_stats = constant_prop(ml)
        # Recompute the cost delta of constant propagation against the
        # baseline by lowering both and diffing.
        proof_after_cp = lower_to_proof(spec, ml_opt, include_all=True)
        cp_stats.before_rows = baseline.constraint_rows()
        cp_stats.before_cost = baseline.total_cost()
        cp_stats.after_rows = proof_after_cp.constraint_rows()
        cp_stats.after_cost = proof_after_cp.total_cost()
        passes.append(cp_stats)

        optimized, dce_stats = eliminate_dead_constraints(proof_after_cp)
        passes.append(dce_stats)
        optimized, q_stats = quantize_fixed_point(optimized)
        passes.append(q_stats)
    else:
        optimized = baseline.copy()

    return CompilationResult(
        spec=spec,
        ml=ml,
        baseline=baseline,
        optimized=optimized,
        passes=passes,
        diagnostics=spec.diagnostics,
    )


def compile_file(path: str, optimize: bool = True) -> CompilationResult:
    with open(path, "r", encoding="utf-8") as fh:
        source = fh.read()
    return compile_source(source, source_path=path, optimize=optimize)
