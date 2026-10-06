"""Fixed-point quantization of the Proof IR.

Real ZKML compilers quantize floating-point networks to fixed-point integers so
that arithmetic can be expressed over a finite field. Quantization also changes
the cost model: once activations are known to live in a bounded ``b``-bit
range, a ReLU no longer needs one booleanization constraint per unit -- a single
shared saturation check per layer suffices -- and Argmax becomes a set of
integer comparisons costing one constraint each instead of two.

The pass annotates every signal with its fixed-point type and rewrites the
constraint costs accordingly.
"""

from __future__ import annotations

from ..proof_ir import ProofModule, constraint_cost
from . import PassResult


def quantize_fixed_point(
    module: ProofModule, bits: int = 16, frac_bits: int = 8
) -> tuple[ProofModule, PassResult]:
    before_rows = module.constraint_rows()
    before_cost = module.total_cost()

    result = module.copy()
    dtype = f"fixed<{bits},{frac_bits}>"

    for sig in result.signals:
        if sig.dtype == "field":
            sig.dtype = dtype

    relu_layers = 0
    for c in result.constraints:
        if c.kind == "relu":
            # Shared saturation check: cost 1 per layer, counted once.
            if c.meta.get("n_out", 0) > 0 and not c.meta.get("_saturation_counted"):
                c.cost = 1
                c.meta["_saturation_counted"] = True
                c.meta["quantized"] = True
                relu_layers += 1
            else:
                c.cost = 0
                c.meta["quantized"] = True
        elif c.kind == "argmax":
            n = int(c.meta.get("n_out", 1))
            c.cost = constraint_cost("argmax", {"n_out": n}) // 2 if n > 1 else 0
            c.meta["quantized"] = True
        elif c.kind == "dense":
            c.meta["quantized"] = True
        elif c.kind == "range":
            c.meta["quantized"] = True

    result.metadata["quantization"] = {"bits": bits, "frac_bits": frac_bits}

    stats = PassResult(
        name="fixed_point_quantization",
        before_rows=before_rows,
        after_rows=result.constraint_rows(),
        before_cost=before_cost,
        after_cost=result.total_cost(),
        details={"bits": bits, "frac_bits": frac_bits, "quantized_relu_layers": relu_layers},
    )
    return result, stats
