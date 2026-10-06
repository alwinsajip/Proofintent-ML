"""A lightweight proof/verification simulator for the emitted circuit.

This is *not* a zero-knowledge proof system. It is a faithful stand-in used by
the compiler's test-suite and benchmarks so that the full pipeline can be
exercised offline on Windows without a Circom/snarkjs toolchain. It reproduces
the two properties the report asks to demonstrate:

* the proof is *bound* to a specific committed model (tampering with a weight
  changes the commitment and verification fails);
* the proof is *bound* to the published prediction (tampering with the output
  changes the transcript digest and verification fails).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from ..compiler import CompilationResult
from ..ml_ir import MLModule
from .inference import forward, logits_signal
from .witness import build_witness, model_commitment


def _digest(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class Proof:
    commitment: str
    outputs: dict
    witness_digest: str
    transcript_digest: str
    checks_ok: bool
    checks: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "commitment": self.commitment,
            "outputs": self.outputs,
            "witness_digest": self.witness_digest,
            "transcript_digest": self.transcript_digest,
            "checks_ok": self.checks_ok,
            "checks": self.checks,
        }


def _transcript(commitment: str, outputs: dict, witness_digest: str) -> str:
    return _digest(
        {"commitment": commitment, "outputs": outputs, "witness_digest": witness_digest}
    )


def prove(result: CompilationResult, x: list[float]) -> Proof:
    witness, checks = build_witness(result, x)
    checks_ok = all(c["ok"] for c in checks)

    out_var = next(
        (i.detail.get("output") for i in result.optimized.intents if i.kind == "inference"),
        None,
    )
    outputs: dict = {}
    if out_var is not None:
        outputs[out_var] = witness.get(out_var)
    if "conf" in witness and witness["conf"]:
        outputs["confidence"] = max(witness["conf"])

    commitment = witness.get("model_commit", "")
    witness_digest = _digest(witness)
    return Proof(
        commitment=commitment,
        outputs=outputs,
        witness_digest=witness_digest,
        transcript_digest=_transcript(commitment, outputs, witness_digest),
        checks_ok=checks_ok,
        checks=checks,
    )


def verify(
    proof: Proof,
    ml: MLModule,
    expected_output: dict | None = None,
) -> tuple[bool, str]:
    commitment = model_commitment(ml)
    if commitment != proof.commitment:
        return False, "model commitment mismatch: the committed model was tampered with"

    recomputed = _transcript(proof.commitment, proof.outputs, proof.witness_digest)
    if recomputed != proof.transcript_digest:
        return False, "transcript mismatch: the published prediction was tampered with"

    if not proof.checks_ok:
        failed = [c for c in proof.checks if not c["ok"]]
        return False, f"circuit constraints not satisfied ({len(failed)} failed)"

    if expected_output is not None:
        for key, value in expected_output.items():
            if proof.outputs.get(key) != value:
                return False, f"expected {key}={value} but proof claims {proof.outputs.get(key)}"

    return True, "ok"


def reference_output(result: CompilationResult, x: list[float]) -> dict:
    """Outputs the reference (non-ZK) evaluator produces, for comparison."""
    ml = result.ml
    values = forward(ml, x)
    out_var = next(
        (i.detail.get("output") for i in result.optimized.intents if i.kind == "inference"),
        None,
    )
    outputs: dict = {}
    if out_var is not None:
        final = ml.ops[-1].output if ml.ops else out_var
        outputs[out_var] = values.get(final, values.get(out_var))
    logits = logits_signal(ml)
    if logits and logits in values:
        from .inference import softmax

        outputs["confidence"] = max(softmax(list(values[logits])))
    return outputs
