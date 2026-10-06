"""Runtime support: reference inference, witness generation, proof simulation."""

from .inference import forward, logits_signal, softmax  # noqa: F401
from .proof_sim import Proof, prove, reference_output, verify  # noqa: F401
from .witness import build_witness, model_commitment  # noqa: F401

__all__ = [
    "forward",
    "logits_signal",
    "softmax",
    "Proof",
    "prove",
    "verify",
    "reference_output",
    "build_witness",
    "model_commitment",
]
