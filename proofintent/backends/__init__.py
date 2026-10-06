"""Backend package: currently a Circom emitter plus a proof/witness simulator."""

from .circom_backend import emit_circom, generate_circom  # noqa: F401

__all__ = ["emit_circom", "generate_circom"]
