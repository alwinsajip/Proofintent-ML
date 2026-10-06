"""Optimization passes used by the ProofIntent-ML pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PassResult:
    name: str
    before_rows: int
    after_rows: int
    before_cost: int
    after_cost: int
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "pass": self.name,
            "before_rows": self.before_rows,
            "after_rows": self.after_rows,
            "before_cost": self.before_cost,
            "after_cost": self.after_cost,
            "cost_removed": self.before_cost - self.after_cost,
            "details": self.details,
        }


from .constant_prop import constant_prop  # noqa: E402
from .dead_constraint import eliminate_dead_constraints  # noqa: E402
from .quantize import quantize_fixed_point  # noqa: E402

__all__ = [
    "PassResult",
    "constant_prop",
    "eliminate_dead_constraints",
    "quantize_fixed_point",
]
