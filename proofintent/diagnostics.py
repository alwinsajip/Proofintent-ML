"""Diagnostics shared across the ProofIntent-ML compiler."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Diagnostic:
    """A single compiler message with a source location."""

    severity: str  # "error" | "warning" | "note"
    code: str
    message: str
    line: int = 0
    col: int = 0

    def format(self) -> str:
        loc = f"{self.line}:{self.col}" if self.line else "-"
        return f"[{self.severity.upper()}] {self.code} at {loc}: {self.message}"


class CompileError(Exception):
    """Raised when the front end cannot proceed (lex/parse)."""

    def __init__(self, message: str, line: int = 0, col: int = 0):
        super().__init__(message)
        self.message = message
        self.line = line
        self.col = col

    def format(self) -> str:
        loc = f"{self.line}:{self.col}" if self.line else "-"
        return f"[ERROR] {loc}: {self.message}"


class SemanticError(Exception):
    """Raised when semantic checking fails and errors should be reported."""

    def __init__(self, diagnostics):
        super().__init__("semantic errors")
        self.diagnostics = list(diagnostics)
