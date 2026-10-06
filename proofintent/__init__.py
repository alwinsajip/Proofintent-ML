"""ProofIntent-ML: a compiler for semantic zero-knowledge verification of ML inference."""

from .compiler import CompilationResult, compile_file, compile_source
from .diagnostics import CompileError, Diagnostic, SemanticError

__version__ = "0.1.0"

__all__ = [
    "CompilationResult",
    "compile_file",
    "compile_source",
    "CompileError",
    "Diagnostic",
    "SemanticError",
    "__version__",
]
