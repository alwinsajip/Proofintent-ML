"""Lexer for the ProofIntent-ML DSL.

Turns source text into a flat list of :class:`Token` objects. The lexer is
hand written on purpose so that line/column information is exact and so that
the compiler has no third-party dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

from .diagnostics import CompileError

KEYWORDS = {
    "model",
    "input",
    "output",
    "private",
    "public",
    "prove",
    "model_is",
    "inference",
    "confidence",
    "dense",
    "linear",
    "relu",
    "argmax",
    "tensor",
    "class",
    "range",
    "true",
    "false",
}

# Multi-character punctuation must be tried before single characters.
_MULTI_PUNCT = ("->", "==", "!=", ">=", "<=", "&&", "||")
_SINGLE_PUNCT = set("{}()[],:;.=><+-*/%")


@dataclass
class Token:
    type: str  # NUMBER | STRING | IDENT | KEYWORD | PUNCT | EOF
    value: object
    line: int
    col: int

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Token({self.type}, {self.value!r}, {self.line}:{self.col})"


class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.col = 1
        self.tokens: list[Token] = []

    # -- helpers -----------------------------------------------------------
    def _peek(self, offset: int = 0) -> str:
        idx = self.pos + offset
        if idx < len(self.source):
            return self.source[idx]
        return ""

    def _advance(self, count: int = 1) -> str:
        chunk = self.source[self.pos:self.pos + count]
        for ch in chunk:
            if ch == "\n":
                self.line += 1
                self.col = 1
            else:
                self.col += 1
        self.pos += count
        return chunk

    # -- driver ------------------------------------------------------------
    def tokenize(self) -> list[Token]:
        while self.pos < len(self.source):
            ch = self._peek()

            if ch in " \t\r\n":
                self._advance()
                continue

            if ch == "#":
                while self.pos < len(self.source) and self._peek() != "\n":
                    self._advance()
                continue

            if ch.isdigit() or (ch == "." and self._peek(1).isdigit()):
                self._read_number()
                continue

            if ch == '"':
                self._read_string()
                continue

            if ch.isalpha() or ch == "_":
                self._read_ident()
                continue

            if self._read_punct():
                continue

            raise CompileError(f"unexpected character {ch!r}", self.line, self.col)

        self.tokens.append(Token("EOF", None, self.line, self.col))
        return self.tokens

    # -- token readers -----------------------------------------------------
    def _read_number(self) -> None:
        line, col = self.line, self.col
        start = self.pos
        is_float = False
        while self._peek().isdigit():
            self._advance()
        if self._peek() == ".":
            is_float = True
            self._advance()
            while self._peek().isdigit():
                self._advance()
        if self._peek() in ("e", "E"):
            is_float = True
            self._advance()
            if self._peek() in ("+", "-"):
                self._advance()
            while self._peek().isdigit():
                self._advance()
        text = self.source[start:self.pos]
        value = float(text) if is_float else int(text)
        self.tokens.append(Token("NUMBER", value, line, col))

    def _read_string(self) -> None:
        line, col = self.line, self.col
        self._advance()  # opening quote
        chars: list[str] = []
        while True:
            ch = self._peek()
            if ch == "":
                raise CompileError("unterminated string literal", line, col)
            if ch == "\\":
                self._advance()
                esc = self._advance()
                mapping = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}
                chars.append(mapping.get(esc, esc))
                continue
            if ch == '"':
                self._advance()
                break
            chars.append(self._advance())
        self.tokens.append(Token("STRING", "".join(chars), line, col))

    def _read_ident(self) -> None:
        line, col = self.line, self.col
        start = self.pos
        while self._peek().isalnum() or self._peek() == "_":
            self._advance()
        text = self.source[start:self.pos]
        tok_type = "KEYWORD" if text in KEYWORDS else "IDENT"
        self.tokens.append(Token(tok_type, text, line, col))

    def _read_punct(self) -> bool:
        line, col = self.line, self.col
        for p in _MULTI_PUNCT:
            if self.source.startswith(p, self.pos):
                self._advance(len(p))
                self.tokens.append(Token("PUNCT", p, line, col))
                return True
        ch = self._peek()
        if ch in _SINGLE_PUNCT:
            self._advance()
            self.tokens.append(Token("PUNCT", ch, line, col))
            return True
        return False


def tokenize(source: str) -> list[Token]:
    return Lexer(source).tokenize()
