"""Recursive-descent parser for the ProofIntent-ML DSL."""

from __future__ import annotations

from . import ast_nodes as ast
from .diagnostics import CompileError
from .lexer import Token, tokenize


class Parser:
    def __init__(self, tokens: list[Token], source_path: str | None = None):
        self.tokens = tokens
        self.pos = 0
        self.source_path = source_path

    # -- token helpers -----------------------------------------------------
    def _cur(self) -> Token:
        return self.tokens[self.pos]

    def _peek(self, offset: int = 1) -> Token:
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return self.tokens[-1]

    def _at_punct(self, value: str) -> bool:
        tok = self._cur()
        return tok.type == "PUNCT" and tok.value == value

    def _at_keyword(self, value: str) -> bool:
        tok = self._cur()
        return tok.type == "KEYWORD" and tok.value == value

    def _at_ident(self) -> bool:
        return self._cur().type == "IDENT"

    def _advance(self) -> Token:
        tok = self._cur()
        if tok.type != "EOF":
            self.pos += 1
        return tok

    def _expect_punct(self, value: str) -> Token:
        if not self._at_punct(value):
            tok = self._cur()
            raise CompileError(
                f"expected {value!r} but found {tok.value!r}", tok.line, tok.col
            )
        return self._advance()

    def _expect_keyword(self, value: str) -> Token:
        if not self._at_keyword(value):
            tok = self._cur()
            raise CompileError(
                f"expected keyword {value!r} but found {tok.value!r}",
                tok.line,
                tok.col,
            )
        return self._advance()

    def _expect_ident(self) -> Token:
        if not self._at_ident():
            tok = self._cur()
            raise CompileError(
                f"expected identifier but found {tok.value!r}", tok.line, tok.col
            )
        return self._advance()

    def _expect_type(self, tok_type: str) -> Token:
        tok = self._cur()
        if tok.type != tok_type:
            raise CompileError(
                f"expected {tok_type} but found {tok.value!r}", tok.line, tok.col
            )
        return self._advance()

    def _optional_semicolon(self) -> None:
        if self._at_punct(";"):
            self._advance()

    # -- entry point -------------------------------------------------------
    def parse_program(self) -> ast.Program:
        program = ast.Program(source_path=self.source_path)
        first = self._cur()
        if not self._at_keyword("model"):
            raise CompileError(
                f"expected 'model' declaration but found {first.value!r}",
                first.line,
                first.col,
            )
        program.model = self._parse_model()

        while self._at_keyword("input") or self._at_keyword("output"):
            program.ios.append(self._parse_io())

        program.prove = self._parse_prove()

        if self._cur().type != "EOF":
            tok = self._cur()
            raise CompileError(
                f"unexpected token {tok.value!r} after end of program",
                tok.line,
                tok.col,
            )
        return program

    # -- model -------------------------------------------------------------
    def _parse_model(self) -> ast.ModelDecl:
        start = self._expect_keyword("model")
        name = self._expect_ident()
        self._expect_punct("{")
        layers: list[ast.Layer] = []
        while not self._at_punct("}"):
            if self._cur().type == "EOF":
                raise CompileError("unterminated model block", start.line, start.col)
            layers.append(self._parse_layer())
        self._expect_punct("}")
        return ast.ModelDecl(
            name=name.value, layers=layers, line=start.line, col=start.col
        )

    def _parse_layer(self) -> ast.Layer:
        tok = self._cur()
        if tok.type != "KEYWORD" or tok.value not in ("dense", "linear", "relu", "argmax"):
            raise CompileError(
                f"expected a layer (dense/linear/relu/argmax) but found {tok.value!r}",
                tok.line,
                tok.col,
            )
        kind = tok.value
        self._advance()
        params: dict = {}
        if self._at_punct("("):
            self._advance()
            while not self._at_punct(")"):
                key = self._expect_type("IDENT")
                self._expect_punct(":")
                if self._cur().type == "NUMBER":
                    value = self._expect_type("NUMBER").value
                elif self._cur().type == "STRING":
                    value = self._expect_type("STRING").value
                else:
                    raise CompileError(
                        "layer parameter value must be a number or string",
                        self._cur().line,
                        self._cur().col,
                    )
                params[key.value] = value
                if self._at_punct(","):
                    self._advance()
                elif not self._at_punct(")"):
                    raise CompileError(
                        "expected ',' or ')' in layer arguments",
                        self._cur().line,
                        self._cur().col,
                    )
            self._expect_punct(")")
        self._optional_semicolon()
        return ast.Layer(kind=kind, params=params, line=tok.line, col=tok.col)

    # -- io decls ----------------------------------------------------------
    def _parse_io(self) -> ast.IODecl:
        direction_tok = self._advance()  # input | output
        name = self._expect_ident()
        self._expect_punct(":")
        vis_tok = self._cur()
        if vis_tok.type != "KEYWORD" or vis_tok.value not in ("private", "public"):
            raise CompileError(
                "expected visibility 'private' or 'public'",
                vis_tok.line,
                vis_tok.col,
            )
        self._advance()
        kind_tok = self._cur()
        if kind_tok.type != "KEYWORD" or kind_tok.value not in ("tensor", "class"):
            raise CompileError(
                "expected type 'tensor' or 'class'", kind_tok.line, kind_tok.col
            )
        self._advance()
        self._expect_punct("[")
        size = self._expect_type("NUMBER")
        self._expect_punct("]")

        low = high = None
        if self._at_keyword("range"):
            self._advance()
            self._expect_punct("[")
            low_expr = self._parse_signed_number()
            self._expect_punct(",")
            high_expr = self._parse_signed_number()
            self._expect_punct("]")
            low, high = low_expr, high_expr

        self._optional_semicolon()
        return ast.IODecl(
            name=name.value,
            direction=direction_tok.value,
            visibility=vis_tok.value,
            kind=kind_tok.value,
            size=int(size.value),
            range_low=low,
            range_high=high,
            line=direction_tok.line,
            col=direction_tok.col,
        )

    def _parse_signed_number(self) -> float:
        sign = 1.0
        if self._at_punct("-"):
            sign = -1.0
            self._advance()
        elif self._at_punct("+"):
            self._advance()
        num = self._expect_type("NUMBER")
        return sign * float(num.value)

    # -- prove block -------------------------------------------------------
    def _parse_prove(self) -> list[ast.ProveStmt]:
        start = self._expect_keyword("prove")
        self._expect_punct("{")
        stmts: list[ast.ProveStmt] = []
        while not self._at_punct("}"):
            if self._cur().type == "EOF":
                raise CompileError("unterminated prove block", start.line, start.col)
            stmts.append(self._parse_prove_stmt())
        self._expect_punct("}")
        return stmts

    def _parse_prove_stmt(self) -> ast.ProveStmt:
        tok = self._cur()
        if self._at_keyword("model_is"):
            self._advance()
            name = self._expect_ident()
            self._optional_semicolon()
            return ast.ModelIsStmt(name=name.value, line=tok.line, col=tok.col)

        if self._at_keyword("inference"):
            self._advance()
            self._expect_punct("(")
            input_name = self._expect_ident()
            self._expect_punct(")")
            self._expect_punct("==")
            output_name = self._expect_ident()
            self._optional_semicolon()
            return ast.InferenceEqStmt(
                input_name=input_name.value,
                output_name=output_name.value,
                line=tok.line,
                col=tok.col,
            )

        if self._at_keyword("confidence"):
            self._advance()
            self._expect_punct("(")
            input_name = self._expect_ident()
            self._expect_punct(")")
            op_tok = self._cur()
            if op_tok.type != "PUNCT" or op_tok.value not in (">=", "<=", ">", "<", "=="):
                raise CompileError(
                    "expected a comparison operator", op_tok.line, op_tok.col
                )
            self._advance()
            threshold = self._parse_signed_number()
            self._optional_semicolon()
            return ast.ConfidenceStmt(
                input_name=input_name.value,
                op=op_tok.value,
                threshold=threshold,
                line=tok.line,
                col=tok.col,
            )

        # otherwise an assignment: IDENT '=' expr
        target = self._expect_ident()
        self._expect_punct("=")
        value = self._parse_expr()
        self._optional_semicolon()
        return ast.AssignStmt(target=target.value, value=value, line=tok.line, col=tok.col)

    def _parse_expr(self) -> ast.Expr:
        tok = self._cur()
        if tok.type == "NUMBER":
            self._advance()
            return ast.NumberExpr(value=float(tok.value), line=tok.line, col=tok.col)
        if tok.type == "PUNCT" and tok.value == "-":
            self._advance()
            num = self._expect_type("NUMBER")
            return ast.NumberExpr(value=-float(num.value), line=tok.line, col=tok.col)
        if tok.type == "IDENT":
            self._advance()
            return ast.VarExpr(name=tok.value, line=tok.line, col=tok.col)
        if tok.type == "KEYWORD" and tok.value in ("inference", "argmax", "confidence"):
            self._advance()
            self._expect_punct("(")
            inner = self._parse_expr()
            self._expect_punct(")")
            return ast.CallExpr(func=tok.value, arg=inner, line=tok.line, col=tok.col)
        raise CompileError(
            f"expected an expression but found {tok.value!r}", tok.line, tok.col
        )


def parse(source: str, source_path: str | None = None) -> ast.Program:
    tokens = tokenize(source)
    return Parser(tokens, source_path).parse_program()
