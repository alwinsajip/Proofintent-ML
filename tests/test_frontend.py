import unittest

from proofintent.diagnostics import CompileError
from proofintent.lexer import tokenize
from proofintent.parser import parse

VALID = """
model M {
    dense(in: 2, out: 3, weights: "w.json", bias: "b.json")
    relu
    argmax
}
input  x : private tensor[2] range[-1.0, 1.0]
output y : public  class[3]
prove {
    model_is M
    inference(x) == y
    confidence(x) >= 0.5
}
"""


class LexerTests(unittest.TestCase):
    def test_keywords_and_punct(self):
        toks = tokenize("model M { relu }")
        kinds = [(t.type, t.value) for t in toks]
        self.assertEqual(kinds[0], ("KEYWORD", "model"))
        self.assertEqual(kinds[1], ("IDENT", "M"))
        self.assertEqual(kinds[2], ("PUNCT", "{"))
        self.assertEqual(kinds[3], ("KEYWORD", "relu"))
        self.assertEqual(kinds[4], ("PUNCT", "}"))
        self.assertEqual(kinds[-1][0], "EOF")

    def test_number_and_string(self):
        toks = tokenize('dense(in: 4, out: 8, weights: "w1.json")')
        values = [t.value for t in toks if t.type in ("NUMBER", "STRING")]
        self.assertIn(4, values)
        self.assertIn(8, values)
        self.assertIn("w1.json", values)

    def test_comments_and_negative_numbers(self):
        toks = tokenize("range[-1.5, 2] # trailing comment")
        types = [t.type for t in toks]
        self.assertEqual(types[0], "KEYWORD")
        self.assertEqual(types[-1], "EOF")

    def test_unterminated_string(self):
        with self.assertRaises(CompileError):
            tokenize('weights: "oops')


class ParserTests(unittest.TestCase):
    def test_parse_valid(self):
        program = parse(VALID)
        self.assertEqual(program.model.name, "M")
        self.assertEqual(len(program.model.layers), 3)
        self.assertEqual(len(program.ios), 2)
        self.assertEqual(len(program.prove), 3)

    def test_layer_params(self):
        program = parse(VALID)
        dense = program.model.layers[0]
        self.assertEqual(dense.kind, "dense")
        self.assertEqual(dense.params["in"], 2)
        self.assertEqual(dense.params["out"], 3)

    def test_range_parsed(self):
        program = parse(VALID)
        x = [d for d in program.ios if d.name == "x"][0]
        self.assertEqual(x.range_low, -1.0)
        self.assertEqual(x.range_high, 1.0)
        self.assertEqual(x.visibility, "private")

    def test_missing_model(self):
        with self.assertRaises(CompileError):
            parse("input x : private tensor[1]")

    def test_unterminated_block(self):
        with self.assertRaises(CompileError):
            parse("model M { dense(in: 1, out: 1) ")

    def test_bad_prove_statement(self):
        with self.assertRaises(CompileError):
            parse("model M { relu }\nprove { model_is }")


if __name__ == "__main__":
    unittest.main()
