import unittest

from proofintent.parser import parse
from proofintent.semantic import analyze


def codes(source: str):
    return [d.code for d in analyze(parse(source)).errors]


BASE_LAYERS = 'dense(in: 2, out: 3, weights: "w", bias: "b") argmax'


class SemanticTests(unittest.TestCase):
    def test_valid_spec(self):
        src = f"""
        model M {{ {BASE_LAYERS} }}
        input  x : private tensor[2]
        output y : public  class[3]
        prove {{ model_is M
                 inference(x) == y }}
        """
        spec = analyze(parse(src))
        self.assertFalse(spec.has_errors, [d.format() for d in spec.errors])

    def test_e001_undefined_model(self):
        src = f"""
        model M {{ {BASE_LAYERS} }}
        input  x : private tensor[2]
        output y : public  class[3]
        prove {{ model_is OTHER
                 inference(x) == y }}
        """
        self.assertIn("E001", codes(src))

    def test_e002_private_to_public_leak(self):
        src = f"""
        model M {{ {BASE_LAYERS} }}
        input  x : private tensor[2]
        output y : public  class[3]
        prove {{ model_is M
                 y = x }}
        """
        self.assertIn("E002", codes(src))

    def test_e002_absent_when_declassified(self):
        src = f"""
        model M {{ {BASE_LAYERS} }}
        input  x : private tensor[2]
        output y : public  class[3]
        prove {{ model_is M
                 y = inference(x) }}
        """
        self.assertNotIn("E002", codes(src))

    def test_e003_unbound_output(self):
        src = f"""
        model M {{ {BASE_LAYERS} }}
        input  x : private tensor[2]
        output y : public  class[3]
        prove {{ model_is M }}
        """
        self.assertIn("E003", codes(src))

    def test_e004_invalid_range(self):
        src = f"""
        model M {{ {BASE_LAYERS} }}
        input  x : private tensor[2] range[5.0, -1.0]
        output y : public  class[3]
        prove {{ model_is M
                 inference(x) == y }}
        """
        self.assertIn("E004", codes(src))

    def test_e005_bad_confidence_threshold(self):
        src = f"""
        model M {{ {BASE_LAYERS} }}
        input  x : private tensor[2]
        output y : public  class[3]
        prove {{ model_is M
                 inference(x) == y
                 confidence(x) >= 1.5 }}
        """
        self.assertIn("E005", codes(src))

    def test_e006_dimension_mismatch(self):
        src = """
        model M {
            dense(in: 2, out: 4, weights: "w1", bias: "b1")
            dense(in: 5, out: 3, weights: "w2", bias: "b2")
            argmax
        }
        input  x : private tensor[2]
        output y : public  class[3]
        prove { model_is M
                inference(x) == y }
        """
        self.assertIn("E006", codes(src))

    def test_e006_argmax_not_terminal(self):
        src = """
        model M {
            dense(in: 2, out: 3, weights: "w1", bias: "b1")
            argmax
            relu
        }
        input  x : private tensor[2]
        output y : public  class[3]
        prove { model_is M
                inference(x) == y }
        """
        self.assertIn("E006", codes(src))

    def test_missing_model_is(self):
        src = f"""
        model M {{ {BASE_LAYERS} }}
        input  x : private tensor[2]
        output y : public  class[3]
        prove {{ inference(x) == y }}
        """
        self.assertIn("E001", codes(src))


if __name__ == "__main__":
    unittest.main()
