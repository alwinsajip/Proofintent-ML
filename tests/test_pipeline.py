import dataclasses
import json
import os
import unittest

from proofintent.backends.circom_backend import generate_circom
from proofintent.cli import _tampered_ml
from proofintent.compiler import compile_file
from proofintent.runtime.proof_sim import prove, reference_output, verify

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMALL = os.path.join(ROOT, "examples", "mlp_small.piml")
LARGE = os.path.join(ROOT, "examples", "mlp_large.piml")


def sample_input(name: str):
    with open(os.path.join(ROOT, "examples", "inputs", f"{name}.json"), encoding="utf-8") as fh:
        return [float(v) for v in json.load(fh)]


class LoweringTests(unittest.TestCase):
    def test_small_pipeline(self):
        result = compile_file(SMALL)
        self.assertEqual(result.ml.name, "MLP_SMALL")
        self.assertGreater(result.baseline.total_cost(), 0)
        self.assertLessEqual(result.optimized.total_cost(), result.baseline.total_cost())

    def test_signals_present(self):
        result = compile_file(SMALL)
        names = {s.name for s in result.optimized.signals}
        self.assertIn("x", names)
        self.assertIn("y", names)
        self.assertIn("model_commit", names)


class PassTests(unittest.TestCase):
    def test_all_passes_never_increase_cost(self):
        result = compile_file(LARGE)
        self.assertTrue(result.passes)
        for p in result.passes:
            self.assertLessEqual(p.after_cost, p.before_cost, p.name)

    def test_optimization_reduces_large(self):
        result = compile_file(LARGE)
        self.assertLess(result.optimized.total_cost(), result.baseline.total_cost())

    def test_constant_prop_folds_zero_layer(self):
        result = compile_file(LARGE)
        cp = result.passes[0]
        self.assertGreater(cp.details["folded_ops"], 0)

    def test_dce_removes_softmax_without_confidence_intent(self):
        result = compile_file(LARGE)
        baseline_kinds = {c.kind for c in result.baseline.constraints}
        opt_kinds = {c.kind for c in result.optimized.constraints}
        self.assertIn("softmax", baseline_kinds)
        self.assertNotIn("softmax", opt_kinds)

    def test_quantization_annotates(self):
        result = compile_file(SMALL)
        self.assertIn("quantization", result.optimized.metadata)
        self.assertTrue(any(s.dtype.startswith("fixed") for s in result.optimized.signals))


class BackendTests(unittest.TestCase):
    def test_circom_emitted(self):
        result = compile_file(SMALL)
        src = generate_circom(result.spec, result.ml, result.optimized)
        self.assertIn("pragma circom 2.0.0;", src)
        self.assertIn("template Dense", src)
        self.assertIn("component main", src)
        self.assertIn("signal input x[4];", src)
        self.assertIn("signal output y;", src)

    def test_public_private_lists(self):
        result = compile_file(SMALL)
        pub = {s.name for s in result.optimized.public_signals()}
        priv = {s.name for s in result.optimized.private_signals()}
        self.assertIn("y", pub)
        self.assertIn("x", priv)
        self.assertNotIn("x", pub)


class RuntimeTests(unittest.TestCase):
    def test_prove_and_verify_honest(self):
        result = compile_file(SMALL)
        x = sample_input("mlp_small")
        proof = prove(result, x)
        self.assertTrue(proof.checks_ok)
        ok, reason = verify(proof, result.ml)
        self.assertTrue(ok, reason)

    def test_reference_matches_proof(self):
        result = compile_file(SMALL)
        x = sample_input("mlp_small")
        proof = prove(result, x)
        ref = reference_output(result, x)
        self.assertEqual(proof.outputs["y"], ref["y"])

    def test_tampered_model_rejected(self):
        result = compile_file(SMALL)
        x = sample_input("mlp_small")
        proof = prove(result, x)
        tampered = _tampered_ml(result.ml)
        ok, reason = verify(proof, tampered)
        self.assertFalse(ok)
        self.assertIn("commitment", reason)

    def test_tampered_prediction_rejected(self):
        result = compile_file(SMALL)
        x = sample_input("mlp_small")
        proof = prove(result, x)
        bad = dataclasses.replace(proof)
        bad.outputs = dict(proof.outputs)
        bad.outputs["y"] = int(proof.outputs["y"]) + 1
        ok, reason = verify(bad, result.ml)
        self.assertFalse(ok)
        self.assertIn("tampered", reason)

    def test_large_proves(self):
        result = compile_file(LARGE)
        x = sample_input("mlp_large")
        proof = prove(result, x)
        self.assertTrue(proof.checks_ok)
        ok, reason = verify(proof, result.ml)
        self.assertTrue(ok, reason)


if __name__ == "__main__":
    unittest.main()
