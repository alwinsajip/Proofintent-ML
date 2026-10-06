"""Command-line interface for the ProofIntent-ML compiler."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import sys
import tempfile

from .backends.circom_backend import emit_circom
from .compiler import compile_file
from .diagnostics import CompileError, SemanticError
from .runtime.inference import forward
from .runtime.proof_sim import prove, reference_output, verify


def _print_diagnostics(diags) -> None:
    for d in diags:
        print(d.format())


def _load_input(spec, raw: str | None) -> list[float]:
    if raw:
        data = json.loads(raw)
    else:
        first = next(iter(spec.inputs.values()), None)
        size = first.size if first else 1
        data = [0.0] * size
    return [float(v) for v in data]


def cmd_compile(args) -> int:
    try:
        result = compile_file(args.spec, optimize=not args.no_optimize)
    except CompileError as exc:
        print(exc.format())
        return 2
    except SemanticError as exc:
        _print_diagnostics(exc.diagnostics)
        print(f"compilation failed with {len(exc.diagnostics)} error(s)")
        return 2

    _print_diagnostics(result.diagnostics)

    if args.emit_ir in ("ml", "both"):
        print("=" * 72)
        print("ML IR")
        print("=" * 72)
        print("\n".join(result.ml.to_lines()))

    if args.emit_ir in ("proof", "both"):
        print("=" * 72)
        print("Proof IR (baseline)")
        print("=" * 72)
        print("\n".join(result.baseline.to_lines()))
        if result.passes:
            print("=" * 72)
            print("Proof IR (optimized)")
            print("=" * 72)
            print("\n".join(result.optimized.to_lines()))

    metrics = result.metrics()
    print("=" * 72)
    print("Metrics")
    print("=" * 72)
    print(f"  baseline  : rows={metrics['baseline']['constraint_rows']} "
          f"cost={metrics['baseline']['r1cs_cost']}")
    print(f"  optimized : rows={metrics['optimized']['constraint_rows']} "
          f"cost={metrics['optimized']['r1cs_cost']}")
    print(f"  removed   : rows={metrics['constraint_rows_removed']} "
          f"cost={metrics['r1cs_cost_removed']} "
          f"({metrics['r1cs_cost_reduction_pct']}%)")
    for p in metrics["passes"]:
        print(f"    - {p['pass']}: -{p['cost_removed']} r1cs "
              f"({p['before_cost']} -> {p['after_cost']})")

    if args.backend == "circom":
        outdir = args.out or os.path.join(os.path.dirname(os.path.abspath(args.spec)), "build")
        written = emit_circom(result, outdir)
        print("Generated:")
        for key, path in written.items():
            print(f"  {key}: {path}")

    if args.json:
        print(json.dumps(metrics, indent=2))
    return 0


def _emit_for(result, args):
    outdir = args.out or os.path.join(os.path.dirname(os.path.abspath(args.spec)), "build")
    return emit_circom(result, outdir)


def cmd_prove(args) -> int:
    try:
        result = compile_file(args.spec, optimize=not args.no_optimize)
    except (CompileError, SemanticError) as exc:
        if isinstance(exc, CompileError):
            print(exc.format())
        else:
            _print_diagnostics(exc.diagnostics)
        return 2

    x = _load_input(result.spec, args.input)
    ref = reference_output(result, x)
    proof = prove(result, x)

    if args.emit:
        _emit_for(result, args)

    print(f"model       : {result.ml.name}")
    print(f"input       : {x}")
    print(f"commitment  : {proof.commitment[:32]}...")
    print(f"outputs     : {json.dumps(proof.outputs, default=str)}")
    print(f"ref outputs : {json.dumps(ref, default=str)}")
    print(f"constraints : {sum(1 for c in proof.checks if c['ok'])}/{len(proof.checks)} satisfied")
    print(f"checks_ok   : {proof.checks_ok}")
    if args.proof_out:
        with open(args.proof_out, "w", encoding="utf-8") as fh:
            json.dump(proof.to_dict(), fh, indent=2)
        print(f"proof saved : {args.proof_out}")
    return 0 if proof.checks_ok else 1


def _tampered_ml(ml):
    tmp = tempfile.mkdtemp(prefix="piml_tamper_")
    basedir = ml.meta.get("basedir", os.getcwd())
    for op in ml.ops:
        if op.op != "dense":
            continue
        for key in ("weights", "bias"):
            p = op.attrs.get(key)
            if not p:
                continue
            src = p if os.path.isabs(p) else os.path.join(basedir, p)
            rel = p if not os.path.isabs(p) else os.path.basename(p)
            dst = os.path.join(tmp, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(src, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
            if isinstance(obj, dict) and "data" in obj:
                data = obj["data"]
                if isinstance(data, list) and data and isinstance(data[0], list):
                    data[0][0] = float(data[0][0]) + 1.0
                elif isinstance(data, list) and data:
                    data[0] = float(data[0]) + 1.0
            elif isinstance(obj, list) and obj:
                if isinstance(obj[0], list):
                    obj[0][0] = float(obj[0][0]) + 1.0
                else:
                    obj[0] = float(obj[0]) + 1.0
            with open(dst, "w", encoding="utf-8") as fh:
                json.dump(obj, fh)
    new_meta = dict(ml.meta)
    new_meta["basedir"] = tmp
    return dataclasses.replace(ml, meta=new_meta)


def cmd_verify(args) -> int:
    try:
        result = compile_file(args.spec, optimize=not args.no_optimize)
    except (CompileError, SemanticError) as exc:
        if isinstance(exc, CompileError):
            print(exc.format())
        else:
            _print_diagnostics(exc.diagnostics)
        return 2

    x = _load_input(result.spec, args.input)
    proof = prove(result, x)

    ok, reason = verify(proof, result.ml)
    print(f"verify (honest)          : {ok} ({reason})")

    code = 0 if ok else 1

    if args.tamper_model:
        tampered = _tampered_ml(result.ml)
        t_ok, t_reason = verify(proof, tampered)
        print(f"verify (tampered model)  : {t_ok} ({t_reason})")
        if t_ok:
            print("  ERROR: tampered model was accepted!")
            code = 1

    if args.tamper_prediction:
        bad = dataclasses.replace(proof)
        bad.outputs = dict(proof.outputs)
        key = next(iter(bad.outputs))
        bad.outputs[key] = (int(bad.outputs[key]) + 1) if isinstance(bad.outputs[key], int) else 999
        t_ok, t_reason = verify(bad, result.ml)
        print(f"verify (tampered predict): {t_ok} ({t_reason})")
        if t_ok:
            print("  ERROR: tampered prediction was accepted!")
            code = 1

    return code


def cmd_bench(args) -> int:
    from bench.run_bench import main as bench_main

    argv = []
    if args.outdir:
        argv += ["--outdir", args.outdir]
    return bench_main(argv)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pimlc", description="ProofIntent-ML compiler (semantic ZK verification of ML inference)"
    )
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("compile", help="compile a .piml spec")
    c.add_argument("spec")
    c.add_argument("--emit-ir", choices=["ml", "proof", "both"], default=None)
    c.add_argument("--no-optimize", action="store_true")
    c.add_argument("--backend", choices=["circom"], default=None)
    c.add_argument("--out", default=None)
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_compile)

    pr = sub.add_parser("prove", help="compile and produce a proof")
    pr.add_argument("spec")
    pr.add_argument("--input", default=None, help="JSON list of input values")
    pr.add_argument("--no-optimize", action="store_true")
    pr.add_argument("--emit", action="store_true", help="emit Circom artifacts")
    pr.add_argument("--out", default=None)
    pr.add_argument("--proof-out", default=None)
    pr.set_defaults(func=cmd_prove)

    v = sub.add_parser("verify", help="compile, prove and verify")
    v.add_argument("spec")
    v.add_argument("--input", default=None)
    v.add_argument("--no-optimize", action="store_true")
    v.add_argument("--tamper-model", action="store_true")
    v.add_argument("--tamper-prediction", action="store_true")
    v.add_argument("--out", default=None)
    v.set_defaults(func=cmd_verify)

    b = sub.add_parser("bench", help="run the baseline-vs-optimized benchmark")
    b.add_argument("--outdir", default=None)
    b.set_defaults(func=cmd_bench)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
