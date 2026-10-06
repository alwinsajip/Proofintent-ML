"""Baseline-vs-optimized benchmark.

For each example spec we compile twice -- with all passes disabled (baseline)
and with the full pass pipeline (optimized) -- then measure, for both circuits:

* R1CS constraint cost (headline "constraint count") and logical rows;
* proving time (witness generation + constraint evaluation);
* verification time;
* peak memory during proving.

Results are written to ``results/bench.csv`` and ``results/bench.md``.
Every figure is measured, never assumed.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import tracemalloc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from proofintent.compiler import compile_file  # noqa: E402
from proofintent.runtime.proof_sim import prove, verify  # noqa: E402

EXAMPLES = os.path.join(ROOT, "examples")
SPECS = [
    ("mlp_small", os.path.join(EXAMPLES, "mlp_small.piml")),
    ("mlp_large", os.path.join(EXAMPLES, "mlp_large.piml")),
]
REPEATS = 5


def _load_input(name: str):
    path = os.path.join(EXAMPLES, "inputs", f"{name}.json")
    with open(path, "r", encoding="utf-8") as fh:
        return [float(v) for v in json.load(fh)]


def _time_prove(result, x):
    tracemalloc.start()
    t0 = time.perf_counter()
    proof = prove(result, x)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return proof, elapsed, peak


def _time_verify(proof, result):
    t0 = time.perf_counter()
    ok, _ = verify(proof, result.ml)
    elapsed = time.perf_counter() - t0
    return ok, elapsed


# A real SNARK prover's cost grows with the number of R1CS constraints. The
# Python simulator has no field arithmetic, so we additionally measure a
# cost-proportional workload (UNIT field-multiply emulations per constraint)
# to illustrate how the constraint reduction translates into proving effort.
UNIT = 4000


def _emulated_prove_ms(result) -> float:
    t0 = time.perf_counter()
    acc = 1
    for c in result.optimized.constraints:
        for _ in range(c.cost * UNIT):
            acc = (acc * 1103515245 + 12345) & 0xFFFFFFFF
    dt = time.perf_counter() - t0
    _ = acc
    return dt * 1000.0


def run():
    rows = []
    for name, spec_path in SPECS:
        x = _load_input(name)
        for mode, optimize in (("baseline", False), ("optimized", True)):
            result = compile_file(spec_path, optimize=optimize)

            prove_times = []
            verify_times = []
            peaks = []
            proof = None
            checks_ok = True
            for _ in range(REPEATS):
                proof, t, peak = _time_prove(result, x)
                prove_times.append(t)
                peaks.append(peak)
                ok, vt = _time_verify(proof, result)
                verify_times.append(vt)
                checks_ok = checks_ok and ok

            metrics = result.metrics()
            emulated = [_emulated_prove_ms(result) for _ in range(REPEATS)]
            rows.append(
                {
                    "model": name,
                    "mode": mode,
                    "constraint_rows": metrics[mode]["constraint_rows"],
                    "r1cs_cost": metrics[mode]["r1cs_cost"],
                    "prove_time_ms": statistics.mean(prove_times) * 1000.0,
                    "emulated_prove_ms": statistics.mean(emulated),
                    "verify_time_ms": statistics.mean(verify_times) * 1000.0,
                    "peak_memory_kb": max(peaks) / 1024.0,
                    "verify_ok": checks_ok,
                }
            )
    return rows


def _write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    headers = [
        "model",
        "mode",
        "constraint_rows",
        "r1cs_cost",
        "prove_time_ms",
        "emulated_prove_ms",
        "verify_time_ms",
        "peak_memory_kb",
        "verify_ok",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(headers) + "\n")
        for r in rows:
            fh.write(
                ",".join(
                    str(round(r[h], 4)) if isinstance(r[h], float) else str(r[h])
                    for h in headers
                )
                + "\n"
            )


def _write_md(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "# ProofIntent-ML benchmark",
        "",
        "Measured on this machine with the Python witness/proof simulator.",
        "Times are the mean of %d runs." % REPEATS,
        "",
        "| Model | Mode | R1CS cost | Rows | Prove (ms) | Emulated prove (ms) | Verify (ms) | Peak mem (KB) | Verify OK |",
        "|-------|------|-----------|------|------------|---------------------|-------------|---------------|-----------|",
    ]
    for r in rows:
        lines.append(
            "| {model} | {mode} | {r1cs_cost} | {constraint_rows} | "
            "{prove_time_ms:.2f} | {emulated_prove_ms:.2f} | {verify_time_ms:.2f} | "
            "{peak_memory_kb:.1f} | "
            "{verify_ok} |".format(**r)
        )
    lines.append("")
    lines.append(
        "`Emulated prove` is a cost-proportional workload (field-multiply "
        "emulations per R1CS constraint) that illustrates how constraint "
        "reduction maps to proving effort on a real backend."
    )
    lines.append("")
    lines.append("## Optimization effect")
    lines.append("")
    lines.append("| Model | Baseline R1CS | Optimized R1CS | Removed | Reduction |")
    lines.append("|-------|---------------|----------------|---------|-----------|")
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], {})[r["mode"]] = r
    for model, modes in by_model.items():
        b = modes["baseline"]["r1cs_cost"]
        o = modes["optimized"]["r1cs_cost"]
        pct = 100.0 * (b - o) / b if b else 0.0
        lines.append(f"| {model} | {b} | {o} | {b - o} | {pct:.2f}% |")
    lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ProofIntent-ML benchmark")
    parser.add_argument("--outdir", default=os.path.join(ROOT, "results"))
    args = parser.parse_args(argv)

    rows = run()
    csv_path = os.path.join(args.outdir, "bench.csv")
    md_path = os.path.join(args.outdir, "bench.md")
    _write_csv(rows, csv_path)
    _write_md(rows, md_path)

    for r in rows:
        print(
            f"{r['model']:<10} {r['mode']:<10} "
            f"r1cs={r['r1cs_cost']:<5} rows={r['constraint_rows']:<3} "
            f"prove={r['prove_time_ms']:7.2f}ms emu_prove={r['emulated_prove_ms']:8.2f}ms "
            f"verify={r['verify_time_ms']:7.2f}ms "
            f"mem={r['peak_memory_kb']:7.1f}KB ok={r['verify_ok']}"
        )
    print(f"\nwrote {csv_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
