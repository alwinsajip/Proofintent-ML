# ProofIntent-ML

ProofIntent-ML is a compiler and optimization framework for semantic Zero-Knowledge (ZK) verification of Machine Learning (ML) inference. It allows developers to specify ML models, tensor inputs/outputs, and proof constraints in a high-level domain-specific language (`.piml`), compile them into optimized constraint representations (R1CS), emit Circom 2.0 circuits, and simulate proof generation and verification.

---

## Key Features

- **Domain-Specific Language (`.piml`)**: Custom high-level syntax for declaring neural network layers (`dense`, `relu`, `argmax`, `softmax`), input ranges, class outputs, and proof intents (`model_is`, `inference`, `confidence`, `range`).
- **Multi-Stage Compiler Pipeline**:
  - **Frontend**: Lexer, parser, and semantic analyzer with source location diagnostics.
  - **Intermediate Representations**: Lowers AST to Machine Learning IR (`MLModule`) and Proof Constraint IR (`ProofModule`).
- **Domain-Specific Optimization Passes**:
  - **Constant Propagation**: Eliminates pruned or dead zero-weight network channels/layers.
  - **Dead Constraint Elimination**: Removes unnecessary constraints based on intent requirements (e.g., omitting softmax if confidence intent is unused).
  - **Fixed-Point Quantization**: Annotates signals and reduces R1CS arithmetic constraint costs.
- **Circom 2.0 Code Generator**: Emits valid Circom 2.0 template code (`Dense`, `ReLU`, `ArgMax`, `Confidence`, `RangeCheck`, `Main`) matching the optimized Proof IR.
- **Proof & Verification Simulator**: Built-in Python runtime simulator for witness generation, model weight commitment hashing (SHA-256), proof verification, and tamper detection testing.
- **Automated Benchmarks**: Built-in benchmark suite measuring R1CS constraint counts, proving/verification latency, memory usage, and constraint reduction.

---

## Technologies Used

- **Python**: 3.7+ (Standard library only: `argparse`, `dataclasses`, `json`, `random`, `statistics`, `tracemalloc`, `unittest`)
- **Circom 2.0**: Target zero-knowledge circuit language format for backend code generation

---

## Project Structure

```
Proofintent-ML/
├── bench/
│   ├── __init__.py
│   └── run_bench.py            # Benchmark suite runner measuring performance & constraint reduction
├── examples/
│   ├── inputs/                 # Sample input JSON files (mlp_small.json, mlp_large.json)
│   ├── weights/                # Deterministic weight and bias JSON files
│   ├── make_models.py          # Script to generate example parameters and inputs
│   ├── mlp_large.piml          # Example specification featuring a pruned network layer
│   └── mlp_small.piml          # Small MLP model specification example
├── proofintent/
│   ├── __init__.py             # Main package exports (compile_file, compile_source, etc.)
│   ├── __main__.py             # Package execution entry point (`python -m proofintent`)
│   ├── ast_nodes.py            # AST node definitions
│   ├── cli.py                  # CLI implementation (`pimlc`)
│   ├── compiler.py             # Compilation pipeline driver
│   ├── diagnostics.py          # Error handling and diagnostic formatting
│   ├── lexer.py                # Lexical analyzer for .piml source code
│   ├── lower.py                # Lowering AST to ML IR and Proof IR
│   ├── ml_ir.py                # Machine Learning Intermediate Representation
│   ├── parser.py               # Parser for .piml grammar
│   ├── proof_ir.py             # Proof Intermediate Representation and R1CS metric summaries
│   ├── semantic.py             # Type checker and semantic validation
│   ├── weights.py              # Weight loading utilities
│   ├── backends/
│   │   ├── __init__.py
│   │   └── circom_backend.py   # Circom 2.0 code emitter and build artifact generator
│   ├── opt/
│   │   ├── __init__.py
│   │   ├── constant_prop.py    # Constant propagation optimization pass
│   │   ├── dead_constraint.py  # Dead constraint elimination optimization pass
│   │   └── quantize.py         # Fixed-point quantization optimization pass
│   └── runtime/
│       ├── __init__.py
│       ├── inference.py        # Reference forward pass runner
│       ├── proof_sim.py        # Proof simulation and verification runner
│       └── witness.py          # Witness signal calculation
├── results/
│   ├── bench.csv               # CSV file containing benchmark output metrics
│   └── bench.md                # Markdown summary of baseline vs. optimized performance
└── tests/
    ├── __init__.py
    ├── test_frontend.py        # Lexer, parser, and AST unit tests
    ├── test_pipeline.py        # Integration tests for lowering, optimization, backend, and runtime
    └── test_semantic.py        # Semantic checker unit tests
```

---

## Prerequisites

- **Python**: Version 3.7 or higher (tested on Python 3.12).
- **Dependencies**: None. The project relies strictly on the Python Standard Library.
- *(Optional)* **Circom 2.0.0+**: Required only if you intend to compile the emitted `.circom` circuit files with the Circom toolchain.

---

## Setup & Installation

1. Clone or download the repository into your workspace.
2. No third-party packages or virtual environment installations are required.
3. *(Optional)* Regenerate or initialize sample model weights and inputs:
   ```bash
   python examples/make_models.py
   ```

---

## Usage & Commands

The project provides a CLI accessible via `python -m proofintent` (or `pimlc`).

### 1. Compile a `.piml` Specification

Compile a DSL file to ML IR, Proof IR, and emit Circom circuits:

```bash
python -m proofintent compile examples/mlp_small.piml --emit-ir both --backend circom --out build/
```

Options:
- `--emit-ir {ml,proof,both}`: Print the specified intermediate representation(s) to stdout.
- `--no-optimize`: Disable optimization passes.
- `--backend {circom}`: Emit Circom circuit artifacts to the build directory.
- `--out OUT`: Target directory for generated Circom code and IR JSON manifests.
- `--json`: Output compilation metrics in JSON format.

### 2. Generate Proof Signals

Simulate proof generation and witness evaluation for a given input:

```bash
python -m proofintent prove examples/mlp_small.piml --input "[0.1, -0.2, 0.5, 0.8]"
```

Options:
- `--input JSON_ARRAY`: JSON array string representing input values.
- `--proof-out PATH`: Save the generated proof output to a JSON file.
- `--emit`: Generate Circom artifacts during proof execution.

### 3. Verify a Proof & Run Tamper Checks

Verify that proof constraint checks are satisfied and test tamper resilience:

```bash
python -m proofintent verify examples/mlp_small.piml --input "[0.1, -0.2, 0.5, 0.8]" --tamper-model --tamper-prediction
```

Options:
- `--tamper-model`: Test rejection when model weights are modified after proof generation.
- `--tamper-prediction`: Test rejection when the output prediction is modified.

### 4. Run Benchmarks

Measure baseline vs. optimized R1CS constraint counts, proving times, and memory usage:

```bash
python -m proofintent bench
```

Or run the benchmark script directly:
```bash
python bench/run_bench.py --outdir results
```

---

## Python API Usage

ProofIntent-ML can also be imported directly in Python scripts:

```python
from proofintent import compile_file
from proofintent.runtime.proof_sim import prove, verify

# Compile file with optimization passes enabled
result = compile_file("examples/mlp_small.piml", optimize=True)

# Generate proof simulation with input vector
input_vector = [0.1, -0.2, 0.5, 0.8]
proof = prove(result, input_vector)

# Verify proof against the compiled ML module
is_valid, reason = verify(proof, result.ml)
print(f"Verification result: {is_valid} ({reason})")
```

---

## Testing

Run the unit and integration test suite using Python's built-in test runner:

```bash
python -m unittest discover -s tests
```

The test suite covers:
- Frontend tokenization, parsing, syntax error handling, and formatting.
- Semantic type checks, shape mismatches, and intent validation.
- Pipeline lowering, pass correctness (ensuring optimization passes never increase constraint costs), Circom output generation, witness evaluation, and tamper rejection.

---

## Configuration & Environment Variables

- No environment variables are required.
- Configuration is specified via CLI arguments or within `.piml` specification files.

---

## Database Information

Not applicable (this project does not use a database).

---

## Known Limitations

- **Supported Layers**: Currently supports Multilayer Perceptron (MLP) components (`dense`, `relu`, `argmax`, `softmax`, `const`). Convolutional, Recurrent, or Transformer layer types are not currently implemented.
- **Prover Integration**: Features a native Python witness simulator and Circom 2.0 generator; direct automated execution of external C++/Rust SNARK provers (such as `snarkjs` or `groth16`) is not bundled within the Python runtime.

---

## Future Improvements

- Support for additional neural network architectures and layer types (e.g., Conv2D, MaxPool, LayerNorm, Attention mechanisms).
- Native execution bridge for external SNARK backends (`snarkjs`, Groth16, PLONK).
- Integration of cryptographic hash functions (such as Poseidon) for on-chain model commitment verification.

---

## Contributors / Authors

Not specified in the codebase.
