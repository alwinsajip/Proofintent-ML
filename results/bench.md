# ProofIntent-ML benchmark

Measured on this machine with the Python witness/proof simulator.
Times are the mean of 5 runs.

| Model | Mode | R1CS cost | Rows | Prove (ms) | Emulated prove (ms) | Verify (ms) | Peak mem (KB) | Verify OK |
|-------|------|-----------|------|------------|---------------------|-------------|---------------|-----------|
| mlp_small | baseline | 31 | 8 | 2.06 | 9.48 | 0.25 | 13.5 | True |
| mlp_small | optimized | 22 | 8 | 2.08 | 9.39 | 0.26 | 12.0 | True |
| mlp_large | baseline | 103 | 9 | 5.95 | 33.17 | 0.43 | 34.6 | True |
| mlp_large | optimized | 20 | 4 | 5.58 | 6.33 | 0.43 | 34.3 | True |

`Emulated prove` is a cost-proportional workload (field-multiply emulations per R1CS constraint) that illustrates how constraint reduction maps to proving effort on a real backend.

## Optimization effect

| Model | Baseline R1CS | Optimized R1CS | Removed | Reduction |
|-------|---------------|----------------|---------|-----------|
| mlp_small | 31 | 22 | 9 | 29.03% |
| mlp_large | 103 | 20 | 83 | 80.58% |
