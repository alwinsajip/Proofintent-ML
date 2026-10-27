# ProofIntent-ML benchmark

Measured on this machine with the Python witness/proof simulator.
Times are the mean of 5 runs.

| Model | Mode | R1CS cost | Rows | Prove (ms) | Emulated prove (ms) | Verify (ms) | Peak mem (KB) | Verify OK |
|-------|------|-----------|------|------------|---------------------|-------------|---------------|-----------|
| mlp_small | baseline | 31 | 8 | 5.54 | 22.90 | 0.89 | 13.4 | True |
| mlp_small | optimized | 22 | 8 | 7.59 | 17.64 | 1.38 | 12.0 | True |
| mlp_large | baseline | 103 | 9 | 13.95 | 84.84 | 1.40 | 34.9 | True |
| mlp_large | optimized | 20 | 4 | 13.41 | 15.42 | 1.47 | 34.6 | True |

`Emulated prove` is a cost-proportional workload (field-multiply emulations per R1CS constraint) that illustrates how constraint reduction maps to proving effort on a real backend.

## Optimization effect

| Model | Baseline R1CS | Optimized R1CS | Removed | Reduction |
|-------|---------------|----------------|---------|-----------|
| mlp_small | 31 | 22 | 9 | 29.03% |
| mlp_large | 103 | 20 | 83 | 80.58% |
