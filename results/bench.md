# ProofIntent-ML benchmark

Measured on this machine with the Python witness/proof simulator.
Times are the mean of 5 runs.

| Model | Mode | R1CS cost | Rows | Prove (ms) | Emulated prove (ms) | Verify (ms) | Peak mem (KB) | Verify OK |
|-------|------|-----------|------|------------|---------------------|-------------|---------------|-----------|
| mlp_small | baseline | 31 | 8 | 7.01 | 19.67 | 1.16 | 13.1 | True |
| mlp_small | optimized | 22 | 8 | 7.07 | 13.54 | 1.25 | 12.0 | True |
| mlp_large | baseline | 103 | 9 | 16.11 | 71.30 | 2.25 | 34.9 | True |
| mlp_large | optimized | 20 | 4 | 16.06 | 13.46 | 1.95 | 34.5 | True |

`Emulated prove` is a cost-proportional workload (field-multiply emulations per R1CS constraint) that illustrates how constraint reduction maps to proving effort on a real backend.

## Optimization effect

| Model | Baseline R1CS | Optimized R1CS | Removed | Reduction |
|-------|---------------|----------------|---------|-----------|
| mlp_small | 31 | 22 | 9 | 29.03% |
| mlp_large | 103 | 20 | 83 | 80.58% |
