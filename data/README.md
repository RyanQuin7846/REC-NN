# Data

The repository contains a compact supervised-learning subset selected from the
newly recovered `REC-NN v2/data` files:

```text
data/
|-- AIG/
|   `-- aig_subset.npy
`-- DHH/
    |-- Duke/
    |   `-- duke.npy
    |-- Ella/
    |   `-- ella.npy
    `-- Tumor/
        |-- 2_mm.npy
        |-- 4_mm.npy
        |-- 6_mm.npy
        `-- 10_mm.npy
```

| Dataset | Samples | Selection |
| --- | ---: | --- |
| AIG | 480 | 16 evenly spaced finite cases from each of 30 geometries |
| DHH/Duke | 392 | All recovered paper-format samples |
| DHH/Ella | 294 | All recovered paper-format samples |
| DHH/Tumor | 80 | All 2, 4, 6, and 10 mm samples |

Every file has shape `(N, 7, 32, 32)`, uses float32 values, and follows:

```text
phase, gradient_x, gradient_y, laplacian, sigma_stab,
sigma_gt - sigma_stab, sigma_gt
```

The loader accepts either `.npy` files or directories. Directories are searched
recursively, so `--data data/DHH` loads Duke, Ella, and Tumor files.

This subset is sufficient for exercising supervised training and evaluation,
but it is smaller than the complete AIG dataset used in the studies.
