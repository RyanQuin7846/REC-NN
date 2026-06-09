# Historical source map

The public implementation was re-audited after additional historical files
were recovered. The closest paper-era sources are:

- `REC-NN v2/network.py`: REC-NN1 residual blocks and channel widths.
- `REC-NN v2/network2.py`: REC-NN2 downsampling/upsampling residual blocks.
- `REC-NN v2/main.py`: supervised MSE training, Adam settings, weight
  initialization, StepLR schedule, and AIG-to-DHH experiment variants.
- `REC-NN v2/test.py`: REC/EPE inference variants and Laplacian ablations.
- `REC-NN v2/data/`: original 7-channel AIG, Duke, Ella, and tumor arrays.

The paper-era 7-channel layout is:

```text
phase, gradient_x, gradient_y, laplacian, sigma_stab,
sigma_gt - sigma_stab, sigma_gt
```

Machine-specific paths, import-time training, generated figures, full-model
pickle checkpoints, and duplicate exploratory scripts are intentionally
excluded from the GitHub implementation.
