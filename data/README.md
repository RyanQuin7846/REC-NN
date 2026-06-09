# Data

`samples/` contains four compact paper-format arrays selected from the newly
recovered `REC-NN v2/data` directory:

- `aig_sample.npy`: artificial irregular geometry sample.
- `duke_sample.npy`: Duke digital human head sample.
- `ella_sample.npy`: Ella digital human head sample.
- `tumor_4mm_sample.npy`: Ella sample with an artificial 4 mm tumor.

Each file has shape `(1, 7, 32, 32)` and uses float32 values. The channels are:

```text
phase, gradient_x, gradient_y, laplacian, sigma_stab,
sigma_gt - sigma_stab, sigma_gt
```

These files are for format inspection and smoke testing. They are not the
complete datasets used to produce the paper results.
