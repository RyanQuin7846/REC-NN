# REC-NN

PyTorch implementation of the Reconstruction Error Compensation Neural Network
(REC-NN) for conductivity reconstruction in Magnetic Resonance Electrical
Property Tomography (MREPT).

This repository is organized around two conference papers:

1. [REC-NN: A Reconstruction Error Compensation Neural Network for Magnetic Resonance Electrical Property Tomography (MREPT)](https://ieeexplore.ieee.org/document/10340423)
2. [Is Laplacian Indispensable to Magnetic Resonance Electrical Property Tomography (MREPT)? An Analysis from the Perspective of Reconstruction Error Compensation Neural Networks](https://ieeexplore.ieee.org/document/10265641)

## Background

MREPT estimates tissue electrical properties from MRI measurements. Analytical
reconstruction methods are physically interpretable, but numerical
differentiation is sensitive to noise and model assumptions can produce
artifacts or reduce tissue contrast.

The papers propose REC-NN as a hybrid strategy: Stab-EPT first provides a
physics-based conductivity reconstruction, and a neural network then predicts
the remaining reconstruction error. This differs from an end-to-end electrical
property estimation network (EPE-NN), which directly predicts conductivity.

## Method

Let `sigma_stab` be the conductivity reconstructed by Stab-EPT and `sigma_gt`
the ground-truth conductivity. The true reconstruction error is

```text
Delta_sigma = sigma_gt - sigma_stab
```

REC-NN estimates this error and compensates the Stab-EPT result:

```text
Delta_sigma_hat = NN(phi, grad_x(phi), grad_y(phi), laplacian(phi), sigma_stab)
sigma_rec = sigma_stab + Delta_sigma_hat
```

The supervised objective described in the papers is

```text
L_REC = MSE(Delta_sigma_hat, Delta_sigma)
```

The EPE-NN baseline uses the phase and derivative features without
`sigma_stab`, and directly estimates `sigma_gt`.

### Input features

- Transceive phase
- First-order phase derivatives in the x and y directions
- Phase Laplacian
- Stab-EPT conductivity for REC-NN only

### Network variants

- **REC-NN1 / EPE-NN1 (`plain`)**: eleven residual blocks without spatial
  downsampling.
- **REC-NN2 / EPE-NN2 (`encoder`)**: residual blocks combined with an
  encoder-decoder structure.

The EMBC paper reports that the plain REC-NN1 obtained the best reconstruction
on the tested Duke slice. The encoder-decoder structure improved estimates in
some homogeneous regions, but generalized less effectively to tumor
conductivity distributions that were absent from training data.

## Findings From The Papers

### Reconstruction error compensation

The EMBC study evaluated Stab-EPT, EPE-NN1/2, and REC-NN1/2 using digital
human head data from Duke and Ella. The dataset contained 30 brain slices, with
additional Ella test samples containing artificial tumors of 2, 4, 6, and
10 mm.

For the reported Duke example, REC-NN achieved higher SSIM than the
corresponding EPE model:

| Comparison | SSIM |
| --- | ---: |
| EPE-NN1 | 0.961 |
| REC-NN1 | 0.977 |
| EPE-NN2 | 0.962 |
| REC-NN2 | 0.968 |

The study also found that REC-NN improved tissue contrast over direct
estimation for unseen tumor samples, although tumor conductivity remained
underestimated. Overcompensation and generalization to unseen electrical
property distributions were identified as open problems.

### Role of the Laplacian

The URSI GASS study trained on artificial irregular geometries (AIG) and tested
on a dissimilar digital human head (DHH) dataset. It reported that REC-NN
improved average SSIM by 23.4% and reduced average NRMSE by 39.1% relative to
Stab-EPT, while EPE-NN performed better than REC-NN in that cross-dataset
experiment.

Gradient analysis found the raw Laplacian to have the lowest local sensitivity
among the REC-NN inputs. However, removing it caused loss of boundary and shape
information. Filtering extreme Laplacian values sharpened boundaries and
reduced overcompensation up to an appropriate threshold.

The resulting interpretation is that the Laplacian has a double-edged role:
it is indispensable for boundary and shape information, but unprocessed
outliers can negatively affect numerical compensation and generalization.

## Repository Layout

```text
REC-NN/
|-- data/
|   |-- README.md
|   `-- samples/
|-- legacy/
|   `-- README.md
|-- references/
|   `-- README.md
|-- src/recnn/
|   |-- data.py
|   |-- evaluate.py
|   |-- metrics.py
|   |-- model.py
|   `-- train.py
|-- tests/
|   `-- test_smoke.py
|-- CITATION.cff
|-- LICENSE
|-- pyproject.toml
`-- requirements.txt
```

## Data Format

The recovered preprocessed arrays have shape `(N, C, H, W)`:

| Channel | Quantity |
| --- | --- |
| 0 | Transceive phase |
| 1 | Phase gradient in x |
| 2 | Phase gradient in y |
| 3 | Phase Laplacian |
| 4 | Reciprocal Stab-EPT conductivity (`gamma_stab`) |
| 5 | Conductivity residual (`sigma_gt - sigma_stab`) |
| 6 | Stab-EPT conductivity |
| 7 | Ground-truth conductivity |
| 8-10 | Ground-truth derivative terms used by later analyses |

REC-NN uses channels `0:5`; EPE-NN uses channels `0:4`. The included files are
small examples recovered from the historical workspace. They are provided for
format inspection and code validation, not as the complete AIG, Duke, Ella, or
tumor datasets used to produce the published results.

## Installation

The original studies used Python 3.9.12, PyTorch 1.12.1, and CUDA 11.3.
The cleaned implementation also supports newer compatible PyTorch releases.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Training

Train REC-NN1:

```bash
python -m recnn.train \
  --data data/samples/AIG_0.01.npy \
  --mode rec \
  --architecture plain
```

Train REC-NN2:

```bash
python -m recnn.train \
  --data data/samples/AIG_0.01.npy \
  --mode rec \
  --architecture encoder
```

Train the direct-estimation baseline:

```bash
python -m recnn.train \
  --data data/samples/AIG_0.01.npy \
  --mode epe \
  --architecture plain
```

Use `--epochs`, `--batch-size`, `--learning-rate`, and `--output-dir` to
configure a run.

## Evaluation

```bash
python -m recnn.evaluate \
  --data data/samples/circular.npy \
  --checkpoint outputs/rec_plain/best.pt
```

The evaluation command reports NRMSE and PSNR.

## Reproducibility Scope

This repository is a cleaned, paper-aligned reconstruction of the historical
REC-NN code. It preserves the residual-compensation formulation, EPE baseline,
input channels, and two network families described in the papers.

It does not currently include the complete proprietary/simulation datasets,
original trained weights, or every preprocessing setting required to reproduce
the exact tables and figures in the publications. Later PINN experiments and
unrelated exploratory scripts were intentionally excluded. See
`legacy/README.md` for the historical source map.

## Citation

Please cite both papers when using this repository:

```bibtex
@inproceedings{qin2023recnn,
  author    = {Ruian Qin and Adan Jafet Garcia Inda and Zhongchao Zhou and
               Yukihiro Enomoto and Tianyi Yang and Nevrez Imamoglu and
               Jose Gomez-Tames and Shaoying Huang and Wenwei Yu},
  title     = {{REC-NN}: A Reconstruction Error Compensation Neural Network
               for Magnetic Resonance Electrical Property Tomography ({MREPT})},
  booktitle = {2023 45th Annual International Conference of the IEEE
               Engineering in Medicine and Biology Society (EMBC)},
  pages     = {1--4},
  year      = {2023},
  doi       = {10.1109/EMBC40787.2023.10340423}
}

@inproceedings{qin2023laplacian,
  author    = {Ruian Qin and Adan Jafet Garcia Inda and Zhongchao Zhou and
               Yukihiro Enomoto and Tianyi Yang and Nevrez Imamoglu and
               Jose Gomez-Tames and Shaoying Huang and Wenwei Yu},
  title     = {Is Laplacian Indispensable to Magnetic Resonance Electrical
               Property Tomography ({MREPT})? An Analysis from the Perspective
               of Reconstruction Error Compensation Neural Networks},
  booktitle = {2023 XXXVth General Assembly and Scientific Symposium of the
               International Union of Radio Science (URSI GASS)},
  pages     = {1--4},
  year      = {2023},
  doi       = {10.23919/URSIGASS57860.2023.10265641}
}
```

IEEE Xplore:

- https://ieeexplore.ieee.org/document/10340423
- https://ieeexplore.ieee.org/document/10265641

## License

The code is released under the MIT License. Dataset and paper redistribution
rights must be considered separately.
