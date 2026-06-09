# REC-NN

Paper-aligned PyTorch implementation of the Reconstruction Error Compensation
Neural Network (REC-NN) for conductivity reconstruction in magnetic resonance
electrical property tomography (MREPT).

REC-NN learns the residual between a Stabilized-EPT (Stab-EPT)
reconstruction and the ground-truth conductivity:

```text
sigma_REC = sigma_Stab + Delta_sigma
```

The repository also includes the direct electrical-property estimation
baseline (EPE-NN) and two network variants:

- `plain`: residual blocks without spatial downsampling (REC-NN1/EPE-NN1).
- `encoder`: residual encoder-decoder blocks (REC-NN2/EPE-NN2).

## Repository layout

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
|-- .gitignore
|-- CITATION.cff
|-- LICENSE
|-- pyproject.toml
`-- requirements.txt
```

## Data format

The historical preprocessed arrays have shape `(N, C, H, W)` with the
following channels:

| Channel | Quantity |
| --- | --- |
| 0 | Transceive phase |
| 1 | Phase gradient in x |
| 2 | Phase gradient in y |
| 3 | Phase Laplacian |
| 4 | Reciprocal Stab-EPT conductivity (`gamma_Stab`) |
| 5 | Conductivity residual (`sigma_gt - sigma_Stab`) |
| 6 | Stab-EPT conductivity |
| 7 | Ground-truth conductivity |
| 8-10 | Ground-truth derivative terms used only by later analysis |

REC-NN uses channels `0:5`; EPE-NN uses channels `0:4`. The included arrays
are compact examples recovered from the historical workspace, not the complete
paper datasets. See `data/README.md`.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Training

Train the paper-aligned residual model:

```bash
python -m recnn.train --data data/samples/AIG_0.01.npy --mode rec --architecture plain
```

Train the direct-estimation baseline:

```bash
python -m recnn.train --data data/samples/AIG_0.01.npy --mode epe --architecture plain
```

The sample files contain very few images and are intended for code validation,
not for reproducing the published numerical results. Use `--epochs`,
`--batch-size`, and `--output-dir` to configure a run.

## Evaluation

```bash
python -m recnn.evaluate \
  --data data/samples/circular.npy \
  --checkpoint outputs/rec_plain/best.pt
```

## Historical scope

The source workspace also contains ML-Stab-MREPT experiments and later
physics-informed neural network work. They are intentionally excluded from the
core package. `legacy/README.md` records which files were identified and why.

## Papers

This implementation was organized from the two conference papers listed in
`references/README.md`. The PDF files are not redistributed in this repository.

## License

The code is released under the MIT License. Dataset and paper redistribution
rights must be checked separately before publishing the repository.
