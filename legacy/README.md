# Historical source map

The following files in the parent workspace were identified as directly
related to REC-NN:

- `test/network.py`: later plain residual network.
- `test/network2.py`: encoder-decoder residual network.
- `test/main_REC.py`: later REC-NN experiment with physics-based losses.
- `test/main_EPE.py`: later direct-estimation experiment.
- `test/dataset maker.py`: channel construction and Stab-EPT residual labels.
- `test/loader_gen.py`: simulation-data loader.
- `test/data/*.npy`: compact preprocessed experiments and raw simulation data.

The GitHub implementation in `src/recnn` keeps the paper's supervised MSE
formulation and removes machine-specific paths, import-time training, repeated
copies, per-epoch images, and full-model pickle checkpoints.

The following were not included in the core implementation:

- `20250122/pytorch_ML_stab_MREPT*.py`: earlier ML-Stab-MREPT experiments.
- `PINN/`, `MREPT_Chiba_revised_202404/`, and most dated scripts in `test/`:
  later physics-informed or exploratory work.
- `__pycache__/`, generated figures, Excel logs, and `.pth` checkpoints.
