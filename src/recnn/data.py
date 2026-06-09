from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset


PathLike = Union[str, Path]

PHASE = 0
GRADIENT_X = 1
GRADIENT_Y = 2
LAPLACIAN = 3
SIGMA_STAB = 4
RESIDUAL = 5
SIGMA_GT = 6


def _load_array(path: PathLike) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim != 4 or array.shape[1] != 7:
        raise ValueError(f"{path}: expected a paper-format array shaped (N, 7, H, W).")
    return array


class RECArrayDataset(Dataset):
    """Load the 7-channel arrays used by the REC-NN studies."""

    def __init__(
        self, paths: Union[PathLike, Sequence[PathLike]], mode: str = "rec"
    ):
        if mode not in {"rec", "epe"}:
            raise ValueError("mode must be 'rec' or 'epe'")
        if isinstance(paths, (str, Path)):
            paths = [paths]

        array = np.concatenate([_load_array(path) for path in paths], axis=0)
        base_features = array[:, :4]
        sigma_stab = array[:, SIGMA_STAB : SIGMA_STAB + 1]

        if mode == "rec":
            inputs = np.concatenate((base_features, sigma_stab), axis=1)
            targets = array[:, RESIDUAL : RESIDUAL + 1]
        else:
            inputs = base_features
            targets = array[:, SIGMA_GT : SIGMA_GT + 1]

        self.inputs = torch.from_numpy(inputs.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))
        self.sigma_stab = torch.from_numpy(sigma_stab.astype(np.float32))
        self.sigma_gt = torch.from_numpy(
            array[:, SIGMA_GT : SIGMA_GT + 1].astype(np.float32)
        )

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int):
        return (
            self.inputs[index],
            self.targets[index],
            self.sigma_stab[index],
            self.sigma_gt[index],
        )
