from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ChannelLayout:
    phase: int = 0
    gradient_x: int = 1
    gradient_y: int = 2
    laplacian: int = 3
    gamma_stab: int = 4
    residual: int = 5
    sigma_stab: int = 6
    sigma_gt: int = 7


CHANNELS = ChannelLayout()


class RECArrayDataset(Dataset):
    def __init__(self, path: str | Path, mode: str = "rec"):
        array = np.load(path, allow_pickle=False)
        if array.ndim == 3:
            array = array[None, ...]
        if array.ndim != 4 or array.shape[1] < 8:
            raise ValueError("Expected an array shaped (N, at least 8, H, W).")
        if mode not in {"rec", "epe"}:
            raise ValueError("mode must be 'rec' or 'epe'")

        input_channels = 5 if mode == "rec" else 4
        target_channel = CHANNELS.residual if mode == "rec" else CHANNELS.sigma_gt
        self.inputs = torch.from_numpy(array[:, :input_channels].astype(np.float32))
        self.targets = torch.from_numpy(
            array[:, target_channel : target_channel + 1].astype(np.float32)
        )
        self.sigma_stab = torch.from_numpy(
            array[:, CHANNELS.sigma_stab : CHANNELS.sigma_stab + 1].astype(np.float32)
        )
        self.sigma_gt = torch.from_numpy(
            array[:, CHANNELS.sigma_gt : CHANNELS.sigma_gt + 1].astype(np.float32)
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
