import numpy as np
import torch

from recnn.data import RECArrayDataset
from recnn.model import build_model


def test_dataset_and_models(tmp_path):
    data = np.random.default_rng(0).normal(size=(2, 11, 32, 32)).astype(np.float32)
    path = tmp_path / "sample.npy"
    np.save(path, data)

    for mode, channels in [("rec", 5), ("epe", 4)]:
        dataset = RECArrayDataset(path, mode=mode)
        features, _, sigma_stab, sigma_gt = dataset[0]
        assert features.shape == (channels, 32, 32)

        for architecture in ["plain", "encoder"]:
            model = build_model(mode, architecture)
            output = model(features[None], sigma_stab[None])
            assert output.shape == sigma_gt[None].shape
            assert torch.isfinite(output).all()
