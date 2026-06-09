import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from recnn.data import RECArrayDataset
from recnn.model import (
    DownsampleResidualBlock,
    ResidualBlock,
    UpsampleResidualBlock,
    build_model,
)


class RECNNTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "sample.npy"
        data = np.random.default_rng(0).normal(
            size=(2, 7, 32, 32)
        ).astype(np.float32)
        data[:, 5] = data[:, 6] - data[:, 4]
        np.save(self.path, data)
        self.data = data

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dataset_channels(self):
        rec = RECArrayDataset(self.path, mode="rec")
        epe = RECArrayDataset(self.path, mode="epe")
        self.assertEqual(rec.inputs.shape, (2, 5, 32, 32))
        self.assertEqual(epe.inputs.shape, (2, 4, 32, 32))
        np.testing.assert_allclose(rec.inputs[:, 4], self.data[:, 4])
        np.testing.assert_allclose(rec.targets[:, 0], self.data[:, 5])
        np.testing.assert_allclose(epe.targets[:, 0], self.data[:, 6])

    def test_dataset_rejects_nonpaper_layout(self):
        invalid = Path(self.temp_dir.name) / "invalid.npy"
        np.save(invalid, np.zeros((1, 11, 32, 32), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "paper-format"):
            RECArrayDataset(invalid)

    def test_model_forward(self):
        rec = RECArrayDataset(self.path, mode="rec")
        features, _, sigma_stab, sigma_gt = rec[0]
        for architecture in ("recnn1", "recnn2"):
            model = build_model("rec", architecture)
            prediction = model.predict(features[None])
            reconstruction = model(features[None], sigma_stab[None])
            torch.testing.assert_close(reconstruction, sigma_stab[None] + prediction)
            self.assertEqual(reconstruction.shape, sigma_gt[None].shape)

    def test_paper_network_block_layouts(self):
        recnn1 = build_model("rec", "recnn1").backbone.blocks
        recnn2 = build_model("rec", "recnn2").backbone.blocks
        self.assertEqual(len(recnn1), 11)
        self.assertTrue(all(isinstance(block, ResidualBlock) for block in recnn1))
        self.assertEqual(sum(isinstance(x, ResidualBlock) for x in recnn2), 7)
        self.assertEqual(
            sum(isinstance(x, DownsampleResidualBlock) for x in recnn2), 2
        )
        self.assertEqual(
            sum(isinstance(x, UpsampleResidualBlock) for x in recnn2), 2
        )


if __name__ == "__main__":
    unittest.main()
