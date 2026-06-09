import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import RECArrayDataset
from .metrics import nrmse, psnr
from .model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a REC-NN checkpoint.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    mode = checkpoint["mode"]
    architecture = checkpoint["architecture"]
    model = build_model(mode, architecture).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = RECArrayDataset(args.data, mode=mode)
    loader = DataLoader(dataset, batch_size=args.batch_size)
    predictions = []
    targets = []
    with torch.no_grad():
        for features, _, sigma_stab, sigma_gt in loader:
            predictions.append(model(features.to(device), sigma_stab.to(device)).cpu())
            targets.append(sigma_gt)

    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    print(f"NRMSE: {nrmse(prediction, target).item():.6f}")
    print(f"PSNR: {psnr(prediction, target).item():.3f} dB")


if __name__ == "__main__":
    main()
