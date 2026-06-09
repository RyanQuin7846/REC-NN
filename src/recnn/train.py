import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import RECArrayDataset
from .metrics import nrmse
from .model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train REC-NN or EPE-NN.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mode", choices=["rec", "epe"], default="rec")
    parser.add_argument("--architecture", choices=["plain", "encoder"], default="plain")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RECArrayDataset(args.data, mode=args.mode)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = build_model(args.mode, args.architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    run_dir = args.output_dir / f"{args.mode}_{args.architecture}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_nrmse = 0.0
        for features, target, sigma_stab, sigma_gt in loader:
            features = features.to(device)
            target = target.to(device)
            sigma_stab = sigma_stab.to(device)
            sigma_gt = sigma_gt.to(device)

            optimizer.zero_grad()
            reconstruction = model(features, sigma_stab)
            loss = criterion(reconstruction, sigma_gt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(features)
            total_nrmse += nrmse(reconstruction.detach(), sigma_gt).item() * len(features)

        epoch_loss = total_loss / len(dataset)
        epoch_nrmse = total_nrmse / len(dataset)
        history.append({"epoch": epoch, "mse": epoch_loss, "nrmse": epoch_nrmse})
        print(f"epoch={epoch:04d} mse={epoch_loss:.6g} nrmse={epoch_nrmse:.6g}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "mode": args.mode,
                    "architecture": args.architecture,
                },
                run_dir / "best.pt",
            )

    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
