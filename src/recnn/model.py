import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional


def initialize_paper_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.01)
        return self.conv2(x) + residual


class DownsampleResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.01)
        return self.conv2(x) + residual


class UpsampleResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.ConvTranspose2d(
            out_channels, out_channels, 3, stride=2, padding=1, output_padding=1
        )
        self.skip = nn.ConvTranspose2d(
            in_channels, out_channels, 3, stride=2, padding=1, output_padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.01)
        return self.conv2(x) + residual


class PlainResNet(nn.Module):
    """REC-NN1: 22 convolutional layers arranged as 11 residual blocks."""

    def __init__(self, in_channels: int):
        super().__init__()
        widths = [32, 32, 64, 64, 128, 128, 64, 64, 32, 32, 1]
        blocks = []
        current = in_channels
        for width in widths:
            blocks.append(ResidualBlock(current, width))
            current = width
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class EncoderDecoderResNet(nn.Module):
    """REC-NN2: seven residual blocks plus four down/up-sampling blocks."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.blocks = nn.Sequential(
            ResidualBlock(in_channels, 32),
            ResidualBlock(32, 32),
            DownsampleResidualBlock(32, 64),
            ResidualBlock(64, 64),
            DownsampleResidualBlock(64, 128),
            ResidualBlock(128, 128),
            UpsampleResidualBlock(128, 64),
            ResidualBlock(64, 64),
            UpsampleResidualBlock(64, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class RECNN(nn.Module):
    def __init__(self, backbone: nn.Module, mode: str = "rec"):
        super().__init__()
        if mode not in {"rec", "epe"}:
            raise ValueError("mode must be 'rec' or 'epe'")
        self.backbone = backbone
        self.mode = mode

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Predict the Stab-EPT residual (REC) or conductivity (EPE)."""
        return self.backbone(features)

    def forward(
        self, features: torch.Tensor, sigma_stab: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        prediction = self.predict(features)
        if self.mode == "rec":
            if sigma_stab is None:
                raise ValueError("sigma_stab is required in REC mode")
            return sigma_stab + prediction
        return prediction


def build_model(mode: str = "rec", architecture: str = "recnn1") -> RECNN:
    if mode not in {"rec", "epe"}:
        raise ValueError("mode must be 'rec' or 'epe'")
    in_channels = 5 if mode == "rec" else 4
    if architecture == "recnn1":
        backbone = PlainResNet(in_channels)
    elif architecture == "recnn2":
        backbone = EncoderDecoderResNet(in_channels)
    else:
        raise ValueError("architecture must be 'recnn1' or 'recnn2'")
    model = RECNN(backbone, mode=mode)
    model.apply(initialize_paper_weights)
    return model
