import torch
from torch import nn


class SignedSigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sigmoid(x) - 0.5 * torch.sigmoid(-x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.activation = SignedSigmoid()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if stride == 1:
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        else:
            self.conv2 = nn.Conv2d(
                out_channels, out_channels, 3, stride=stride, padding=1
            )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        return self.activation(x + residual)


class UpsampleResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.activation = SignedSigmoid()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.ConvTranspose2d(
            out_channels, out_channels, 3, stride=2, padding=1, output_padding=1
        )
        self.skip = nn.ConvTranspose2d(
            in_channels, out_channels, 3, stride=2, padding=1, output_padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.activation(self.conv1(x))
        return self.activation(self.conv2(x) + residual)


class PlainResNet(nn.Module):
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
    def __init__(self, in_channels: int):
        super().__init__()
        self.body = nn.Sequential(
            ResidualBlock(in_channels, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            UpsampleResidualBlock(128, 64),
            ResidualBlock(64, 64),
            UpsampleResidualBlock(64, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class RECNN(nn.Module):
    def __init__(self, backbone: nn.Module, mode: str = "rec"):
        super().__init__()
        if mode not in {"rec", "epe"}:
            raise ValueError("mode must be 'rec' or 'epe'")
        self.backbone = backbone
        self.mode = mode

    def forward(
        self, features: torch.Tensor, sigma_stab: torch.Tensor | None = None
    ) -> torch.Tensor:
        prediction = self.backbone(features)
        if self.mode == "rec":
            if sigma_stab is None:
                raise ValueError("sigma_stab is required in REC mode")
            return sigma_stab + prediction
        return prediction


def build_model(mode: str = "rec", architecture: str = "plain") -> RECNN:
    in_channels = 5 if mode == "rec" else 4
    if architecture == "plain":
        backbone = PlainResNet(in_channels)
    elif architecture == "encoder":
        backbone = EncoderDecoderResNet(in_channels)
    else:
        raise ValueError("architecture must be 'plain' or 'encoder'")
    return RECNN(backbone, mode=mode)
