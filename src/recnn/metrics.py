import torch


def nrmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = torch.linalg.vector_norm(prediction - target)
    scale = torch.linalg.vector_norm(target)
    return error / scale.clamp_min(torch.finfo(target.dtype).eps)


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((prediction - target) ** 2)
    data_range = target.max() - target.min()
    return 20 * torch.log10(data_range.clamp_min(1e-8)) - 10 * torch.log10(
        mse.clamp_min(1e-12)
    )
