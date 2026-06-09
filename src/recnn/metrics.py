import numpy as np
from skimage.metrics import normalized_root_mse, structural_similarity


def nrmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(normalized_root_mse(target, prediction))


def ssim(prediction: np.ndarray, target: np.ndarray) -> float:
    scores = []
    for predicted_sample, target_sample in zip(prediction, target):
        predicted_image = np.squeeze(predicted_sample)
        target_image = np.squeeze(target_sample)
        data_range = float(target_image.max() - target_image.min())
        scores.append(
            structural_similarity(
                target_image,
                predicted_image,
                data_range=max(data_range, np.finfo(np.float32).eps),
            )
        )
    return float(np.mean(scores))
