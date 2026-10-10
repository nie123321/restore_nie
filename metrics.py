import math
import cv2
import numpy as np
import torch
from skimage.color import deltaE_ciede2000, rgb2lab


def psnr(target: np.ndarray, candidate: np.ndarray) -> float:
    target_f = target.astype(np.float32) / 255.0
    candidate_f = candidate.astype(np.float32) / 255.0
    mse = float(np.mean((target_f - candidate_f) ** 2))
    return 100.0 if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


def channel_ssim(target: np.ndarray, candidate: np.ndarray) -> float:
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    target = target.astype(np.float64)
    candidate = candidate.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(target, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(candidate, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(target ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(candidate ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(target * candidate, -1, window)[5:-5, 5:-5] - mu1_mu2
    result = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(result.mean())


def ssim(target: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.mean([channel_ssim(target[:, :, i], candidate[:, :, i]) for i in range(3)]))


def lpips_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    value = torch.from_numpy(image.transpose(2, 0, 1).copy()).unsqueeze(0).float() / 255.0
    return (value * 2.0 - 1.0).to(device)


def ciede2000(target: np.ndarray, candidate: np.ndarray) -> float:
    target_f = target.astype(np.float32) / 255.0
    candidate_f = candidate.astype(np.float32) / 255.0
    return float(np.mean(deltaE_ciede2000(rgb2lab(target_f), rgb2lab(candidate_f))))
