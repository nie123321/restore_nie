import math
import torch
import torch.nn.functional as F


def gt_mean_l1(output, target, sigma=0.1, eps=1e-06):
    if sigma <= 0 or eps <= 0:
        raise ValueError('sigma and eps must be positive')
    if output.shape != target.shape or output.ndim != 4 or output.shape[1] != 3:
        raise ValueError('Expected matching NCHW RGB output and target')
    with torch.autocast(device_type=output.device.type, enabled=False):
        (prediction, reference) = (output.float(), target.float())
        gray = prediction.new_tensor([0.2989, 0.587, 0.114]).view(1, 3, 1, 1)
        mean_prediction = (prediction * gray).sum(1).mean((1, 2))
        mean_reference = (reference * gray).sum(1).mean((1, 2))
        valid = (mean_prediction > eps) & (mean_reference > eps)
        positive_prediction = mean_prediction.abs().clamp_min(eps)
        positive_reference = mean_reference.abs().clamp_min(eps)
        difference_squared = (positive_prediction - positive_reference).square()
        sum_squared = positive_prediction.square() + positive_reference.square()
        distance = difference_squared / (4 * sigma ** 2 * sum_squared)
        distance = distance + 0.5 * torch.log1p(difference_squared / (2 * positive_prediction * positive_reference))
        weight = torch.where(valid, distance.clamp(0, 1), torch.ones_like(distance)).detach()
        gain = mean_reference / mean_prediction.clamp_min(eps)
        aligned = (prediction * gain[:, None, None, None]).clamp(0, 1)
        raw_l1 = (prediction - reference).abs().flatten(1).mean(1)
        aligned_l1 = (aligned - reference).abs().flatten(1).mean(1)
        loss = (weight * raw_l1 + (1 - weight) * aligned_l1).mean()
        diagnostics = {'loss_raw_l1': raw_l1.mean().detach(), 'loss_aligned_l1': aligned_l1.mean().detach(), 'gt_mean_weight_mean': weight.mean().detach(), 'gt_mean_gain_mean': gain.mean().detach()}
    return (loss, diagnostics)


def complex_fft_l1(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if output.ndim != 4 or output.shape != target.shape:
        raise ValueError('FFT loss expects matching NCHW tensors.')
    with torch.autocast(device_type=output.device.type, enabled=False):
        output_fft = torch.fft.fft2(output.float(), dim=(-2, -1), norm='backward')
        target_fft = torch.fft.fft2(target.float(), dim=(-2, -1), norm='backward')
        return F.l1_loss(torch.view_as_real(output_fft), torch.view_as_real(target_fft))


def gt_mean_fft_l1(output: torch.Tensor, target: torch.Tensor, sigma: float=0.1, fft_weight: float=0.1) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not math.isfinite(fft_weight) or fft_weight < 0:
        raise ValueError('FFT loss weight must be finite and nonnegative.')
    (base, parts) = gt_mean_l1(output, target, sigma=sigma)
    frequency = complex_fft_l1(output, target)
    weighted_frequency = fft_weight * frequency
    return (base + weighted_frequency, {**parts, 'loss_gt_mean': base.detach(), 'loss_fft': frequency.detach(), 'loss_fft_weighted': weighted_frequency.detach()})
