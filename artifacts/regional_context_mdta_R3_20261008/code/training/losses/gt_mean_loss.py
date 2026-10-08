"""GT-Mean L1 from Liao et al., ICCV 2025, equations (1), (4), (5).

Formula implemented directly from the paper. Matches the official code's
grayscale means, detached weight, differentiable gain and aligned-output clamp.
https://github.com/jingxiLiao/GT-mean-loss/blob/main/basicsr/models/losses/losses.py
The loss runs in FP32. Nonpositive/near-zero means fall back to raw L1.
"""
import torch


def gt_mean_l1(output, target, sigma=0.1, eps=1e-6):
    if sigma <= 0 or eps <= 0:
        raise ValueError('sigma and eps must be positive')
    if output.shape != target.shape or output.ndim != 4 or output.shape[1] != 3:
        raise ValueError('Expected matching NCHW RGB output and target')
    with torch.autocast(device_type=output.device.type, enabled=False):
        prediction, reference = output.float(), target.float()
        gray = prediction.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
        mean_prediction = (prediction * gray).sum(1).mean((1, 2))
        mean_reference = (reference * gray).sum(1).mean((1, 2))
        valid = (mean_prediction > eps) & (mean_reference > eps)
        positive_prediction = mean_prediction.abs().clamp_min(eps)
        positive_reference = mean_reference.abs().clamp_min(eps)
        difference_squared = (positive_prediction - positive_reference).square()
        sum_squared = positive_prediction.square() + positive_reference.square()
        # Bhattacharyya distance for std_i = sigma * mean_i.
        distance = difference_squared / (4 * sigma**2 * sum_squared)
        distance = distance + 0.5 * torch.log1p(
            difference_squared / (2 * positive_prediction * positive_reference))
        weight = torch.where(valid, distance.clamp(0, 1), torch.ones_like(distance)).detach()
        gain = mean_reference / mean_prediction.clamp_min(eps)
        aligned = (prediction * gain[:, None, None, None]).clamp(0, 1)
        raw_l1 = (prediction - reference).abs().flatten(1).mean(1)
        aligned_l1 = (aligned - reference).abs().flatten(1).mean(1)
        loss = (weight * raw_l1 + (1 - weight) * aligned_l1).mean()
        diagnostics = {
            'loss_raw_l1': raw_l1.mean().detach(),
            'loss_aligned_l1': aligned_l1.mean().detach(),
            'gt_mean_weight_mean': weight.mean().detach(),
            'gt_mean_gain_mean': gain.mean().detach(),
        }
    return loss, diagnostics
