import math
import torch
from models.low_frequency_blocks import CoarseLowFrequencyAffineMixer, FineDetailMixer
from wavelet_blocks import haar_dwt, haar_idwt

def verify_band_roles():
    """Test coefficient behavior, raw-statistic context, and signed high updates."""
    channels = 24
    coarse, fine = CoarseLowFrequencyAffineMixer(channels), FineDetailMixer(channels)
    x = torch.randn(1, channels, 32, 48)
    ll1, h1, size1 = haar_dwt(x)
    ll2, h2, size2 = haar_dwt(ll1)
    with torch.no_grad():
        new_ll2, new_h2 = coarse(ll2, h2)
        assert torch.equal(new_ll2, ll2)
        assert all(torch.equal(a, b) for a, b in zip(new_h2, h2))
        rebuilt_ll1 = haar_idwt(new_ll2, new_h2, size2)
        low_pass, new_h1 = fine(rebuilt_ll1, h1)
        assert low_pass is rebuilt_ll1
        identity = haar_idwt(low_pass, new_h1, size1)
        torch.testing.assert_close(identity, x, rtol=1e-6, atol=1e-6)

        # A known LL2 affine correction must survive the fine reconstruction.
        gain, bias = 1.2, 0.15
        coarse.affine_head.bias[:channels].fill_(math.atanh(math.log(gain)))
        coarse.affine_head.bias[channels:].fill_(bias)
        new_ll2, new_h2 = coarse(ll2, h2)
        rebuilt_ll1 = haar_idwt(new_ll2, new_h2, size2)
        low_pass, new_h1 = fine(rebuilt_ll1, h1)
        rebuilt = haar_idwt(low_pass, new_h1, size1)
        observed_ll1, observed_h1, _ = haar_dwt(rebuilt)
        observed_ll2, observed_h2, _ = haar_dwt(observed_ll1)
        torch.testing.assert_close(observed_ll2, gain * ll2 + bias, rtol=1e-5, atol=2e-6)
        for actual, expected in zip(observed_h1 + observed_h2, h1 + h2):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)

        # Normalizing LL alone loses a common offset; the raw mean context retains it.
        captured = []
        hook = coarse.context_pw.register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[0].detach().clone()))
        coarse.affine_parameters(ll2)
        coarse.affine_parameters(ll2 + 0.7)
        hook.remove()
        first, shifted = captured
        torch.testing.assert_close(shifted[:, :channels], first[:, :channels], rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(shifted[:, channels:2*channels],
                                   first[:, channels:2*channels] + 0.7, rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(shifted[:, 2*channels:], first[:, 2*channels:], rtol=1e-5, atol=2e-6)
        # Independent DW3 updates retain signed detail; no sigmoid/clamp of coefficients.
        for layer in fine.high.filters:
            layer.weight[:, 0, 1, 1].fill_(0.2)
        negative_high = tuple(-torch.rand_like(ll2) for _ in range(3))
        same_low, signed_high = fine(ll2, negative_high)
        assert same_low is ll2
        for actual, expected in zip(signed_high, negative_high):
            torch.testing.assert_close(actual, 1.2 * expected)
            assert (actual < 0).all()
    return {"identity_max_error": float((identity-x).abs().max()),
            "ll_affine_reconstruction_max_error": float((observed_ll2-gain*ll2-bias).abs().max()),
            "raw_statistics_preserved": True, "fine_LL_passthrough": True,
            "signed_high_updates": True}
