"""Synthetic implementation checks; never use the formal paired dataset."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training" / "losses"))
from models.final_model import (
    BASE_MODEL_CONFIG, FINAL_MODEL_CONFIG, MODEL_SPEC, ReferenceModel,
    build_model, load_checkpoint, validate_model_config,
)
from models.baseline_a_blocks import ALL_OWN_SITES, install_eleven_own_blocks
from models.experiment_blocks import JointWaveletFusionBlock, install_joint_fusion
from models.candidate_blocks import RegionalLLContext, RegionalCoarseLowFrequencyAffineMixer
from star_blocks import StarBlock, StarModule, DFFN
from wavelet_blocks import WaveletMDTAResidual
from frequency_loss import gt_mean_fft_l1
from band_checks import verify_band_roles


def fingerprint(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def region_semantics():
    context = RegionalLLContext(24)
    low = torch.randn(1, 24, 12, 20, requires_grad=True)
    tokens, weights = context.token_attention(low)
    assert tokens.shape == (1, 16, 48)
    assert weights.shape == (1, 4, 16, 16)
    torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
    tokens[0, 0].square().mean().backward()
    # The first region receives information from the distant bottom-right region.
    remote_gradient = float(low.grad[..., 9:, 15:].abs().sum())
    assert remote_gradient > 0, "No cross-region information exchange"
    assert context(low.detach()).shape == (1, 48, 12, 20)
    assert torch.isfinite(context(torch.randn(1, 24, 1, 1))).all()
    return {"tokens": [1, 16, 48], "attention": [1, 4, 16, 16],
            "distant_region_gradient": remote_gradient, "restored_spatial_size": [12, 20]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(100)
    band_roles = verify_band_roles()
    region_checks = region_semantics() if MODEL_SPEC["recipe"]["candidate"] == "regional-ll2" else None
    torch.manual_seed(100)
    baseline = ReferenceModel(**BASE_MODEL_CONFIG)
    install_eleven_own_blocks(baseline)
    install_joint_fusion(baseline, "spatial-first")
    baseline_state = baseline.state_dict()
    torch.manual_seed(100)
    model = build_model()
    state = model.state_dict()
    assert set(baseline_state) <= set(state)
    assert all(torch.equal(value, state[key]) for key, value in baseline_state.items()), "Changed V2 initialization"
    common_fingerprint = fingerprint(baseline_state)
    initial_fingerprint = fingerprint(state)
    assert sum(type(m) is JointWaveletFusionBlock for m in model.modules()) == 11
    expected_mdta = MODEL_SPEC["recipe"]["mdta_count"]
    assert sum(type(m) is WaveletMDTAResidual for m in model.modules()) == expected_mdta
    assert not any(type(m) in (StarBlock, StarModule, DFFN) for m in model.modules())
    assert all(p.dtype == torch.float32 for p in model.parameters())
    if expected_mdta == 2:
        first = dict(model.mdta.named_parameters())
        second = dict(model.mdta_second.named_parameters())
        assert first.keys() == second.keys()
        assert all(first[key].data_ptr() != second[key].data_ptr() for key in first)
        assert not torch.equal(first["attn.qkv.weight"], second["attn.qkv.weight"])
    else:
        assert sum(type(m) is RegionalCoarseLowFrequencyAffineMixer for m in model.modules()) == 11
        for site in ALL_OWN_SITES:
            block = model.get_submodule(site)
            low = torch.randn(1, block.channels, 5, 7)
            high = tuple(torch.randn_like(low) for _ in range(3))
            with torch.no_grad():
                actual_low, actual_high = block.coarse(low, high)
                torch.testing.assert_close(actual_low, low, rtol=0, atol=0)
                assert all(torch.equal(a, b) for a, b in zip(actual_high, high))
    trace = []
    ordered_names = MODEL_SPEC["recipe"]["execution_order"]
    hooks = [model.get_submodule(name).register_forward_hook(
        lambda module, inputs, output, site=name: trace.append(site)) for name in ordered_names]
    with torch.no_grad():
        input_cpu = torch.rand(1, 3, 65, 97)
        output = model(input_cpu)
        torch.testing.assert_close(output, baseline(input_cpu), rtol=0, atol=0)
    for hook in hooks:
        hook.remove()
    assert trace == ordered_names, (trace, ordered_names)
    del baseline, baseline_state, state
    model.to(device=args.device, dtype=torch.float32).train()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    x = torch.rand(args.batch_size, 3, 128, 128, device=args.device)
    target = torch.rand_like(x)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    losses, gradients = [], {}
    for step in range(1, 4):
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        assert output.shape == x.shape and output.dtype == torch.float32
        assert torch.isfinite(output).all()
        loss, _ = gt_mean_fft_l1(output, target, sigma=.1, fft_weight=.1)
        assert torch.isfinite(loss)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all(), name
        if step == 3:
            scopes = []
            for site in ALL_OWN_SITES:
                scopes.extend(f"{site}.{scope}" for scope in (
                    "coarse.context_pw", "coarse.context_dw", "coarse.affine_head",
                    "coarse.high.filters.0", "coarse.high.filters.1", "coarse.high.filters.2",
                    "fine.high.filters.0", "fine.high.filters.1", "fine.high.filters.2",
                    "spatial.expand", "spatial.project", "spatial.ffn_in", "spatial.ffn_out"))
                if expected_mdta == 1:
                    scopes.extend(f"{site}.coarse.{scope}" for scope in (
                        "regional.input_project", "regional.norm", "regional.qkv",
                        "regional.output_project", "context_fusion"))
            for attention_name in ("mdta", "mdta_second")[:expected_mdta]:
                scopes.extend(f"{attention_name}.{scope}" for scope in (
                    "attn.qkv", "attn.qkv_dwconv", "attn.project_out"))
            for scope in scopes:
                parameters = list(model.get_submodule(scope).parameters())
                assert parameters and all(p.grad is not None for p in parameters), scope
                magnitude = sum(float(p.grad.abs().sum()) for p in parameters)
                assert magnitude > 0, scope
                gradients[scope] = magnitude
            if expected_mdta == 1:
                for site in ALL_OWN_SITES:
                    parameter = model.get_submodule(site).coarse.regional.position
                    assert parameter.grad is not None and float(parameter.grad.abs().sum()) > 0, site
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append({"synthetic_step": step, "loss": float(loss.detach()), "grad_norm": float(norm)})
        print(json.dumps(losses[-1]), flush=True)
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        odd = torch.rand(1, 3, 65, 97, device=args.device)
        expected = model(odd)
        assert expected.shape == odd.shape and torch.isfinite(expected).all()
        full = torch.rand(1, 3, 224, 448, device=args.device)
        full_output = model(full)
        assert full_output.shape == full.shape and torch.isfinite(full_output).all()
        buffer = io.BytesIO()
        torch.save({"model": model.state_dict(), "config": {"model": dict(model.config)}, "step": 0}, buffer)
        buffer.seek(0)
        restored, _ = load_checkpoint(buffer, args.device)
        torch.testing.assert_close(restored(odd), expected, rtol=1e-5, atol=2e-6)
    for other in ("a-joint-spatial-first-20261008", "old-d-eleven-own-ll-affine", "other-candidate"):
        try:
            validate_model_config(dict(FINAL_MODEL_CONFIG, experiment_variant=other))
        except ValueError:
            pass
        else:
            raise AssertionError(f"Wrong experiment accepted: {other}")
    report = {
        "result": "SYNTHETIC_SMOKE_PASS", "variant": MODEL_SPEC["variant"],
        "candidate": MODEL_SPEC["recipe"]["candidate"], "device": args.device,
        "python": sys.executable, "torch": torch.__version__, "precision": "FP32",
        "amp": False, "tf32": False, "batch_size": args.batch_size, "crop_size": [128, 128],
        "own_blocks": 11, "original_star_blocks": 0, "original_star_modules": 0,
        "dffn_blocks": 0, "mdta_blocks": expected_mdta, "V2_common_initialization_preserved": True,
        "common_initial_state_sha256": common_fingerprint, "initial_state_sha256": initial_fingerprint,
        "independent_mdta_weights": expected_mdta == 2,
        "execution_trace": trace, "regional_checks": region_checks,
        "band_roles": band_roles, "synthetic_losses": losses, "gradients": gradients,
        "odd_size": [65, 97], "full_size": [224, 448], "checkpoint_roundtrip": True,
        "foreign_checkpoint_rejected": True,
        "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20 if args.device == "cuda" else None,
        "elapsed_seconds": time.perf_counter()-started, "parameter_accounting": "not performed",
        "formal_training_started": False, "formal_dataset_images_read": 0,
    }
    destination = ROOT / "verification" / "reports"
    destination.mkdir(exist_ok=True)
    (destination / f"synthetic_{args.device}.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(report["result"], flush=True)


if __name__ == "__main__":
    main()
