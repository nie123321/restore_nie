"""CPU synthetic checks for Prior-A. Does not read real images."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch

from model import PriorA
from priors import six_luminance
from run import StepBatches, atomic_save, build_model, learning_rate, rng_state, restore_rng


ROOT = Path(__file__).resolve().parent
FROZEN_A = Path(
    r"M:\picture data\cholec80_t\code\cut\endo_enhancement_demo\runs"
    r"\A_spatial_fusion_on_off_10k_seed100_20260924\code\model.py"
)


def record(name: str, passed: bool, **details) -> dict:
    return {"name": name, "passed": passed, **details}


def finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all())


def load_frozen_a():
    spec = importlib.util.spec_from_file_location("frozen_enhancement_demo", FROZEN_A)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EnhancementDemo


def reference_a(seed: int):
    enhancement = load_frozen_a()
    torch.manual_seed(seed)
    return enhancement(width=24, output_mode="direct", spectral_mode="off", fusion_mode="gated")


def prior(seed: int) -> PriorA:
    torch.manual_seed(seed)
    return PriorA()


def check_luma_formula() -> dict:
    rgb = torch.tensor([0.2, 0.4, 0.8], dtype=torch.float32).view(1, 3, 1, 1)
    got = six_luminance(rgb)[0, :, 0, 0]
    red, green, blue = 0.2, 0.4, 0.8
    expected = torch.tensor([
        (red + green + blue) / 3.0,
        0.2126 * red + 0.7152 * green + 0.0722 * blue,
        max(red, green, blue),
        (max(red, green, blue) + min(red, green, blue)) / 2.0,
        0.25 * red + 0.5 * green + 0.25 * blue,
        math.sqrt(red * red + green * green + blue * blue + 1e-6) / math.sqrt(3.0),
    ])
    err = float((got - expected).abs().max())
    black = six_luminance(torch.zeros(1, 3, 2, 2))
    const = six_luminance(torch.full((1, 3, 2, 2), 0.3))
    model = PriorA().eval()
    image = torch.rand(1, 3, 8, 8)
    debug = model(image, return_debug=True)
    ok = err < 1e-6 and finite(black) and finite(const) and finite(debug["gamma_l"]) and finite(debug["structure_bias"])
    fft = [name for name, _ in model.named_parameters() if "filter" in name or "fft" in name]
    count = sum(p.numel() for p in model.parameters())
    return record("priors_and_count", ok and count == 208027 and not fft,
                  formula_max_error=err, parameters=count, fft_parameters=fft,
                  gamma_shape=list(debug["gamma_l"].shape),
                  structure_shape=list(debug["structure_bias"].shape))


def check_shared_init() -> dict:
    left = reference_a(11)
    right = prior(11)
    left_state, right_state = left.state_dict(), right.state_dict()
    missing, mismatch = [], []
    max_err = 0.0
    for name, value in left_state.items():
        if name not in right_state:
            missing.append(name)
            continue
        other = right_state[name]
        if other.shape != value.shape:
            mismatch.append(name)
            continue
        err = float((other - value).abs().max())
        max_err = max(max_err, err)
        if err > 0:
            mismatch.append(name)
    extra = sorted(set(right_state) - set(left_state))
    adapters = [name for name in extra if name.startswith(("luma_adapter", "structure_adapter", "structure_extractor"))]
    unexpected = [name for name in extra if name not in adapters]
    return record("shared_init", not missing and not mismatch and not unexpected,
                  max_error=max_err, missing=missing[:8], mismatch=mismatch[:8],
                  extra_count=len(extra))


def _a_features(model, image):
    skip0 = model.encoder0(model.stem(image))
    skip1 = model.encoder1(model.down1(skip0))
    encoded = model.encoder2(model.down2(skip1))
    pooled = encoded.mean(dim=(-2, -1))
    log_gain = image.new_zeros(image.shape[0], 1, *encoded.shape[-2:])
    gain_mean = log_gain.mean(dim=(1, 2, 3))
    gain_rms = torch.sqrt(log_gain.square().mean(dim=(1, 2, 3)) + 1e-6)
    condition = torch.cat((pooled, gain_mean[:, None], gain_rms[:, None]), dim=1)
    bottom, _ = model.spectral(encoded, condition)
    feature = model.fuse1(torch.cat((
        torch.nn.functional.interpolate(bottom, size=skip1.shape[-2:], mode="bilinear", align_corners=False),
        skip1), dim=1))
    feature = model.decoder1(feature)
    feature = model.fuse0(torch.cat((
        torch.nn.functional.interpolate(feature, size=skip0.shape[-2:], mode="bilinear", align_corners=False),
        skip0), dim=1))
    decoded = model.decoder0(feature)
    return encoded, bottom, decoded


def check_zero_prior() -> dict:
    left = reference_a(3)
    right = prior(3)
    image = torch.rand(1, 3, 32, 48)
    with torch.no_grad():
        a_enc, a_spec, a_dec = _a_features(left, image)
        debug = right(image, return_debug=True)
        enc = float((debug["encoder2"] - a_enc).abs().max())
        spec = float((debug["spectral"] - a_spec).abs().max())
        dec = float((debug["decoder0"] - a_dec).abs().max())
        left.direct_head.weight.fill_(0.01)
        right.direct_head.weight.copy_(left.direct_head.weight)
        left.direct_head.bias.fill_(0.02)
        right.direct_head.bias.copy_(left.direct_head.bias)
        out = float((right(image) - left(image)).abs().max())
    passed = max(enc, spec, dec, out) <= 1e-6
    return record("zero_prior_match", passed, encoder2=enc, spectral=spec, decoder0=dec, nonzero_head=out)


def check_conditions() -> dict:
    model = PriorA().train()
    image = torch.rand(1, 3, 16, 20, requires_grad=True)
    down = model.down2(model.encoder1(model.down1(model.encoder0(model.stem(image)))))
    gamma = torch.zeros(1, 96, *down.shape[-2:])
    gamma[:, :4] = 0.25
    changed = model.encoder2(down, gamma)
    base = model.encoder2(down, None)
    luma_effect = float((changed - base).detach().abs().max())
    encoded = model.encoder2(down, None).detach()
    pooled = encoded.mean(dim=(-2, -1))
    condition = torch.cat((
        pooled,
        encoded.new_zeros(1, 1),
        encoded.new_full((1, 1), 1e-3),
    ), dim=1)
    bias = torch.zeros_like(encoded)
    bias[:, :3] = 0.4
    spec_base, _ = model.spectral(encoded, condition, None)
    spec_new, _ = model.spectral(encoded, condition, bias)
    structure_effect = float((spec_new - spec_base).detach().abs().max())
    debug = model(image, return_debug=True)
    debug["output"].sum().backward()
    backward_ok = image.grad is not None and finite(image.grad)
    with torch.no_grad():
        model.direct_head.bias.fill_(0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    before_last = model.luma_adapter.project_out.weight.detach().clone()
    before_early = model.luma_adapter.project_in.weight.detach().clone()
    target = torch.rand_like(image)
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(image) - target).abs().mean()
        loss.backward()
        optimizer.step()
    last_moved = float((model.luma_adapter.project_out.weight.detach() - before_last).abs().max()) > 0
    struct_moved = float(model.structure_adapter.project_out.weight.grad.abs().mean()) > 0 if model.structure_adapter.project_out.weight.grad is not None else False
    early_moved = float((model.luma_adapter.project_in.weight.detach() - before_early).abs().max()) > 0
    passed = luma_effect > 0 and structure_effect > 0 and backward_ok and last_moved and struct_moved and early_moved
    return record("conditions", passed, luma_effect=luma_effect, structure_effect=structure_effect,
                  last_layer_updated=last_moved, early_layer_updated=early_moved, structure_grad=struct_moved)


def check_sizes_and_save() -> dict:
    model = PriorA().eval()
    details = {}
    with torch.no_grad():
        for size in ((32, 48), (33, 49)):
            image = torch.rand(1, 3, *size)
            output = model(image)
            details[f"{size[0]}x{size[1]}"] = output.shape == image.shape and finite(output)
    clone = build_model(model.config())
    blob = {"model": model.state_dict(), "config": {"arch_version": "prior_a_v1", "model": model.config()}}
    # A newly created TemporaryDirectory can have an ACL that rejects files
    # created within it in a sandboxed Windows session. Use the existing
    # writable checks directory; atomic_save still uses an adjacent temp file.
    path = ROOT / "checks" / f"verify_checkpoint_{os.getpid()}.pt"
    try:
        atomic_save(blob, path)
        saved = torch.load(path, map_location="cpu", weights_only=False)
    finally:
        path.unlink(missing_ok=True)
    clone.load_state_dict(saved["model"], strict=True)
    image = torch.rand(1, 3, 16, 18)
    with torch.no_grad():
        err = float((model(image) - clone(image)).abs().max())
    batches = list(StepBatches(8, 2, 100, 2, 3))
    order_ok = batches[0] == list(StepBatches(8, 2, 100, 2, 3))[0]
    passed = all(details.values()) and err < 1e-6 and order_ok
    return record("sizes_and_save", passed, shapes=details, restore_error=err, sample_order_ok=order_ok)


def check_help() -> dict:
    commands = {}
    for args in (
        [sys.executable, str(ROOT / "run.py"), "train", "--help"],
        [sys.executable, str(ROOT / "run.py"), "infer", "--help"],
        [sys.executable, str(ROOT / "eval_val.py"), "--help"],
    ):
        completed = subprocess.run(args, capture_output=True, text=True)
        commands[" ".join(args[-2:])] = completed.returncode == 0
    return record("entry_help", all(commands.values()), commands=commands)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.threads))
    checks = [
        check_luma_formula(), check_shared_init(), check_zero_prior(),
        check_conditions(), check_sizes_and_save(), check_help(),
    ]
    report = {
        "device": "cpu",
        "arch_version": "prior_a_v1",
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "gpu_amp": "not run",
        "training": "not run",
        "val_test_metrics": "not run",
    }
    out = ROOT / "checks" / "cpu_checks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "failed": [c["name"] for c in checks if not c["passed"]]}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
