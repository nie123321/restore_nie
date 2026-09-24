"""CPU synthetic checks for ColorWaveletNet. Does not read real images."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import torch
import torch.nn.functional as F

from losses import color_wavelet_losses, srgb_to_cielab_d65
from lut import mix_residual_luts, sample_one_lut
from model import ColorWaveletNet, pad_to_multiple
from run import StepBatches, atomic_save, build_model, learning_rate, rng_state, restore_rng
from wavelet import haar_dwt2d, haar_iwt2d


ROOT = Path(__file__).resolve().parent
# Bruce Lindbloom sRGB D65 2° Lab, encoded RGB in {0,1}.
# http://www.brucelindbloom.com/ (sRGB, D65, 2°, no adaptation)
LAB_REFERENCE = {
    "black": (0.0, 0.0, 0.0),
    "white": (100.0, 0.0, 0.0),
    "red": (53.2408, 80.0925, 67.2032),
    "green": (87.7347, -86.1827, 83.1793),
    "blue": (32.2970, 79.1875, -107.8602),
}


def finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all())


def record(name: str, passed: bool, **details) -> dict:
    return {"name": name, "passed": passed, **details}


def check_shapes(device: torch.device) -> dict:
    model = ColorWaveletNet().to(device).eval()
    details = {}
    with torch.no_grad():
        for key, size in {"normal": (8, 12), "odd": (5, 7), "unit": (1, 1)}.items():
            x = torch.rand(1, 3, *size, device=device)
            aux = model(x, return_aux=True)
            y, z, w = aux["output"], aux["coarse"], aux["weights"]
            ok = y.shape == x.shape and z.shape == x.shape and w.shape == (1, 4, *size)
            ok = ok and finite(y) and finite(z) and finite(w)
            details[key] = {"shape": list(size), "ok": ok}
            if not ok:
                return record("shapes_finite", False, **details)
    return record("shapes_finite", True, **details)


def check_dwt() -> dict:
    torch.manual_seed(0)
    x = torch.randn(2, 8, 16, 16)
    low, high = haar_dwt2d(x)
    recon = haar_iwt2d(low, high)
    round_err = float((recon - x).abs().max())
    constant = torch.ones(1, 3, 8, 8)
    _ll, high_c = haar_dwt2d(constant)
    const_high = float(high_c.abs().max())
    signs = {}
    for name, block in {
        "a": torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
        "b": torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
        "c": torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        "d": torch.tensor([[0.0, 0.0], [0.0, 1.0]]),
    }.items():
        sample = block.view(1, 1, 2, 2)
        ll, hh = haar_dwt2d(sample)
        lh, hl, hhv = hh[0, 0, 0, 0].item(), hh[0, 1, 0, 0].item(), hh[0, 2, 0, 0].item()
        signs[name] = [round(ll.item(), 5), round(lh, 5), round(hl, 5), round(hhv, 5)]
    expected = {
        "a": [0.5, -0.5, -0.5, 0.5],
        "b": [0.5, -0.5, 0.5, -0.5],
        "c": [0.5, 0.5, -0.5, -0.5],
        "d": [0.5, 0.5, 0.5, 0.5],
    }
    signs_ok = all(signs[k] == expected[k] for k in expected)
    passed = round_err < 1e-5 and const_high < 1e-6 and signs_ok
    return record("dwt_iwt", passed, roundtrip_max_error=round_err,
                  constant_high_max=const_high, signs=signs)


def permutation_lut(size: int) -> torch.Tensor:
    """Residual that maps encoded RGB to [G, B, R]."""
    lut = torch.zeros(3, size, size, size)
    coords = torch.linspace(0, 1, size)
    b, g, r = torch.meshgrid(coords, coords, coords, indexing="ij")
    lut[0] = g - r
    lut[1] = b - g
    lut[2] = r - b
    return lut


def check_lut_axes() -> dict:
    size = 17
    rgb = torch.tensor([0.2, 0.5, 0.8]).view(1, 3, 1, 1).expand(1, 3, 4, 5).contiguous()
    sampled = sample_one_lut(permutation_lut(size), rgb)
    predicted = rgb + sampled
    target = torch.stack((rgb[:, 1], rgb[:, 2], rgb[:, 0]), dim=1)
    err = float((predicted - target).abs().max())
    return record("lut_axis_order", err < 1e-5, max_error=err)


def check_weights() -> dict:
    rgb = torch.rand(1, 3, 6, 8)
    luts = torch.zeros(4, 3, 17, 17, 17)
    for k, delta in enumerate(([0.10, 0.0, 0.0], [0.0, 0.10, 0.0], [0.0, 0.0, 0.10], [0.05, 0.05, 0.0])):
        luts[k] = torch.tensor(delta).view(3, 1, 1, 1)
    model = ColorWaveletNet().eval()
    with torch.no_grad():
        aux = model(rgb, return_aux=True)
        weights = aux["weights"]
    mass = weights.sum(dim=1)
    simplex = bool((weights >= -1e-6).all() and (mass - 1).abs().max() < 1e-5)
    one_hot = torch.zeros_like(weights)
    one_hot[:, 2] = 1
    mixed = rgb + mix_residual_luts(luts, rgb, one_hot)
    expected = rgb.clone()
    expected[:, 2] += 0.10
    formula = float((mixed - expected).abs().max())
    return record("spatial_weights", simplex and formula < 1e-6,
                  simplex=simplex, formula_error=formula)


def named_params(model: ColorWaveletNet) -> dict[str, list[str]]:
    groups = {
        "lut": [], "routing": [], "low": [], "high": [], "guide": [], "output": [],
    }
    for name, _ in model.named_parameters():
        if name.startswith("delta_luts"):
            groups["lut"].append(name)
        elif name.startswith("context"):
            groups["routing"].append(name)
        elif "guide" in name and "affine" not in name:
            groups["low"].append(name)
        elif name.startswith("high"):
            groups["high"].append(name)
        elif "affine" in name or "condition" in name:
            groups["guide"].append(name)
        elif name.startswith("residual_head"):
            groups["output"].append(name)
    return groups


def check_gradients(device: torch.device) -> dict:
    torch.manual_seed(1)
    model = ColorWaveletNet().to(device).train()
    x = torch.rand(2, 3, 8, 8, device=device)
    gt = torch.rand(2, 3, 8, 8, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    last_grads = {}
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        aux = model(x, return_aux=True)
        loss, _ = color_wavelet_losses(
            aux["output"], aux["coarse"], gt, model.delta_luts, 0.2, 0.05, 1e-4,
        )
        if not torch.isfinite(loss):
            return record("key_gradients", False, reason="nonfinite loss")
        loss.backward()
        for name, param in model.named_parameters():
            grad = param.grad
            last_grads[name] = None if grad is None else float(grad.detach().abs().mean())
        optimizer.step()
    groups = named_params(model)
    report = {}
    passed = True
    for group, names in groups.items():
        grads = [last_grads[n] for n in names]
        changed = [float((model.get_parameter(n).detach() - before[n]).abs().max()) for n in names]
        ok = all(g is not None and math.isfinite(g) and g > 0 for g in grads) and all(c > 0 for c in changed)
        report[group] = {"ok": ok, "mean_abs_grad": grads, "max_update": changed}
        passed = passed and ok
    return record("key_gradients", passed, groups=report)


def check_stage1_from_y(device: torch.device) -> dict:
    torch.manual_seed(2)
    model = ColorWaveletNet().to(device).train()
    x = torch.rand(1, 3, 8, 8, device=device)
    gt = torch.rand(1, 3, 8, 8, device=device)
    y = model(x)
    loss = (y - gt).abs().mean()
    loss.backward()
    lut_grad = model.delta_luts.grad
    ctx = [p.grad for n, p in model.named_parameters() if n.startswith("context") and p.requires_grad]
    lut_ok = lut_grad is not None and finite(lut_grad) and float(lut_grad.abs().mean()) > 0
    ctx_ok = all(g is not None and finite(g) and float(g.abs().mean()) > 0 for g in ctx)
    return record("y_loss_reaches_stage1", lut_ok and ctx_ok, lut_ok=lut_ok, context_ok=ctx_ok)


def check_guide_used(device: torch.device) -> dict:
    torch.manual_seed(3)
    model = ColorWaveletNet().to(device).eval()
    x = torch.rand(1, 3, 8, 8, device=device)
    with torch.no_grad():
        aux = model(x, return_debug=True)
        high_from_low = model.high1(aux["high1"], aux["low1_hat"] + 0.25, aux["query1"])
        high_from_q = model.high1(aux["high1"], aux["low1_hat"], aux["query1"] + 0.25)
        low_delta = float((high_from_low - aux["high1_hat"]).abs().max())
        query_delta = float((high_from_q - aux["high1_hat"]).abs().max())
    passed = low_delta > 1e-6 and query_delta > 1e-6
    return record("guide_affects_output", passed, high_from_lhat=low_delta, high_from_query=query_delta)


def check_lab() -> dict:
    def lab_of(rgb):
        tensor = torch.tensor(rgb, dtype=torch.float32).view(1, 3, 1, 1)
        return srgb_to_cielab_d65(tensor)[0, :, 0, 0].tolist()

    measured = {
        "black": lab_of((0.0, 0.0, 0.0)),
        "white": lab_of((1.0, 1.0, 1.0)),
        "gray": lab_of((0.5, 0.5, 0.5)),
        "red": lab_of((1.0, 0.0, 0.0)),
        "green": lab_of((0.0, 1.0, 0.0)),
        "blue": lab_of((0.0, 0.0, 1.0)),
    }
    chroma_ok = abs(measured["black"][1]) < 1e-4 and abs(measured["black"][2]) < 1e-4
    chroma_ok = chroma_ok and abs(measured["white"][1]) < 0.05 and abs(measured["white"][2]) < 0.05
    chroma_ok = chroma_ok and abs(measured["gray"][1]) < 0.05 and abs(measured["gray"][2]) < 0.05
    refs = {}
    ref_ok = True
    for name in ("red", "green", "blue"):
        err = [abs(a - b) for a, b in zip(measured[name], LAB_REFERENCE[name])]
        refs[name] = {"measured": measured[name], "reference": list(LAB_REFERENCE[name]), "abs_error": err}
        ref_ok = ref_ok and max(err) < 0.05
    rgb = torch.rand(1, 3, 4, 4, requires_grad=True)
    lab = srgb_to_cielab_d65(rgb.clamp(0, 1))
    lab.mean().backward()
    grad_ok = rgb.grad is not None and finite(rgb.grad)
    return record("lab_values", chroma_ok and ref_ok and grad_ok, measured=measured,
                  references=refs, chroma_ok=chroma_ok, reference_ok=ref_ok, grad_ok=grad_ok)


def check_black_lab_grad() -> dict:
    black = torch.zeros(1, 3, 4, 4, requires_grad=True)
    lab = srgb_to_cielab_d65(black.clamp(0, 1))
    if not finite(lab):
        return record("black_lab_grad", False, reason="nonfinite Lab at black")
    lab.mean().backward()
    grad = black.grad
    ok = grad is not None and finite(grad)
    model = ColorWaveletNet().train()
    x = torch.zeros(1, 3, 8, 8)
    gt = torch.rand(1, 3, 8, 8)
    aux = model(x, return_aux=True)
    loss, _ = color_wavelet_losses(aux["output"], aux["coarse"], gt, model.delta_luts, 0.2, 0.05, 1e-4)
    loss.backward()
    lut_grad = model.delta_luts.grad
    loss_ok = math.isfinite(float(loss.detach())) and lut_grad is not None and finite(lut_grad)
    return record("black_lab_grad", ok and loss_ok, lab_grad_finite=ok, loss_finite=math.isfinite(float(loss.detach())),
                  lut_grad_finite=lut_grad is not None and finite(lut_grad) if lut_grad is not None else False)


def check_serialize(device: torch.device) -> dict:
    torch.manual_seed(4)
    model = ColorWaveletNet().to(device).eval()
    x = torch.rand(1, 3, 8, 8, device=device)
    with torch.no_grad():
        y1 = model(x)
    blob = {"model": model.state_dict(), "config": {"arch_version": "color_wavelet_v1", "model": model.config()}}
    clone = build_model(blob["config"]["model"]).to(device)
    clone.load_state_dict(blob["model"], strict=True)
    clone.eval()
    with torch.no_grad():
        y2 = clone(x)
    infer_err = float((y1 - y2).abs().max())

    torch.manual_seed(5)
    model = ColorWaveletNet().to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    config = {"lr": 2e-4, "min_lr": 2e-6, "lr_schedule": "cosine", "steps": 30}
    x = torch.rand(2, 3, 8, 8, device=device)
    gt = torch.rand(2, 3, 8, 8, device=device)
    batches = list(StepBatches(8, 2, 100, 0, 3))
    def one_step(step: int):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(config, step)
        optimizer.zero_grad(set_to_none=True)
        aux = model(x, return_aux=True)
        loss, _ = color_wavelet_losses(aux["output"], aux["coarse"], gt, model.delta_luts, 0.2, 0.05, 1e-4)
        loss.backward()
        optimizer.step()
        return optimizer.param_groups[0]["lr"]

    one_step(1)
    one_step(2)
    payload = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "step": 2, "rng": rng_state(), "config": {"model": model.config(), "arch_version": "color_wavelet_v1"},
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        atomic_save(payload, path)
        saved = torch.load(path, map_location="cpu", weights_only=False)
    lr_before = one_step(3)
    after = {n: p.detach().clone() for n, p in model.named_parameters()}
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    restore_rng(saved["rng"])
    lr_resume = one_step(3)
    param_err = max(float((model.get_parameter(n).detach() - after[n]).abs().max()) for n, _ in model.named_parameters())
    order_ok = batches[2] == list(StepBatches(8, 2, 100, 2, 3))[0]
    passed = infer_err < 1e-6 and abs(lr_before - lr_resume) < 1e-12 and param_err < 1e-5 and order_ok
    return record("serialize_resume", passed, infer_max_error=infer_err, lr_match=lr_before == lr_resume,
                  param_max_error=param_err, sample_order_ok=order_ok, next_lr=lr_resume)


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


def _amp_step(model, optimizer, scaler, x, gt):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
        aux = model(x, return_aux=True)
    loss, _ = color_wavelet_losses(
        aux["output"], aux["coarse"], gt, model.delta_luts, 0.2, 0.05, 1e-4,
    )
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    skipped = bool(scaler.get_scale() < previous_scale)
    return loss, skipped


def cuda_full_shape(device: torch.device, batch: int = 8, timed_steps: int = 4) -> dict:
    if device.type != "cuda":
        raise RuntimeError("CUDA was requested but is unavailable.")
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats(device)
    model = ColorWaveletNet().to(device).train()
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    x = torch.rand(batch, 3, 224, 448, device=device)
    gt = torch.rand(batch, 3, 224, 448, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0)
    loss, skipped = _amp_step(model, optimizer, scaler, x, gt)
    if not torch.isfinite(loss):
        return record("cuda_full_shape", False, reason="nonfinite loss", batch=batch)
    grads_ok = True
    for param in model.parameters():
        if param.requires_grad and (param.grad is None or not torch.isfinite(param.grad).all()):
            grads_ok = False
            break
    updated = any(
        float((param.detach() - before[name]).abs().max()) > 0
        for name, param in model.named_parameters()
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(timed_steps):
        loss, skipped = _amp_step(model, optimizer, scaler, x, gt)
    torch.cuda.synchronize()
    seconds_per_step = (time.perf_counter() - started) / max(1, timed_steps)
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
        torch.cuda.synchronize()
        val_started = time.perf_counter()
        _ = model(x[:1])
        torch.cuda.synchronize()
        val_seconds = time.perf_counter() - val_started
    train_hours = 30000 * seconds_per_step / 3600
    val_hours = (30000 / 500) * 200 * val_seconds / 3600
    passed = grads_ok and not skipped and updated
    return record(
        "cuda_full_shape", passed, batch=batch, loss=float(loss.detach()),
        grads_finite=grads_ok, amp_skipped_step=skipped, params_updated=updated,
        peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
        seconds_per_train_step=seconds_per_step,
        seconds_per_val_image=val_seconds,
        estimate_hours_30k_train_only=train_hours,
        estimate_hours_30k_with_val_every_500=train_hours + val_hours,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--full-shape", action="store_true",
                        help="Optional CUDA AMP step at 3, 224, 448. Do not run unless requested.")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--timed-steps", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.threads))
    device = torch.device(args.device)
    if args.full_shape:
        result = cuda_full_shape(device, batch=args.batch, timed_steps=args.timed_steps)
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise SystemExit(1)
        return
    checks = [
        check_shapes(device),
        check_dwt(),
        check_lut_axes(),
        check_weights(),
        check_gradients(device),
        check_stage1_from_y(device),
        check_guide_used(device),
        check_lab(),
        check_black_lab_grad(),
        check_serialize(device),
        check_help(),
    ]
    model = ColorWaveletNet()
    parameters = sum(p.numel() for p in model.parameters())
    report = {
        "device": str(device),
        "parameters": parameters,
        "arch_version": "color_wavelet_v1",
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "cuda_full_shape": "not run; start with verify.py --device cuda --full-shape",
    }
    out = ROOT / "checks" / "cpu_checks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "parameters": parameters,
                      "failed": [c["name"] for c in checks if not c["passed"]]}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
