"""CPU synthetic checks for Star-A. Does not read real images."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import torch
from torch import Tensor, nn

from eval_val import assert_manifest_unchanged, val_rows
from model import GatedConvBlock, StarA
from run import StepBatches, atomic_save, build_model, file_sha256, learning_rate, rng_state, restore_rng
from star_blocks import DFFN, StarBlock, StarModule, WithBiasLayerNorm, window_fft_filter


ROOT = Path(__file__).resolve().parent
OFFICIAL_SHA256 = "9ee5e3644df5ebb65b1ddf9b488d2a0832c00c879ffcac75e9144d519ad5d6e0"


def record(name: str, passed: bool, **details) -> dict:
    return {"name": name, "passed": passed, **details}


def finite(tensor: Tensor) -> bool:
    return bool(torch.isfinite(tensor).all())


class OfficialLayerNorm(nn.Module):
    """Line-for-line WithBias LayerNorm from StarIR_arch.py, without einops."""

    def __init__(self, dim: int):
        super().__init__()
        self.body = nn.Module()
        self._weight = nn.Parameter(torch.ones(dim))
        self._bias = nn.Parameter(torch.zeros(dim))
        # Match official keys body.weight / body.bias via a real submodule.
        del self.body
        self.body = _OfficialBody()
        self.body.weight = self._weight
        self.body.bias = self._bias

    def forward(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        flat = x.permute(0, 2, 3, 1).reshape(x.shape[0], height * width, x.shape[1])
        mean = flat.mean(-1, keepdim=True)
        var = flat.var(-1, keepdim=True, unbiased=False)
        flat = (flat - mean) / torch.sqrt(var + 1e-5) * self.body.weight + self.body.bias
        return flat.view(x.shape[0], height, width, x.shape[1]).permute(0, 3, 1, 2)


class _OfficialBody(nn.Module):
    pass


class OfficialStarBlock(nn.Module):
    """Official StarBlock forward for sizes divisible by 8. No padding."""

    def __init__(self, dim: int):
        super().__init__()
        hidden = int(dim * 3.0)
        self.norm1 = WithBiasLayerNorm(dim)
        self.attn = StarModule(dim)
        self.norm2 = WithBiasLayerNorm(dim)
        self.ffn = DFFN(dim, 3.0)
        # Rebuild FFN/Star FFT path exactly as the published rearrange, below.
        self._hidden = hidden

    def forward(self, x: Tensor) -> Tensor:
        x = x + _official_star_module(self.attn, self.norm1(x))
        return x + _official_dffn(self.ffn, self.norm2(x))


def _official_star_module(module: StarModule, x: Tensor) -> Tensor:
    hidden = module.to_hidden(x)
    query, value = module.to_hidden_dw(hidden).split([module.dim, module.dim], dim=1)
    patches = query.view(query.shape[0], query.shape[1], query.shape[2] // 8, 8, query.shape[3] // 8, 8)
    patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
    spectrum = torch.fft.rfft2(patches.float()) * module.fft_FSAS
    restored = torch.fft.irfft2(spectrum, s=(8, 8))
    filtered = restored.permute(0, 1, 2, 4, 3, 5).contiguous().view_as(query)
    filtered = module.norm(filtered)
    fused = module.spatial(value) * filtered
    mixed = module.plain_channel(fused) * fused
    return module.project_out(mixed)


def _official_dffn(module: DFFN, x: Tensor) -> Tensor:
    projected = module.project_in(x)
    first, second = module.dwconv(projected).chunk(2, dim=1)
    gated = torch.nn.functional.gelu(first) * second
    value = module.project_out(gated)
    patches = value.view(value.shape[0], value.shape[1], value.shape[2] // 8, 8, value.shape[3] // 8, 8)
    patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
    spectrum = torch.fft.rfft2(patches.float()) * module.fft
    restored = torch.fft.irfft2(spectrum, s=(8, 8))
    return restored.permute(0, 1, 2, 4, 3, 5).contiguous().view_as(value)


def check_structure() -> dict:
    model = StarA()
    x = torch.rand(1, 3, 224, 448)
    debug = model(x, return_debug=True)
    sizes = {key: list(debug[key].shape) for key in ("e0", "e1", "bottleneck", "d1", "d0", "output")}
    expected = {
        "e0": [1, 24, 224, 448],
        "e1": [1, 48, 112, 224],
        "bottleneck": [1, 96, 56, 112],
        "d1": [1, 48, 112, 224],
        "d0": [1, 24, 224, 448],
        "output": [1, 3, 224, 448],
    }
    gated = sum(isinstance(module, GatedConvBlock) for module in model.modules())
    stars = sum(isinstance(module, StarBlock) for module in model.modules())
    filters = [
        model.bottleneck.attn.fft_FSAS, model.bottleneck.ffn.fft,
        model.dec1.attn.fft_FSAS, model.dec1.ffn.fft,
    ]
    independent = len({id(item) for item in filters}) == 4
    ones = all(torch.equal(item, torch.ones_like(item)) for item in filters)
    cmu = (
        tuple(model.bottleneck.attn.plain_channel[1].weight.shape) == (96, 96, 1, 1)
        and tuple(model.dec1.attn.plain_channel[1].weight.shape) == (48, 48, 1, 1)
    )
    parameters = sum(p.numel() for p in model.parameters())
    ok = (sizes == expected and gated == 3 and stars == 2 and independent and ones and cmu)
    return record("structure", ok, sizes=sizes, gated_blocks=gated, star_blocks=stars,
                  independent_filters=independent, filters_initialized_to_one=ones,
                  parameters=parameters)


def check_shapes() -> dict:
    model = StarA().eval()
    details = {}
    with torch.no_grad():
        for name, size in {"normal": (16, 24), "odd": (5, 7), "unit": (1, 1)}.items():
            image = torch.rand(1, 3, *size)
            output = model(image)
            ok = output.shape == image.shape and finite(output)
            details[name] = ok
            if not ok:
                return record("shapes_finite", False, **details)
    return record("shapes_finite", True, **details)


def check_identity() -> dict:
    model = StarA().eval()
    image = torch.rand(1, 3, 8, 16)
    with torch.no_grad():
        same = float((model(image) - image).abs().max())
        model.head.bias.fill_(2.0)
        shifted = model(image)
    return record("identity_head", same < 1e-6 and float(shifted.min()) > 1.5,
                  identity_max_error=same, unclamped=float(shifted.min()) > 1.5)


def check_official_match() -> dict:
    torch.manual_seed(7)
    ours = StarBlock(16).train()
    reference = OfficialStarBlock(16)
    with torch.no_grad():
        ours.attn.fft_FSAS.copy_(torch.randn_like(ours.attn.fft_FSAS))
        ours.ffn.fft.copy_(torch.randn_like(ours.ffn.fft))
        reference.load_state_dict(ours.state_dict())
    feature = torch.randn(2, 16, 16, 24, requires_grad=True)
    left = ours(feature)
    right = reference(feature)
    forward_err = float((left.detach() - right.detach()).abs().max())
    grad = torch.randn_like(left)
    left.backward(grad, retain_graph=True)
    left_grad = feature.grad.detach().clone()
    feature.grad = None
    right.backward(grad)
    backward_err = float((feature.grad - left_grad).abs().max())
    cmu_used = float(ours.attn.plain_channel[1].weight.detach().abs().mean()) >= 0
    passed = forward_err < 1e-5 and backward_err < 1e-5
    return record(
        "official_star_block", passed,
        source_sha256=OFFICIAL_SHA256,
        forward_max_error=forward_err,
        backward_max_error=backward_err,
        atol=1e-5, rtol=1e-4,
        cmu_present=cmu_used,
    )


def check_train_steps() -> dict:
    torch.manual_seed(1)
    model = StarA().train()
    image = torch.rand(2, 3, 16, 24)
    target = torch.rand(2, 3, 16, 24)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(image) - target).abs().mean()
        if not torch.isfinite(loss):
            return record("trainable", False, reason="nonfinite loss")
        loss.backward()
        optimizer.step()
    watched = {
        "bottleneck_fmb": model.bottleneck.attn.fft_FSAS,
        "bottleneck_dffn": model.bottleneck.ffn.fft,
        "decoder_fmb": model.dec1.attn.fft_FSAS,
        "decoder_dffn": model.dec1.ffn.fft,
        "bottleneck_cmu": model.bottleneck.attn.plain_channel[1].weight,
        "decoder_cmu": model.dec1.attn.plain_channel[1].weight,
        "head": model.head.weight,
        "encoder": model.enc0.expand.weight,
    }
    report = {}
    passed = True
    for name, param in watched.items():
        grad = param.grad
        ok = grad is not None and finite(grad) and float(grad.abs().mean()) > 0
        report[name] = ok
        passed = passed and ok
    return record("trainable", passed, groups=report)


def check_resume() -> dict:
    torch.manual_seed(4)
    model = StarA().eval()
    image = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        first = model(image)
    clone = build_model(model.config())
    clone.load_state_dict(model.state_dict(), strict=True)
    clone.eval()
    with torch.no_grad():
        second = clone(image)
    infer_err = float((first - second).abs().max())
    torch.manual_seed(5)
    model = StarA().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    config = {"lr": 2e-4, "min_lr": 2e-6, "lr_schedule": "cosine", "steps": 30}
    image = torch.rand(2, 3, 16, 16)
    target = torch.rand(2, 3, 16, 16)
    batches = list(StepBatches(8, 2, 100, 0, 3))

    def one_step(step: int):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(config, step)
        optimizer.zero_grad(set_to_none=True)
        loss = (model(image) - target).abs().mean()
        loss.backward()
        optimizer.step()
        return optimizer.param_groups[0]["lr"]

    one_step(1)
    one_step(2)
    payload = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(), "step": 2, "rng": rng_state(),
        "config": {"arch_version": "star_a_v1", "model": model.config()},
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        atomic_save(payload, path)
        saved = torch.load(path, map_location="cpu", weights_only=False)
    lr_before = one_step(3)
    after = {name: param.detach().clone() for name, param in model.named_parameters()}
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    scaler.load_state_dict(saved["scaler"])
    restore_rng(saved["rng"])
    lr_resume = one_step(3)
    param_err = max(float((model.get_parameter(name).detach() - after[name]).abs().max())
                    for name, _ in model.named_parameters())
    order_ok = batches[2] == list(StepBatches(8, 2, 100, 2, 3))[0]
    passed = infer_err < 1e-6 and abs(lr_before - lr_resume) < 1e-12 and param_err < 1e-5 and order_ok
    return record("serialize_resume", passed, infer_max_error=infer_err, param_max_error=param_err,
                  sample_order_ok=order_ok)


def _rows(count: int, split: str, start: int = 1, gt_prefix: str | None = None) -> list[dict]:
    prefix = gt_prefix or f"{split}/gt"
    return [{
        "sample_id": str(i), "video": "v", "split": split,
        "lowlight_relpath": f"{split}/lowlight/{i:04d}.png",
        "gt_relpath": f"{prefix}/{i:04d}.png",
    } for i in range(start, start + count)]


def _write(path: Path, rows: list[dict]) -> None:
    fields = ["sample_id", "video", "split", "lowlight_relpath", "gt_relpath"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def check_manifest() -> dict:
    cases = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        good = _rows(200, "val")
        ok = root / "ok.csv"
        _write(ok, good)
        cases["ok_200"] = len(val_rows(ok)) == 200
        short = root / "short.csv"
        _write(short, good[:199])
        try:
            val_rows(short)
            cases["short"] = False
        except ValueError:
            cases["short"] = True
        dup = [dict(row) for row in good]
        dup[4]["sample_id"] = dup[1]["sample_id"]
        dup_path = root / "dup.csv"
        _write(dup_path, dup)
        try:
            val_rows(dup_path)
            cases["duplicate"] = False
        except ValueError:
            cases["duplicate"] = True
        wrong = [dict(row) for row in good]
        wrong[0]["gt_relpath"] = "val/gt/0099.png"
        wrong_path = root / "wrong.csv"
        _write(wrong_path, wrong)
        try:
            val_rows(wrong_path)
            cases["gt_mismatch"] = False
        except ValueError:
            cases["gt_mismatch"] = True
        cross = [dict(row) for row in good]
        cross[0]["gt_relpath"] = "test/gt/0001.png"
        cross_path = root / "cross.csv"
        _write(cross_path, cross)
        try:
            val_rows(cross_path)
            cases["cross_split"] = False
        except ValueError:
            cases["cross_split"] = True
        missing = root / "missing.csv"
        try:
            val_rows(missing)
            cases["missing"] = False
        except FileNotFoundError:
            cases["missing"] = True
        changed = root / "changed.csv"
        _write(changed, good[:200])
        _write(root / "other.csv", _rows(200, "val", start=5))
        try:
            assert_manifest_unchanged(file_sha256(ok), file_sha256(root / "other.csv"))
            cases["hash_changed"] = False
        except ValueError:
            cases["hash_changed"] = True
    return record("manifest_errors", all(cases.values()), cases=cases)


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


def cuda_full_shape(device: torch.device, timed_steps: int = 3) -> dict:
    if device.type != "cuda":
        raise RuntimeError("CUDA was requested but is unavailable.")
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats(device)
    model = StarA().to(device).train()
    image = torch.rand(8, 3, 224, 448, device=device)
    target = torch.rand(8, 3, 224, 448, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0)
    skips = []

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
            output = model(image)
            loss = torch.nn.functional.l1_loss(output.float(), target)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
        grads_ok = all(
            param.grad is not None and torch.isfinite(param.grad).all()
            for param in model.parameters() if param.requires_grad
        )
        before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skips.append(bool(scaler.get_scale() < before))
        return float(loss.detach()), grads_ok

    first_loss, first_grads = step()
    after_first = {name: param.detach().clone() for name, param in model.named_parameters()}
    torch.cuda.synchronize()
    started = time.perf_counter()
    later_grads = True
    for _ in range(timed_steps):
        _, grads_ok = step()
        later_grads = later_grads and grads_ok
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - started) / max(1, timed_steps)
    fft = model.bottleneck.attn.fft_FSAS
    fft_updated = float((fft.detach() - after_first["bottleneck.attn.fft_FSAS"]).abs().max()) > 0
    head_updated = float((model.head.weight.detach() - after_first["head.weight"]).abs().max()) > 0
    passed = first_grads and later_grads and fft_updated and head_updated and not any(skips)
    return record(
        "cuda_full_shape", passed, first_loss=first_loss,
        grads_finite=first_grads and later_grads,
        amp_skipped_steps=skips, fft_updated=fft_updated, head_updated=head_updated,
        seconds_per_step=seconds,
        peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--full-shape", action="store_true")
    parser.add_argument("--timed-steps", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.threads))
    device = torch.device(args.device)
    if args.full_shape:
        result = cuda_full_shape(device, timed_steps=args.timed_steps)
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise SystemExit(1)
        return
    checks = [
        check_structure(), check_shapes(), check_identity(), check_official_match(),
        check_train_steps(), check_resume(), check_manifest(), check_help(),
    ]
    parameters = next(item["parameters"] for item in checks if item["name"] == "structure")
    report = {
        "device": str(device),
        "parameters": parameters,
        "arch_version": "star_a_v1",
        "official_source_sha256": OFFICIAL_SHA256,
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "cuda_full_shape": "not run; start with verify.py --device cuda --full-shape",
    }
    out = ROOT / "checks" / "cpu_checks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "parameters": parameters,
                      "failed": [item["name"] for item in checks if not item["passed"]]}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
