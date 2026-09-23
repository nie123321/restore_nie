"""CPU synthetic checks for Deep-A. Does not read real images."""
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

from eval_val import val_rows
from model import DeepA, GatedConvBlock, LayerNorm2d, SpatialFusion
from run import StepBatches, atomic_save, build_model, learning_rate, rng_state, restore_rng


ROOT = Path(__file__).resolve().parent


def record(name: str, passed: bool, **details) -> dict:
    return {"name": name, "passed": passed, **details}


def finite(tensor: Tensor) -> bool:
    return bool(torch.isfinite(tensor).all())


class ReferenceOffFusion(nn.Module):
    """Frozen copy of ConditionalSpectralBlock spatial path with mode=off."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.project_in = nn.Conv2d(channels, 2 * channels, 1)
        self.depthwise = nn.Conv2d(2 * channels, 2 * channels, 3,
                                   padding=1, groups=2 * channels)
        self.frequency_norm = LayerNorm2d(channels)
        self.spatial_gate = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.project_out = nn.Conv2d(channels, channels, 1)
        self.residual_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, x: Tensor, condition: Tensor):
        q, v = self.depthwise(self.project_in(self.norm(x))).chunk(2, dim=1)
        response = None
        fused = self.frequency_norm(q) * (v * torch.sigmoid(self.spatial_gate(v)))
        return x + self.residual_scale * self.project_out(fused), response


def _count_blocks(module: nn.Sequential) -> int:
    return sum(isinstance(child, GatedConvBlock) for child in module)


def check_architecture() -> dict:
    model = DeepA()
    x = torch.rand(2, 3, 224, 448)
    debug = model(x, return_debug=True)
    sizes = {
        "e0": list(debug["e0"].shape),
        "e1": list(debug["e1"].shape),
        "e2": list(debug["e2"].shape),
        "e3": list(debug["e3"].shape),
        "d2": list(debug["d2"].shape),
        "d1": list(debug["d1"].shape),
        "d0": list(debug["d0"].shape),
        "out": list(debug["output"].shape),
    }
    expected = {
        "e0": [2, 24, 224, 448],
        "e1": [2, 48, 112, 224],
        "e2": [2, 96, 56, 112],
        "e3": [2, 192, 28, 56],
        "d2": [2, 96, 56, 112],
        "d1": [2, 48, 112, 224],
        "d0": [2, 24, 224, 448],
        "out": [2, 3, 224, 448],
    }
    fusion_count = sum(isinstance(module, SpatialFusion) for module in model.modules())
    blocks = {
        "enc0": _count_blocks(model.enc0),
        "enc1": _count_blocks(model.enc1),
        "enc2": _count_blocks(model.enc2),
        "enc3": _count_blocks(model.enc3),
        "dec2": _count_blocks(model.dec2),
        "dec1": _count_blocks(model.dec1),
        "dec0": _count_blocks(model.dec0),
    }
    parameters = sum(p.numel() for p in model.parameters())
    ok = sizes == expected and fusion_count == 1 and blocks == {
        "enc0": 1, "enc1": 2, "enc2": 2, "enc3": 2, "dec2": 2, "dec1": 2, "dec0": 1,
    }
    return record("architecture", ok, sizes=sizes, fusion_count=fusion_count,
                  gated_blocks=blocks, parameters=parameters)


def check_shapes() -> dict:
    model = DeepA().eval()
    details = {}
    with torch.no_grad():
        for name, size in {"normal": (16, 20), "odd": (5, 7), "unit": (1, 1)}.items():
            x = torch.rand(1, 3, *size)
            y = model(x)
            ok = y.shape == x.shape and finite(y)
            details[name] = {"shape": list(size), "ok": ok}
            if not ok:
                return record("shapes_finite", False, **details)
    return record("shapes_finite", True, **details)


def check_identity_and_no_clamp() -> dict:
    model = DeepA().eval()
    x = torch.rand(1, 3, 8, 12)
    with torch.no_grad():
        y0 = model(x)
        identity_err = float((y0 - x).abs().max())
        model.head.bias.fill_(2.0)
        y1 = model(x)
    unclamped = bool((y1 > 1.5).any() and (y1 < 0).any() is False)
    # y1 = x + 2 is in (2,3); confirm it was not clipped to 1.
    unclamped = bool(float(y1.min()) > 1.5)
    return record("identity_head", identity_err < 1e-6 and unclamped,
                  identity_max_error=identity_err, unclamped=unclamped)


def check_spatial_fusion() -> dict:
    torch.manual_seed(0)
    reference = ReferenceOffFusion(32)
    fusion = SpatialFusion(32)
    fusion.load_state_dict(reference.state_dict())
    x = torch.randn(2, 32, 9, 11, requires_grad=True)
    dummy = torch.zeros(2, 34)
    y_ref, _ = reference(x, dummy)
    y_new = fusion(x)
    forward_err = float((y_ref.detach() - y_new.detach()).abs().max())
    grad = torch.randn_like(y_ref)
    y_ref.backward(grad, retain_graph=True)
    ref_grad = x.grad.detach().clone()
    x.grad = None
    y_new.backward(grad)
    backward_err = float((x.grad - ref_grad).abs().max())
    return record("spatial_fusion_match", forward_err < 1e-6 and backward_err < 1e-6,
                  forward_max_error=forward_err, backward_max_error=backward_err)


def _module_updated(module: nn.Module, before: dict[str, Tensor], prefix: str) -> bool:
    for name, param in module.named_parameters():
        key = f"{prefix}.{name}" if prefix else name
        if float((param.detach() - before[key]).abs().max()) > 0:
            return True
    return False


def _module_grad_ok(module: nn.Module) -> bool:
    grads = [param.grad for param in module.parameters() if param.requires_grad]
    return all(g is not None and finite(g) and float(g.abs().mean()) > 0 for g in grads)


def check_gradients() -> dict:
    torch.manual_seed(1)
    model = DeepA().train()
    x = torch.rand(2, 3, 16, 24)
    gt = torch.rand(2, 3, 16, 24)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x) - gt).abs().mean()
        if not torch.isfinite(loss):
            return record("key_gradients", False, reason="nonfinite loss")
        loss.backward()
        optimizer.step()
    groups = {
        "down3": model.down3,
        "enc3": model.enc3,
        "fusion": model.fusion,
        "dec2": model.dec2,
        "dec1": model.dec1,
        "dec0": model.dec0,
        "head": model.head,
    }
    report = {}
    passed = True
    for name, module in groups.items():
        ok = _module_grad_ok(module) and _module_updated(module, before, name)
        report[name] = ok
        passed = passed and ok
    return record("key_gradients", passed, groups=report)


def check_serialize() -> dict:
    torch.manual_seed(4)
    model = DeepA().eval()
    x = torch.rand(1, 3, 8, 8)
    with torch.no_grad():
        y1 = model(x)
    blob = {"model": model.state_dict(), "config": {"arch_version": "deep_a_v1", "model": model.config()}}
    clone = build_model(blob["config"]["model"])
    clone.load_state_dict(blob["model"], strict=True)
    clone.eval()
    with torch.no_grad():
        y2 = clone(x)
    infer_err = float((y1 - y2).abs().max())

    torch.manual_seed(5)
    model = DeepA().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    config = {"lr": 2e-4, "min_lr": 2e-6, "lr_schedule": "cosine", "steps": 30}
    x = torch.rand(2, 3, 8, 8)
    gt = torch.rand(2, 3, 8, 8)
    batches = list(StepBatches(8, 2, 100, 0, 3))

    def one_step(step: int):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(config, step)
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x) - gt).abs().mean()
        loss.backward()
        optimizer.step()
        return optimizer.param_groups[0]["lr"]

    one_step(1)
    one_step(2)
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "step": 2, "rng": rng_state(), "config": {"model": model.config(), "arch_version": "deep_a_v1"}}
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
    param_err = max(float((model.get_parameter(n).detach() - after[n]).abs().max())
                    for n, _ in model.named_parameters())
    order_ok = batches[2] == list(StepBatches(8, 2, 100, 2, 3))[0]
    passed = infer_err < 1e-6 and abs(lr_before - lr_resume) < 1e-12 and param_err < 1e-5 and order_ok
    return record("serialize_resume", passed, infer_max_error=infer_err,
                  lr_match=abs(lr_before - lr_resume) < 1e-12, param_max_error=param_err,
                  sample_order_ok=order_ok)


def _write_manifest(path: Path, rows: list[dict]) -> None:
    fields = ["sample_id", "video", "split", "lowlight_relpath", "gt_relpath"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def check_manifest() -> dict:
    cases = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        good = [{"sample_id": str(i), "video": "v", "split": "val",
                 "lowlight_relpath": f"val/lowlight/{i:04d}.png",
                 "gt_relpath": f"val/gt/{i:04d}.png"} for i in range(1, 201)]
        path = root / "ok.csv"
        _write_manifest(path, good)
        rows = val_rows(path)
        cases["ok_200"] = len(rows) == 200

        short = root / "short.csv"
        _write_manifest(short, good[:199])
        try:
            val_rows(short)
            cases["short"] = False
        except ValueError:
            cases["short"] = True

        dup = good.copy()
        dup[10] = dict(dup[3])
        dup_path = root / "dup.csv"
        _write_manifest(dup_path, dup)
        try:
            val_rows(dup_path)
            cases["duplicate"] = False
        except ValueError:
            cases["duplicate"] = True

        missing = root / "missing.csv"
        try:
            val_rows(missing)
            cases["missing_file"] = False
        except FileNotFoundError:
            cases["missing_file"] = True

        wrong_gt = [dict(row) for row in good]
        wrong_gt[0]["gt_relpath"] = "test/gt/0001.png"
        wrong_path = root / "wrong_gt.csv"
        _write_manifest(wrong_path, wrong_gt)
        try:
            val_rows(wrong_path)
            cases["gt_outside_split"] = False
        except ValueError:
            cases["gt_outside_split"] = True
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


def _amp_step(model, optimizer, scaler, x, gt):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
        output = model(x)
        loss = torch.nn.functional.l1_loss(output.float(), gt)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    skipped = bool(scaler.get_scale() < previous_scale)
    return loss, skipped


def cuda_full_shape(device: torch.device, timed_steps: int = 3) -> dict:
    if device.type != "cuda":
        raise RuntimeError("CUDA was requested but is unavailable.")
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats(device)
    model = DeepA().to(device).train()
    x = torch.rand(8, 3, 224, 448, device=device)
    gt = torch.rand(8, 3, 224, 448, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0)
    loss, skipped = _amp_step(model, optimizer, scaler, x, gt)
    if not torch.isfinite(loss):
        return record("cuda_full_shape", False, reason="nonfinite first loss")
    after_first = {name: param.detach().clone() for name, param in model.named_parameters()}
    torch.cuda.synchronize()
    started = time.perf_counter()
    skipped_any = False
    for _ in range(timed_steps):
        loss, skipped = _amp_step(model, optimizer, scaler, x, gt)
        skipped_any = skipped_any or skipped
        if not torch.isfinite(loss):
            return record("cuda_full_shape", False, reason="nonfinite later loss")
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - started) / max(1, timed_steps)
    grads_ok = all(param.grad is not None and torch.isfinite(param.grad).all()
                   for param in model.parameters() if param.requires_grad)
    backbone = [name for name, _ in model.named_parameters() if not name.startswith("head.")]
    backbone_updated = any(
        float((model.get_parameter(name).detach() - after_first[name]).abs().max()) > 0
        for name in backbone
    )
    head_updated = any(
        float((model.get_parameter(name).detach() - after_first[name]).abs().max()) > 0
        for name, _ in model.named_parameters() if name.startswith("head.")
    )
    passed = grads_ok and not skipped_any and backbone_updated and head_updated
    return record(
        "cuda_full_shape", passed, loss=float(loss.detach()),
        grads_finite=grads_ok, amp_skipped_step=skipped_any,
        backbone_updated_after_first_step=backbone_updated,
        head_updated_after_first_step=head_updated,
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
        check_architecture(),
        check_shapes(),
        check_identity_and_no_clamp(),
        check_spatial_fusion(),
        check_gradients(),
        check_serialize(),
        check_manifest(),
        check_help(),
    ]
    parameters = sum(p.numel() for p in DeepA().parameters())
    report = {
        "device": str(device),
        "parameters": parameters,
        "arch_version": "deep_a_v1",
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
