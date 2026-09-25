"""Bounded CPU checks for Prior-Fusion-U3. No optimizer and no real images."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from unittest.mock import patch
from pathlib import Path

import torch
import torch.nn.functional as F

from model import GlobalSpatialBlock, LocalPriorBlock, PriorFusionU3, PriorResBlock, default_config, _spatial_attention
from priors import six_luminance
from run import atomic_save, build_model


ROOT = Path(__file__).resolve().parent
STATIC = {
    "local64": 253448, "local128": 1423896, "local256": 1834000,
    "global": 1354240, "luma_encoder": 870496, "structure_encoder": 869056,
    "total": 7145363,
}


def record(name: str, passed: bool, **details) -> dict:
    return {"name": name, "passed": bool(passed), **details}


def finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all())


def counts(model: PriorFusionU3) -> dict:
    return {
        "local": sum(isinstance(m, LocalPriorBlock) for m in model.modules()),
        "global": sum(isinstance(m, GlobalSpatialBlock) for m in model.modules()),
        "prior_res": sum(isinstance(m, PriorResBlock) for m in model.modules()),
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def check_shapes() -> dict:
    model = PriorFusionU3().eval()
    info = counts(model)
    ok = info["local"] == 12 and info["global"] == 2 and info["prior_res"] == 12
    details = {}
    with torch.no_grad():
        for size in ((32, 48), (31, 47)):
            image = torch.rand(1, 3, *size)
            debug = model(image, return_debug=True)
            output = debug["output"]
            scales = [tuple(t.shape[-2:]) for t in debug["luma"]]
            details[f"{size[0]}x{size[1]}"] = {
                "output": list(output.shape) == [1, 3, *size] and finite(output),
                "luma_scales": scales,
                "structure_scales": [tuple(t.shape[-2:]) for t in debug["structure"]],
            }
            ok = ok and details[f"{size[0]}x{size[1]}"]["output"]
            ok = ok and scales[0] == size and scales[1][0] == (size[0] + 1) // 2
    return record("shapes", ok, counts=info, details=details)


def check_boundaries() -> dict:
    model = PriorFusionU3().eval()
    black = torch.zeros(1, 3, 8, 8)
    constant = torch.full((1, 3, 8, 8), 0.4)
    luma = six_luminance(constant)
    structure = model.structure_prior(torch.rand(1, 3, 16, 16))
    with torch.no_grad():
        outputs = [model(black), model(constant), model(torch.rand(1, 3, 8, 10))]
    luma_not_normalized = abs(float(luma[0, 0].mean()) - 0.4) < 1e-5
    structure_signed = bool((structure < 0).any() and (structure > 0).any())
    ok = all(finite(item) for item in outputs) and finite(luma) and finite(structure)
    ok = ok and luma_not_normalized and structure_signed
    return record("boundaries", ok, luma_mean=float(luma[0, 0].mean()),
                  structure_min=float(structure.min()), structure_max=float(structure.max()))


def check_dependencies() -> dict:
    torch.manual_seed(0)
    model = PriorFusionU3().eval()
    image = torch.rand(1, 3, 16, 20)
    with torch.no_grad():
        base = model(image, return_debug=True)
        luma = base["luma"][2]
        structure = base["structure"][2]
        feature = torch.randn(1, 256, *luma.shape[-2:])
        changed_l = model.b0(feature, luma + 0.5, structure)
        changed_s = model.b0(feature, luma, structure + 0.5)
        same = model.b0(feature, luma, structure)
    luma_delta = float((changed_l - same).abs().max())
    structure_delta = float((changed_s - same).abs().max())
    return record("dependencies", luma_delta > 0 and structure_delta > 0,
                  luma_delta=luma_delta, structure_delta=structure_delta)


def check_gradients() -> dict:
    torch.manual_seed(1)
    model = PriorFusionU3().train()
    image = torch.rand(1, 3, 16, 20)
    target = torch.rand(1, 3, 16, 20)
    loss = (model(image) - target).abs().mean()
    loss.backward()
    groups = {
        "luma_encoder": model.luma_encoder,
        "structure_encoder": model.structure_encoder,
        "structure_kv": model.b0.kv_proj,
        "luma_head": model.b0.gamma_out,
        "local": model.e0[0],
        "global": model.b1,
        "skip": model.skip0,
    }
    report = {}
    passed = finite(loss)
    for name, module in groups.items():
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        ok = bool(grads) and all(g is not None and torch.isfinite(g).all() and float(g.abs().mean()) > 0 for g in grads)
        report[name] = ok
        passed = passed and ok
    return record("gradients", passed, loss=float(loss.detach()), groups=report)


def check_attention_axes() -> dict:
    torch.manual_seed(2)
    query = torch.randn(1, 4, 3, 5)
    key = torch.randn(1, 4, 3, 5)
    value = torch.randn(1, 4, 3, 5)
    q = query.reshape(1, 1, 4, 15).float()
    k = key.reshape(1, 1, 4, 15).float()
    v = value.reshape(1, 1, 4, 15).float()
    q = F.normalize(q, dim=-1, eps=1e-6)
    k = F.normalize(k, dim=-1, eps=1e-6)
    manual = torch.softmax(torch.matmul(q, k.transpose(-2, -1)), dim=-1)
    channel_axis = manual.shape[-1] == 4
    spatial_q = torch.randn(1, 8, 6, 32)
    spatial_k = torch.randn(1, 8, 3, 32)
    spatial_v = torch.randn(1, 8, 3, 32)
    logits = torch.matmul(spatial_q, spatial_k.transpose(-2, -1)) / (32 ** 0.5)
    spatial = torch.softmax(logits, dim=-1)
    sdpa = F.scaled_dot_product_attention(spatial_q, spatial_k, spatial_v, dropout_p=0.0, is_causal=False)
    manual_out = torch.matmul(spatial, spatial_v)
    err = float((sdpa - manual_out).abs().max())
    return record("attention_axes", channel_axis and spatial.shape[-1] == 3 and err < 1e-5,
                  channel_softmax_dim=manual.shape[-1], spatial_softmax_dim=spatial.shape[-1], sdpa_error=err)


def check_config() -> dict:
    real = PriorFusionU3()
    zero = PriorFusionU3({**default_config(), "luma_input_mode": "zero", "structure_input_mode": "zero"})
    same_count = counts(real)["parameters"] == counts(zero)["parameters"] == STATIC["total"]
    image = torch.rand(1, 3, 12, 14)
    with torch.no_grad():
        first = real(image)
    clone = build_model(real.config())
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        atomic_save({"model": real.state_dict(), "config": {"arch_version": "prior_fusion_u3_v1", "model": real.config()}}, path)
        saved = torch.load(path, map_location="cpu", weights_only=False)
    clone.load_state_dict(saved["model"], strict=True)
    with torch.no_grad():
        err = float((first - clone(image)).abs().max())
    return record("config", same_count and err < 1e-6 and saved["config"]["model"]["arch_version"] == "prior_fusion_u3_v1",
                  parameters=counts(real)["parameters"], static_total=STATIC["total"], restore_error=err)


def check_help() -> dict:
    ok = {}
    for args in (
        [sys.executable, str(ROOT / "run.py"), "--help"],
        [sys.executable, str(ROOT / "run.py"), "infer", "--help"],
        [sys.executable, str(ROOT / "eval_val.py"), "--help"],
    ):
        completed = subprocess.run(args, capture_output=True, text=True)
        ok[" ".join(args[-2:])] = completed.returncode == 0
    bare = subprocess.run([sys.executable, str(ROOT / "run.py"), "train"], capture_output=True, text=True)
    ok["train_requires_steps"] = bare.returncode != 0
    return record("help", all(ok.values()), commands=ok)


def check_memory_execution() -> dict:
    """Guard AMP residual dtype, recomputation gradients, and fused-SDPA layout."""
    torch.manual_seed(23)
    plain = PriorFusionU3({"activation_checkpointing": False}).train()
    recompute = PriorFusionU3({"activation_checkpointing": True}).train()
    recompute.load_state_dict(plain.state_dict(), strict=True)
    image = torch.rand(1, 3, 8, 12)
    target = torch.rand_like(image)
    observed = []
    handles = [recompute.get_submodule(name).register_forward_hook(
        lambda module, inputs, output: observed.append(output.dtype))
        for name in ("luma_encoder.block0", "structure_encoder.block0", "e0.0", "b1", "d0.0")]
    outputs = []
    for net in (plain, recompute):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = net(image)
            loss = (output - target).square().mean()
        loss.backward()
        outputs.append(output.detach())
    for handle in handles:
        handle.remove()
    gradients_equal = all(p.grad is not None and q.grad is not None
                          and finite(p.grad) and finite(q.grad) and torch.equal(p.grad, q.grad)
                          for p, q in zip(plain.parameters(), recompute.parameters()))
    sdpa = F.scaled_dot_product_attention
    strides = []

    def capture(q, k, v, **kwargs):
        strides.extend([q.stride(-1), k.stride(-1), v.stride(-1)])
        return sdpa(q, k, v, **kwargs)

    with torch.no_grad(), patch("model.F.scaled_dot_product_attention", side_effect=capture):
        _spatial_attention(torch.rand(1, 256, 8, 12), torch.rand(1, 256, 2, 3),
                           torch.rand(1, 256, 2, 3), heads=8, head_dim=32)
    legacy = plain.config()
    legacy.pop("activation_checkpointing")
    legacy_load = build_model(legacy)
    legacy_load.load_state_dict(plain.state_dict(), strict=True)
    passed = (torch.equal(*outputs) and gradients_equal and bool(observed)
              and all(dtype == torch.bfloat16 for dtype in observed) and strides == [1, 1, 1])
    return record("memory_execution", passed, checkpoint_gradients_equal=gradients_equal,
                  amp_block_dtype="bfloat16", sdpa_last_strides=strides,
                  legacy_weights_loaded=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.threads))
    checks = [check_shapes(), check_boundaries(), check_dependencies(), check_gradients(),
              check_attention_axes(), check_config(), check_help(), check_memory_execution()]
    print(json.dumps({"passed": all(item["passed"] for item in checks),
                      "parameters": checks[0]["counts"]["parameters"],
                      "failed": [item["name"] for item in checks if not item["passed"]]}, indent=2))
    out = ROOT / "checks" / "cpu_checks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"passed": all(c["passed"] for c in checks), "checks": checks}, indent=2) + "\n", encoding="utf-8")
    if not all(item["passed"] for item in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
