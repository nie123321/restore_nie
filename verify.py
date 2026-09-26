"""Bounded CPU compatibility and optional real-batch AMP checks."""
from __future__ import annotations
import importlib.util
import io
import json
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader
from .data import DEFAULT_DATA, PairedImages, StepBatches, inspect_split
from .model import ARCH_VERSION, LocalPriorBlock, PriorAU3, PriorSkipFusion
from .priors import FixedPriors
from .runtime import HERE, compute_loss, seed_all, write_json


def load_reference():
    directory = HERE.parent / "endo_prior_fusion_u3"
    old_path = list(sys.path)
    old_priors = sys.modules.pop("priors", None)
    try:
        sys.path.insert(0, str(directory))
        spec = importlib.util.spec_from_file_location("u3_reference", directory / "model.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path
        sys.modules.pop("priors", None)
        if old_priors is not None:
            sys.modules["priors"] = old_priors


def verify(args):
    seed_all(100)
    torch.set_num_threads(4)
    reference = load_reference()
    deltas = {}
    with torch.no_grad():
        for cls in (LocalPriorBlock, PriorSkipFusion):
            original = getattr(reference, cls.__name__)(64).eval()
            copied = cls(64).eval()
            copied.load_state_dict(original.state_dict(), strict=True)
            f = torch.randn(2, 64, 12, 20)
            lp, sp = torch.randn(2, 32, 12, 20), torch.randn(2, 32, 12, 20)
            inputs = (f, lp, sp) if cls is LocalPriorBlock else (f, f*.7, lp, sp)
            before, after = original(*inputs), copied(*inputs)
            delta = float((before-after).abs().max())
            assert delta == 0, (cls.__name__, delta)
            deltas[cls.__name__] = delta
        priors = FixedPriors()
        assert not list(priors.parameters())
        rgb = torch.rand(2, 3, 31, 45)
        lp, sp = priors(rgb)
        assert torch.equal(lp, reference.six_luminance(rgb))
        assert torch.equal(sp, reference.CIConvW()(rgb))
        for rgb in (torch.zeros(1, 3, 16, 20), torch.full((1, 3, 16, 20), .1)):
            assert all(torch.isfinite(x).all() for x in priors(rgb))
        model = PriorAU3().eval()
        shape_results = []
        for h, w in ((192, 384), (224, 448), (225, 449)):
            rgb = torch.rand(1, 3, h, w)
            output = model(rgb)
            assert output.shape == rgb.shape and torch.equal(output, rgb)
            shape_results.append([h, w])
        assert model.stem.in_channels == 3 and model.structure_encoder.stem.in_channels == 1
        assert not any("brightness" in name or "weight_features" in name for name, _ in model.named_parameters())
        assert float(compute_loss(torch.ones(1, 3, 8, 8)*2, torch.zeros(1, 3, 8, 8))) == 2.
        serialized = io.BytesIO()
        torch.save(model.state_dict(), serialized)
        serialized.seek(0)
        restored = PriorAU3(model.config())
        restored.load_state_dict(torch.load(serialized, weights_only=True), strict=True)
    result = dict(arch_version=ARCH_VERSION, parameters=sum(p.numel() for p in model.parameters()),
                  cpu_passed=True, u3_module_max_difference=deltas, fixed_priors_match_u3=True,
                  neutral_initial_output=True, input_shapes=shape_results, pure_raw_L1=True,
                  strict_state_roundtrip=True, formal_training_started=False)
    print(json.dumps(result), flush=True)
    if args.gpu_smoke:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        seed_all(100)
        split = inspect_split(DEFAULT_DATA)
        dataset = PairedImages(DEFAULT_DATA, split["splits"]["train"], training=True)
        loader = DataLoader(dataset, batch_sampler=StepBatches(1500, 8, 100, 0, args.gpu_steps),
                            num_workers=0, generator=torch.Generator().manual_seed(100100))
        model = PriorAU3().cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        rows = []
        for step, (low, target) in enumerate(loader, 1):
            low, target = low.cuda(), target.cuda()
            assert low.shape == (8, 3, 192, 384)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                output = model(low)
                loss = compute_loss(output, target)
            assert torch.isfinite(loss)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            branches = {"luma": model.luma_encoder, "structure": model.structure_encoder,
                        "luma_coarse": model.luma_encoder.levels[2], "structure_coarse": model.structure_encoder.levels[2]}
            gradients = {name: float(torch.stack([p.grad.float().square().sum() for p in block.parameters()
                         if p.grad is not None]).sum().sqrt()) for name, block in branches.items()}
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            assert scaler.get_scale() == old_scale, "AMP skipped an update"
            row = dict(step=step, loss=float(loss.detach()), grad_norm=float(norm), gradients=gradients,
                       seconds=time.perf_counter()-start, amp_skipped=False)
            rows.append(row)
            print(json.dumps(row), flush=True)
            del low, target, output, loss
        assert all(value > 0 for value in rows[-1]["gradients"].values())
        result["gpu"] = dict(passed=True, batch=8, crop=[192, 384], amp=True,
            activation_checkpointing=True, steps=rows,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    destination = HERE / "checks" / ("verification_gpu.json" if args.gpu_smoke else "verification_cpu.json")
    destination.parent.mkdir(exist_ok=True)
    write_json(destination, result)
    print(json.dumps({"passed": True, "path": str(destination), "parameters": result["parameters"],
                      "memory": {k:v for k,v in result.get("gpu", {}).items() if k != "steps"}}), flush=True)
