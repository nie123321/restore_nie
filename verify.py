"""Bounded implementation checks; never starts formal training or test evaluation."""
from __future__ import annotations
import copy
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import random
import time
import uuid
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import DEFAULT_DATA, PairedImages, StepBatches, augment_pair, inspect_split, read_rgb
from .model import ARCH_VERSION, GuidedRestoration, PriorAV2
from .priors import FixedPriors
from .runtime import (HERE, atomic_save, check_resume, compute_loss, learning_rate,
                      load_checkpoint, restore_rng, rng_state, seed_all, validate, write_json)
from .run import training_config


def default_training_args():
    return SimpleNamespace(data_root=DEFAULT_DATA, batch_size=4, crop_height=192, crop_width=384,
        seed=100, epochs=40, lr=2e-4, min_lr=2e-6, weight_decay=1e-4, grad_clip=1.,
        amp=True, workers=0, val_every=1000, save_every=1000, log_every=50)


@contextmanager
def smoke_checkpoint_directory():
    # Python 3.13 mkdtemp's private Windows ACL conflicts with the restricted
    # execution identity here. Normal mkdir inherits the workspace ACL.
    directory = HERE / "checks" / ("resume_smoke_" + uuid.uuid4().hex)
    directory.mkdir(parents=True)
    try:
        yield directory
    finally:
        (directory / "last.pt").unlink(missing_ok=True)
        directory.rmdir()


def cpu_checks(split):
    seed_all(100)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = PriorAV2().eval()
    parameters = sum(p.numel() for p in model.parameters())
    shapes = []
    with torch.no_grad():
        for h, w in ((17, 31), (192, 384), (224, 448), (225, 449)):
            low = torch.rand(1, 3, h, w)
            packed = model(low, return_aux=True)
            assert packed["output"].shape == low.shape
            assert all(torch.isfinite(v).all() for v in packed.values())
            assert torch.equal(packed["corrected"], low)
            assert torch.equal(packed["output"], low)
            assert torch.equal(packed["gain"], torch.ones_like(packed["gain"]))
            assert torch.count_nonzero(packed["bias"]) == 0
            shapes.append([h, w])
        # Stress the parameterization, independently of trained head values.
        low = torch.rand(2, 3, 16, 20)
        luma = [None, torch.rand(2, 32, 8, 10), torch.rand(2, 64, 4, 5)]
        model.brightness_head.bias.copy_(torch.tensor([100., -100.]))
        _, gain, bias = model.brightness(low, luma)
        assert gain.min() >= 1/8 - 1e-6 and gain.max() <= 8 + 1e-6
        assert bias.abs().max() <= .1 + 1e-7
        model.brightness_head.bias.copy_(torch.tensor([-100., 100.]))
        _, gain, bias = model.brightness(low, luma)
        assert gain.min() >= 1/8 - 1e-6 and gain.max() <= 8 + 1e-6
        assert bias.abs().max() <= .1 + 1e-7

    priors = FixedPriors()
    assert not list(priors.parameters())
    with torch.no_grad():
        for image in (torch.zeros(2, 3, 21, 33), torch.full((2, 3, 21, 33), .2),
                      torch.rand(2, 3, 21, 33)*1e-4):
            lp, sp = priors(image)
            assert lp.shape == (2, 6, 21, 33) and sp.shape == (2, 5, 21, 33)
            assert torch.isfinite(lp).all() and torch.isfinite(sp).all()
        dark_gradient_peak = float(sp[:, 3:].abs().max())
        assert dark_gradient_peak < .01
        ramp = torch.linspace(0, 1, 41)[None, None, None, :].expand(1, 3, 25, 41)
        grad = priors.signed_gradients(ramp)
        flipped = priors.signed_gradients(ramp.flip(-1))
        assert grad[:, 0, 6:-6, 6:-6].mean() > 0
        assert grad[:, 1, 6:-6, 6:-6].abs().max() < 1e-5
        assert torch.allclose(flipped[:, :1], -grad[:, :1].flip(-1), atol=2e-5)
        vertical = priors.signed_gradients(ramp.transpose(-1, -2))
        assert torch.allclose(vertical[:, 1:], grad[:, :1].transpose(-1, -2), atol=2e-5)

    block = GuidedRestoration(8, 4, 4, groups=4).eval()
    feature, structure, luma = torch.rand(2, 8, 5, 7), torch.rand(2, 4, 5, 7), torch.rand(2, 4, 5, 7)
    with torch.no_grad():
        weights = block.weights(feature, structure)
        assert (weights >= 0).all() and torch.allclose(weights.sum(2), torch.ones(2, 4, 5, 7), atol=1e-6)
        assert torch.equal(block.guided_residual(feature, structure, luma), feature)
        # Independent unfold reference verifies channel groups AND row-major offsets.
        patches = F.unfold(F.pad(feature, (1, 1, 1, 1), mode="replicate"), 3).reshape(2, 4, 2, 9, 5, 7)
        reference = (patches * weights[:, :, None]).sum(3).reshape_as(feature)
        assert torch.allclose(block.aggregate(feature, weights), reference, atol=1e-6)
        one_hot = torch.zeros_like(weights)
        for group in range(4):
            one_hot[:, group, group*2] = 1
        reference = (patches * one_hot[:, :, None]).sum(3).reshape_as(feature)
        assert torch.equal(block.aggregate(feature, one_hot), reference)

    low = torch.arange(3*24*40, dtype=torch.float32).reshape(3, 24, 40)
    target = low + 4096
    transforms, flipped_x, flipped_y = set(), False, False
    for epoch in range(20):
        a, b = augment_pair(low, target, (16, 28), 100, epoch, "paired-check")
        repeat, _ = augment_pair(low, target, (16, 28), 100, epoch, "paired-check")
        assert torch.equal(b-a, torch.full_like(a, 4096)) and torch.equal(a, repeat)
        transforms.add(tuple(a[0, :2, :2].flatten().tolist()))
        flipped_x |= bool(a[0, 0, 1] < a[0, 0, 0])
        flipped_y |= bool(a[0, 1, 0] < a[0, 0, 0])
    assert len(transforms) > 1 and flipped_x and flipped_y
    full = list(StepBatches(1500, 4, 100, 370, 380))
    assert full[4:] == list(StepBatches(1500, 4, 100, 374, 380))
    assert len({i for batch in StepBatches(1500, 4, 100, 0, 375) for i, _ in batch}) == 1500
    val = PairedImages(DEFAULT_DATA, split["splits"]["val"])
    a, _ = val[0]
    assert torch.equal(a, read_rgb(DEFAULT_DATA / split["splits"]["val"][0]["lowlight_relpath"]))
    target = torch.zeros(1, 3, 16, 20)
    loss, main, aux = compute_loss({"output": target+2, "corrected": target+.5}, target)
    assert abs(float(main)-2) < 1e-6 and abs(float(aux)-.5) < 1e-6 and abs(float(loss)-2.05) < 1e-6
    config = training_config(default_training_args(), split, torch.device("cuda"))
    assert config["steps"] == 15000 and config["steps_per_epoch"] == 375
    assert abs(learning_rate(config, 1)-2e-4) < 1e-12
    assert abs(learning_rate(config, 15000)-2e-6) < 1e-12
    saved_rng = rng_state()
    expected = (random.random(), float(np.random.rand()), torch.rand(3))
    restore_rng(saved_rng)
    assert random.random() == expected[0] and float(np.random.rand()) == expected[1]
    assert torch.equal(torch.rand(3), expected[2])
    return dict(passed=True, parameters=parameters, full_and_odd_shapes=shapes,
        neutral_initial_output=True, gain_and_bias_bounds=True, fixed_priors_finite=True,
        dark_signed_gradient_peak=dark_gradient_peak, direction_flip_consistent=True,
        neighborhood_matches_unfold=True, paired_augmentation=True, resume_sampler=True,
        validation_no_augmentation=True, raw_loss_formula=True, rng_restore=True,
        default_steps=15000)


def gradient_norms(model):
    groups = {"luma_encoder": model.luma_encoder, "structure_encoder": model.structure_encoder,
              "guided_logits": model.guide_half.logits, "guided_weight_features": model.guide_half.weight_features,
              "luma_coarse": model.luma_encoder.levels[2], "structure_coarse": model.structure_encoder.levels[2]}
    return {name: float(torch.stack([p.grad.float().square().sum() for p in module.parameters()
                if p.grad is not None]).sum().sqrt()) for name, module in groups.items()}


def gpu_smoke(split, steps):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: GPU smoke did not run")
    device = torch.device("cuda")
    seed_all(100)
    config = training_config(default_training_args(), split, device)
    sampler = StepBatches(1500, 4, 100, 0, steps)
    names = {s: [r["lowlight_relpath"] for r in split["splits"][s]] for s in ("train", "val", "test")}
    dataset = PairedImages(DEFAULT_DATA, split["splits"]["train"], training=True)
    batches = list(DataLoader(dataset, batch_sampler=sampler, num_workers=0,
                        generator=torch.Generator().manual_seed(100100)))
    model = PriorAV2().to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    rows = []

    def update(index, resumed=False):
        low, target = (x.to(device) for x in batches[index])
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(config, index+1)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            packed = model(low, return_aux=True)
            loss, main, aux = compute_loss(packed, target)
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        norms = gradient_norms(model)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        assert old_scale == scaler.get_scale(), "AMP skipped an update"
        row = dict(step=index+1, resumed=resumed, loss=float(loss.detach()), loss_main=float(main.detach()), loss_aux=float(aux.detach()),
                   grad_norm_before_clip=float(grad), gradients=norms, seconds=time.perf_counter()-started,
                   amp_skipped=False)
        rows.append(row)
        print(json.dumps(dict(smoke_step=row)), flush=True)

    checks_dir = HERE / "checks"
    checks_dir.mkdir(exist_ok=True)
    # A temporary checkpoint is serialized and reloaded; no trained run is created.
    with smoke_checkpoint_directory() as temporary:
        checkpoint = Path(temporary) / "last.pt"
        update(0)
        update(1)
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                     step=2, config=config, rng=rng_state(), sampler=sampler.state(2), filenames=names,
                     best_val=None, best_step=None)
        atomic_save(state, checkpoint)
        del state
        update(2)
        reference_loss = rows[-1]["loss"]
        reference = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        peak_allocated = torch.cuda.max_memory_allocated()/2**20
        peak_reserved = torch.cuda.max_memory_reserved()/2**20
        del model, optimizer, scaler
        gc.collect()
        torch.cuda.empty_cache()
        saved = load_checkpoint(checkpoint)
        check_resume(saved, config, sampler, names)
        changed = copy.deepcopy(config)
        changed["crop"][0] = 128
        try:
            check_resume(saved, changed, sampler, names)
        except ValueError:
            pass
        else:
            raise AssertionError("Changed crop was accepted on resume")
        model = PriorAV2(saved["config"]["model"]).to(device).train()
        model.load_state_dict(saved["model"], strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        optimizer.load_state_dict(saved["optimizer"])
        scaler = torch.amp.GradScaler("cuda")
        scaler.load_state_dict(saved["scaler"])
        restore_rng(saved["rng"])
        del saved
        update(2, resumed=True)
        max_difference = max(float((reference[k]-v.detach().cpu()).abs().max()) for k, v in model.state_dict().items())
        assert max_difference < 2e-6, f"Resume parameter difference {max_difference}"
        assert abs(reference_loss-rows[-1]["loss"]) < 2e-6
        for index in range(3, steps):
            update(index, resumed=True)
        assert all(norm > 0 for norm in rows[-1]["gradients"].values()), rows[-1]["gradients"]
        peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated()/2**20)
        peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved()/2**20)
        # Validate mechanics on one complete val image, without treating it as a quality result.
        optimizer.zero_grad(set_to_none=True)
        loader = DataLoader(PairedImages(DEFAULT_DATA, split["splits"]["val"][:1]), batch_size=1)
        metrics = validate(model, loader, device)
        assert metrics["val_count"] == 1 and model.training
        assert np.isfinite(metrics["val_l1"]) and np.isfinite(metrics["val_psnr_rgb_float"])
        result = dict(passed=True, gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
            batch=4, crop=[192, 384], amp=True, activation_checkpointing=True,
            optimizer_updates=len(rows), logical_steps=steps, includes_one_repeated_resume_step=True,
            peak_training_allocated_mib=peak_allocated, peak_training_reserved_mib=peak_reserved,
            peak_including_single_val_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            resume_max_parameter_difference=max_difference, strict_resume_rejects_crop_change=True,
            both_prior_encoders_and_guidance_receive_gradients=True, full_image_val_smoke_count=1,
            test_images_opened=0, steps=rows)
        del model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return result


def verify(args):
    split = inspect_split(DEFAULT_DATA)
    result = dict(arch_version=ARCH_VERSION, manifest_sha256=split["manifest_sha256"], counts=split["counts"],
                  formal_training_started=False, cpu=cpu_checks(split))
    print(json.dumps({"cpu": result["cpu"]}), flush=True)
    if args.gpu_smoke:
        # Three logical steps plus one repeated step = four total optimizer updates.
        result["gpu"] = gpu_smoke(split, args.gpu_steps)
    path = HERE / "checks" / ("verification_gpu.json" if args.gpu_smoke else "verification_cpu.json")
    path.parent.mkdir(exist_ok=True)
    write_json(path, result)
    print(json.dumps({"passed": True, "result": str(path)}), flush=True)
    return result
