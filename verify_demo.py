"""Bounded numerical checks for the enhancement prototype; no dataset is read."""

from __future__ import annotations

import argparse
import io
import json
import sys

import torch
import torch.nn.functional as F

from model import ConditionalSpectralBlock, EnhancementDemo


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def finite(tensor: torch.Tensor, name: str) -> None:
    require(bool(torch.isfinite(tensor).all()), f"{name} contains nonfinite values")


def build(device: torch.device, spectral_mode: str, output_mode: str):
    return EnhancementDemo(
        width=24, spectral_mode=spectral_mode, output_mode=output_mode
    ).to(device)


def check_spectral_conditioning(device: torch.device) -> dict:
    """Exercise routing independently of the zero-initialized image output heads."""
    channels = 8
    block = ConditionalSpectralBlock(channels, mode="conditional").to(device)
    condition = torch.zeros((2, channels + 2), device=device)
    condition[1, 0] = 1.0
    with torch.no_grad():
        initial_response, initial_weights = block.frequency_response(condition)
    require(
        torch.allclose(initial_response, torch.ones_like(initial_response), atol=1e-6),
        "Conditional spectral initialization is not neutral",
    )
    require(
        torch.allclose(initial_weights, torch.full_like(initial_weights, 1 / 3)),
        "Initial spectral routing is not uniform",
    )
    # A deliberate diagnostic intervention tests capacity, not learned behavior.
    with torch.no_grad():
        block.router.weight.zero_()
        block.router.bias.zero_()
        block.router.weight[0, 0] = 2.0
        block.router.weight[1, 0] = -2.0
    condition.requires_grad_()
    response, routing = block.frequency_response(condition)
    finite(response, "conditioned frequency response")
    require(response.dtype == torch.float32, "Frequency response is not FP32")
    require(bool(((response >= 0.5) & (response <= 1.5)).all()), "Filter bound violation")
    require(not torch.allclose(routing[0], routing[1]), "Routing ignores condition")
    delta = response[1] - response[0]
    require(delta.abs().max().item() > 1e-6, "Condition does not change response")
    frequency_variation = delta.flatten(1).std(dim=1).max().item()
    require(frequency_variation > 1e-6, "Condition only applies a frequency-independent scale")
    feature = torch.randn((1, channels, 11, 7), device=device).repeat(2, 1, 1, 1)
    restored, reported_weights = block(feature, condition)
    require(restored.shape == feature.shape, "Spectral padding/crop changes feature shape")
    finite(restored, "spectral features")
    require(torch.allclose(routing, reported_weights), "Reported routing is inconsistent")
    require(
        (restored[1] - restored[0]).abs().max().item() > 1e-7,
        "Different routing has no effect on matched input features",
    )
    restored.square().mean().backward()
    for name in ("router.weight", "filter_basis"):
        gradient = dict(block.named_parameters())[name].grad
        require(gradient is not None, f"No spectral gradient for {name}")
        finite(gradient, f"spectral gradient for {name}")
        require(gradient.abs().max().item() > 0, f"Zero spectral gradient for {name}")
    for mode in ("static", "off"):
        other = ConditionalSpectralBlock(channels, mode=mode).to(device)
        with torch.no_grad():
            other_response, other_weights = other.frequency_response(condition.detach())
        require(other_weights is None, f"{mode} unexpectedly reports dynamic weights")
        if mode == "static":
            require(torch.equal(other_response[0], other_response[1]), "Static response varies by input")
        else:
            require(other_response is None, "Off mode still produces a frequency response")
    return {
        "spectral_routing_capacity_verified": True,
        "conditioned_frequency_variation": frequency_variation,
        "spectral_check_uses_synthetic_router_weights": True,
    }


def check_aux(aux: dict, x: torch.Tensor) -> dict:
    batch, _, height, width = x.shape
    expected = {
        "output": x.shape,
        "coarse": x.shape,
        "gain": (batch, 1, height, width),
        "global_color_gain": (batch, 3, 1, 1),
        "luminance_residual": (batch, 1, height, width),
        "chroma_residual": x.shape,
    }
    for name, shape in expected.items():
        require(name in aux, f"Missing auxiliary output: {name}")
        require(tuple(aux[name].shape) == tuple(shape), f"Unexpected {name} shape")
        finite(aux[name], name)
    global_gain = aux["global_color_gain"]
    require(bool((global_gain > 0).all()), "Global color gains must be positive")
    gain_error = global_gain.log().sum(dim=1).abs().max().item()
    require(gain_error < 2e-5, "Global log gains do not sum to zero")
    weights = x.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
    chroma_error = (weights * aux["chroma_residual"]).sum(1).abs().max().item()
    require(chroma_error < 2e-5, "Chroma residual changes defined luminance")
    reconstructed_coarse = x * aux["gain"] * global_gain
    require(
        torch.allclose(aux["coarse"], reconstructed_coarse, atol=2e-5, rtol=2e-5),
        "Coarse result does not match reported gains",
    )
    reconstructed = (
        aux["coarse"] + aux["luminance_residual"] + aux["chroma_residual"]
    )
    require(
        torch.allclose(aux["output"], reconstructed, atol=2e-5, rtol=2e-5),
        "Structured output does not match reported additive residuals",
    )
    spectral_weights = aux.get("spectral_weights")
    if spectral_weights is not None:
        require(spectral_weights.ndim == 2, "Spectral weights must have shape B,K")
        require(spectral_weights.shape[0] == batch, "Spectral weight batch mismatch")
        finite(spectral_weights, "spectral_weights")
    return {"global_log_gain_error": gain_error, "chroma_luminance_error": chroma_error}


def run_checks(device: torch.device) -> dict:
    torch.manual_seed(20260921)
    report = {"device": str(device), "threads": torch.get_num_threads()}
    shape_checks = 0
    identity_error = 0.0
    for spectral_mode in ("conditional", "static", "off"):
        for output_mode in ("structured", "direct"):
            model = build(device, spectral_mode, output_mode).eval()
            for shape in ((2, 3, 17, 29), (1, 3, 1, 1)):
                x = torch.rand(shape, device=device)
                with torch.no_grad():
                    output = model(x)
                require(output.shape == x.shape, "Output shape differs from input")
                finite(output, "output")
                error = (output - x).abs().max().item()
                require(error < 2e-6, f"Initialization is not identity: {error}")
                identity_error = max(identity_error, error)
                if output_mode == "structured":
                    with torch.no_grad():
                        check_aux(model(x, return_aux=True), x)
                shape_checks += 1
    report["shape_checks"] = shape_checks
    report["initial_identity_max_error"] = identity_error

    model = build(device, "conditional", "structured").train()
    report["parameters"] = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.rand((2, 3, 19, 27), device=device)
    target = torch.rand_like(x)
    step_losses = []
    first_active = set()
    later_active = set()
    auxiliary_gradients = {}
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        aux = model(x, return_aux=True)
        check_aux(aux, x)
        observed = ("coarse", "luminance_residual", "chroma_residual")
        for name in observed:
            require(aux[name].requires_grad, f"{name} is detached")
            aux[name].retain_grad()
        loss = F.l1_loss(aux["output"], target)
        finite(loss, "loss")
        loss.backward()
        active = set()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                finite(parameter.grad, f"gradient of {name}")
                if parameter.grad.abs().max().item() > 0:
                    active.add(name)
        require(bool(active), "No parameter receives a nonzero gradient")
        if step == 0:
            first_active = active
            for head_name in (
                "gain_head", "global_color_head", "luminance_head", "chroma_head"
            ):
                require(
                    any(name.startswith(head_name + ".") for name in active),
                    f"{head_name} receives no nonzero parameter gradient",
                )
            for name in observed:
                gradient = aux[name].grad
                require(gradient is not None, f"No gradient for {name}")
                finite(gradient, f"gradient of {name}")
                norm = gradient.abs().sum().item()
                require(norm > 0, f"Zero gradient for {name}")
                auxiliary_gradients[name] = norm
        else:
            later_active |= active
        optimizer.step()
        step_losses.append(loss.item())
    newly_active = later_active - first_active
    require(bool(newly_active), "Gradients did not propagate beyond zero-initialized heads")
    backbone_gradients = [
        name for name in newly_active if name.startswith("stem.")
    ]
    require(bool(backbone_gradients), "Stem did not receive gradients after head updates")
    report["bounded_optimization_steps"] = len(step_losses)
    report["losses"] = step_losses
    report["auxiliary_gradient_l1"] = auxiliary_gradients
    report["newly_active_parameter_tensors"] = len(newly_active)
    report["stem_gradient_verified"] = True

    model.eval()
    with torch.no_grad():
        report.update(check_aux(model(x, return_aux=True), x))
        expected = model(x)
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = build(device, "conditional", "structured").eval()
    restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    with torch.no_grad():
        actual = restored(x)
    checkpoint_error = (expected - actual).abs().max().item()
    require(checkpoint_error < 1e-7, "Checkpoint round trip changed output")
    report["checkpoint_max_error"] = checkpoint_error

    # In-range inputs and an explicit additive head bias must remain unclipped.
    for output_mode, head_name in (
        ("structured", "luminance_head"),
        ("direct", "direct_head"),
    ):
        clipping_model = build(device, "off", output_mode).eval()
        head = getattr(clipping_model, head_name, None)
        require(head is not None, f"Cannot locate {head_name} for clipping check")
        convolutions = [module for module in head.modules() if isinstance(module, torch.nn.Conv2d)]
        require(bool(convolutions), f"{head_name} has no convolution")
        last = convolutions[-1]
        require(last.bias is not None, f"{head_name} needs bias for clipping check")
        with torch.no_grad():
            last.weight.zero_()
            last.bias.fill_(2.0)
            above_one = clipping_model(torch.full((1, 3, 5, 7), 0.5, device=device))
            last.bias.fill_(-2.0)
            below_zero = clipping_model(torch.full((1, 3, 5, 7), 0.5, device=device))
        finite(above_one, "unclipped positive output")
        finite(below_zero, "unclipped negative output")
        require(above_one.min().item() > 1.0, f"{output_mode} clips output above one")
        require(below_zero.max().item() < 0.0, f"{output_mode} clips output below zero")
    report["raw_output_unclipped"] = True
    report.update(check_spectral_conditioning(device))
    report["passed"] = True
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    try:
        require(args.threads > 0, "--threads must be positive")
        torch.set_num_threads(args.threads)
        require(args.device != "cuda" or torch.cuda.is_available(), "CUDA is unavailable")
        result = run_checks(torch.device(args.device))
    except Exception as error:
        print(json.dumps({"passed": False, "error": f"{type(error).__name__}: {error}"}))
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
