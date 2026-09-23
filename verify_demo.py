"""Bounded numerical checks for the enhancement prototype; no dataset is read."""

from __future__ import annotations

import argparse
import io
import json
import sys

import torch
import torch.nn.functional as F

from model import ConditionalSpectralBlock, EnhancementDemo, RGBInteractionHead


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


def _weight_grad_peak(model: torch.nn.Module, prefix: str) -> float:
    peak = None
    for name, parameter in model.named_parameters():
        if not (name.startswith(prefix) and name.endswith("weight")):
            continue
        require(parameter.grad is not None, f"No gradient for {name}")
        finite(parameter.grad, f"gradient of {name}")
        value = parameter.grad.detach().abs().max().item()
        peak = value if peak is None else max(peak, value)
    require(peak is not None, f"No weights found for {prefix}")
    return peak


def _assert_interaction_structure(head: RGBInteractionHead, width: int) -> None:
    require(len(head.branches) == 3 and len(head.residual) == 3, "RGB head branch count")
    branch_ptrs = []
    for branch in head.branches:
        layers = list(branch.children())
        require(len(layers) == 4, "Each colour branch must be conv, GELU, conv, GELU")
        conv1, activation1, conv2, activation2 = layers
        require(isinstance(activation1, torch.nn.GELU) and isinstance(activation2, torch.nn.GELU),
                "Branch activations must be GELU")
        require(conv1.in_channels == width + 1 and conv1.out_channels == 8, "First branch conv shape")
        require(conv2.in_channels == 8 and conv2.out_channels == 8, "Second branch conv shape")
        for convolution in (conv1, conv2):
            require(convolution.kernel_size == (3, 3), "Branch kernel")
            require(convolution.stride == (1, 1) and convolution.padding == (1, 1), "Branch geometry")
            require(convolution.groups == 1 and convolution.bias is not None, "Branch convolution")
        require(torch.count_nonzero(conv1.weight).item() > 0, "Branch was zero-initialized")
        branch_ptrs.append(conv1.weight.data_ptr())
    require(len(set(branch_ptrs)) == 3, "Colour branches share weights")
    residual_ptrs = []
    for layer in head.residual:
        require(layer.in_channels == 8 and layer.out_channels == 1, "Residual conv shape")
        require(layer.kernel_size == (3, 3) and layer.stride == (1, 1) and layer.padding == (1, 1),
                "Residual geometry")
        require(torch.count_nonzero(layer.weight).item() == 0, "Residual weight is not zero")
        require(torch.count_nonzero(layer.bias).item() == 0, "Residual bias is not zero")
        residual_ptrs.append(layer.weight.data_ptr())
    require(len(set(residual_ptrs)) == 3, "Residual convolutions share weights")
    require(head.mix.in_channels == 24 and head.mix.out_channels == 24, "Mix shape")
    require(head.mix.kernel_size == (1, 1) and head.mix.stride == (1, 1) and head.mix.padding == (0, 0),
            "Mix geometry")
    require(head.gate.in_channels == 24 and head.gate.out_channels == 3, "Gate shape")
    require(head.gate.kernel_size == (1, 1) and head.gate.stride == (1, 1), "Gate geometry")
    require(torch.count_nonzero(head.mix.weight).item() > 0, "Mix was zero-initialized")
    require(torch.count_nonzero(head.gate.weight).item() > 0, "Gate was zero-initialized")


def check_rgb_interaction(device: torch.device) -> dict:
    """Shape, identity, gradients, gate intervention and serialization."""
    result = {"device": str(device)}
    rejected = False
    try:
        EnhancementDemo(output_mode="rgb")
    except ValueError:
        rejected = True
    require(rejected, "A truncated output mode name was accepted")

    default = EnhancementDemo()
    direct = build(device, "off", "direct")
    rgb = build(device, "off", "rgb_interaction")
    direct_parameters = sum(parameter.numel() for parameter in direct.parameters())
    rgb_parameters = sum(parameter.numel() for parameter in rgb.parameters())
    default_parameters = sum(parameter.numel() for parameter in default.parameters())
    direct_head_parameters = sum(parameter.numel() for parameter in direct.direct_head.parameters())
    interaction_parameters = sum(parameter.numel() for parameter in rgb.interaction_head.parameters())
    require(direct_parameters == 206067, f"direct/off parameter count changed: {direct_parameters}")
    require(default_parameters == 223024, f"default parameter count changed: {default_parameters}")
    require(default.config == dict(width=24, spectral_mode="conditional", output_mode="structured"),
            "Default config changed")
    require(rgb_parameters == direct_parameters - direct_head_parameters + interaction_parameters,
            "rgb_interaction did not replace only the direct head")
    require("interaction_head" not in {name for name, _ in direct.named_children()},
            "direct mode instantiates the interaction head")
    require("interaction_head" not in {name for name, _ in default.named_children()},
            "structured mode instantiates the interaction head")
    require("direct_head" not in {name for name, _ in rgb.named_children()},
            "rgb_interaction still instantiates direct_head")
    require(not any(name.startswith("interaction_head.") for name, _ in direct.named_parameters()),
            "direct parameters include the interaction head")
    require(not any(name.startswith("direct_head.") for name, _ in rgb.named_parameters()),
            "rgb_interaction parameters include direct_head")
    _assert_interaction_structure(rgb.interaction_head, 24)
    result["parameters"] = {
        "direct_off": direct_parameters,
        "rgb_interaction_off": rgb_parameters,
        "rgb_interaction_head": interaction_parameters,
        "structured_conditional": default_parameters,
    }

    torch.manual_seed(20260923)
    identity_error = 0.0
    shape_checks = 0
    aux_keys = {
        "output", "coarse", "gain", "global_color_gain",
        "luminance_residual", "chroma_residual", "spectral_weights",
    }
    for spectral_mode in ("off", "static", "conditional"):
        model = build(device, spectral_mode, "rgb_interaction").eval()
        require(
            model.config == dict(width=24, spectral_mode=spectral_mode, output_mode="rgb_interaction"),
            "rgb config mismatch",
        )
        for shape in ((2, 3, 16, 32), (2, 3, 17, 29), (1, 3, 1, 1)):
            sample = torch.rand(shape, device=device)
            with torch.no_grad():
                output = model(sample)
                aux = model(sample, return_aux=True)
            require(tuple(output.shape) == shape, "rgb output shape differs from input")
            finite(output, "rgb output")
            error = (output - sample).abs().max().item()
            require(error < 2e-6, f"rgb initialization is not identity: {error}")
            identity_error = max(identity_error, error)
            require(set(aux) == aux_keys, f"rgb auxiliary keys changed: {sorted(aux)}")
            check_aux(aux, sample)
            require(torch.equal(aux["coarse"], sample), "rgb coarse is not the input")
            require(torch.equal(aux["gain"], torch.ones_like(aux["gain"])), "rgb gain is not neutral")
            require(torch.equal(aux["global_color_gain"], torch.ones_like(aux["global_color_gain"])),
                    "rgb colour gain is not neutral")
            if spectral_mode == "off":
                require(aux["spectral_weights"] is None, "off mode reports spectral weights")
            elif spectral_mode == "conditional":
                require(aux["spectral_weights"] is not None, "conditional mode has no spectral weights")
            shape_checks += 1
    narrow = EnhancementDemo(width=8, spectral_mode="off", output_mode="rgb_interaction").to(device).eval()
    _assert_interaction_structure(narrow.interaction_head, 8)
    narrow_sample = torch.rand((1, 3, 3, 5), device=device)
    with torch.no_grad():
        narrow_output = narrow(narrow_sample)
    finite(narrow_output, "narrow rgb output")
    require(narrow_output.shape == narrow_sample.shape, "narrow rgb shape")
    narrow_error = (narrow_output - narrow_sample).abs().max().item()
    require(narrow_error < 2e-6, f"narrow rgb initialization is not identity: {narrow_error}")
    result["shape_checks"] = shape_checks
    result["initial_identity_max_error"] = max(identity_error, narrow_error)

    torch.manual_seed(20260924)
    model = build(device, "off", "rgb_interaction").train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sample = torch.rand((2, 3, 19, 27), device=device)
    target = torch.rand_like(sample)
    losses = []
    gain_error = 0.0
    chroma_error = 0.0
    step0_residual = 0.0
    step0_upstream = 0.0
    later_weight_peak = {"branches": 0.0, "mix": 0.0, "gate": 0.0, "stem": 0.0, "residual": 0.0}
    later_branch_peak = [0.0, 0.0, 0.0]
    first_upstream_step = None
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        aux = model(sample, return_aux=True)
        stats = check_aux(aux, sample)
        gain_error = max(gain_error, stats["global_log_gain_error"])
        chroma_error = max(chroma_error, stats["chroma_luminance_error"])
        require(torch.equal(aux["coarse"], sample), "rgb coarse changed during training")
        loss = F.l1_loss(aux["output"], target)
        finite(loss, "rgb loss")
        loss.backward()
        if step == 0:
            offenders = []
            for name, parameter in model.named_parameters():
                require(parameter.grad is not None, f"Missing gradient for {name}")
                finite(parameter.grad, f"gradient of {name}")
                peak = parameter.grad.detach().abs().max().item()
                if name.startswith("interaction_head.residual."):
                    step0_residual = max(step0_residual, peak)
                else:
                    step0_upstream = max(step0_upstream, peak)
                    if peak != 0.0:
                        offenders.append(f"{name}={peak}")
            require(step0_residual > 0.0, "Step 0 did not update the residual convolutions")
            require(not offenders, "Step 0 leaked upstream gradients: " + ", ".join(offenders[:6]))
        else:
            peaks = {
                "branches": _weight_grad_peak(model, "interaction_head.branches."),
                "mix": _weight_grad_peak(model, "interaction_head.mix."),
                "gate": _weight_grad_peak(model, "interaction_head.gate."),
                "stem": _weight_grad_peak(model, "stem."),
                "residual": _weight_grad_peak(model, "interaction_head.residual."),
            }
            for name, peak in peaks.items():
                later_weight_peak[name] = max(later_weight_peak[name], peak)
            for index in range(3):
                later_branch_peak[index] = max(
                    later_branch_peak[index],
                    _weight_grad_peak(model, f"interaction_head.branches.{index}."),
                )
            if first_upstream_step is None and all(
                peaks[name] > 0.0 for name in ("branches", "mix", "gate", "stem")
            ):
                first_upstream_step = step
        losses.append(loss.item())
        optimizer.step()
    for name, peak in later_weight_peak.items():
        require(peak > 0.0, f"{name} weight gradient stayed zero after residual updates")
    for index, peak in enumerate(later_branch_peak):
        require(peak > 0.0, f"colour branch {index} weight gradient stayed zero")
    require(first_upstream_step == 1, f"upstream gradients did not appear on the second step: {first_upstream_step}")
    result["optimization"] = {
        "optimizer": "Adam",
        "lr": 1e-3,
        "steps": len(losses),
        "losses": losses,
        "step0_residual_grad_max": step0_residual,
        "step0_upstream_grad_max": step0_upstream,
        "later_weight_grad_max": later_weight_peak,
        "later_per_branch_weight_grad_max": later_branch_peak,
        "first_step_index_with_upstream_gradients": first_upstream_step,
        "global_log_gain_error": gain_error,
        "chroma_luminance_error": chroma_error,
    }

    model.eval()
    cloned = json.loads(json.dumps(model.config))
    require(cloned == model.config, "config JSON round trip changed the model config")
    require(list(cloned) == ["width", "spectral_mode", "output_mode"], "config keys changed")
    restored = EnhancementDemo(**cloned).to(device).eval()
    blob = io.BytesIO()
    torch.save(model.state_dict(), blob)
    blob.seek(0)
    restored.load_state_dict(torch.load(blob, map_location=device, weights_only=True), strict=True)
    for key, value in model.state_dict().items():
        require(torch.equal(value, restored.state_dict()[key]), f"state round trip changed {key}")
    with torch.no_grad():
        roundtrip_error = (model(sample) - restored(sample)).abs().max().item()
        probe = torch.rand((1, 3, 1, 1), device=device)
        probe_error = (model(probe) - restored(probe)).abs().max().item()
    require(roundtrip_error < 1e-7, f"rgb checkpoint round trip changed output: {roundtrip_error}")
    require(probe_error < 1e-7, f"rgb checkpoint round trip changed a 1x1 output: {probe_error}")
    result["checkpoint_max_error"] = max(roundtrip_error, probe_error)
    result["config_roundtrip"] = True

    clipping = build(device, "off", "rgb_interaction").eval()
    with torch.no_grad():
        for layer in clipping.interaction_head.residual:
            layer.weight.zero_()
            layer.bias.fill_(2.0)
        above = clipping(torch.full((1, 3, 5, 7), 0.5, device=device))
        above_tiny = clipping(torch.full((1, 3, 1, 1), 0.5, device=device))
        for layer in clipping.interaction_head.residual:
            layer.bias.fill_(-2.0)
        below = clipping(torch.full((1, 3, 5, 7), 0.5, device=device))
        below_tiny = clipping(torch.full((1, 3, 1, 1), 0.5, device=device))
    finite(above, "rgb unclipped high output")
    finite(below, "rgb unclipped low output")
    require(above.min().item() > 1.0 and above_tiny.min().item() > 1.0, "rgb clips output above one")
    require(below.max().item() < 0.0 and below_tiny.max().item() < 0.0, "rgb clips output below zero")
    result["positive_bias_output_min"] = min(above.min().item(), above_tiny.min().item())
    result["negative_bias_output_max"] = max(below.max().item(), below_tiny.max().item())
    result["raw_output_unclipped"] = True

    # Fixed synthetic weights: red is copied into green, then green's gate is closed.
    torch.manual_seed(20260925)
    head = RGBInteractionHead(24).to(device).eval()
    with torch.no_grad():
        for index in (0, 1):
            first, second = head.branches[index][0], head.branches[index][2]
            first.weight.zero_()
            first.bias.zero_()
            first.weight[:, -1, 1, 1] = 1
            second.weight.zero_()
            second.bias.zero_()
            second.weight[:, :, 1, 1] = torch.eye(8, device=device, dtype=second.weight.dtype)
        head.mix.weight.zero_()
        head.mix.bias.zero_()
        head.mix.weight[8:16, 0:8, 0, 0] = torch.eye(8, device=device, dtype=head.mix.weight.dtype)
        head.gate.weight.zero_()
        head.gate.bias.fill_(20.0)
        for layer in head.residual:
            layer.weight.zero_()
            layer.bias.zero_()
        head.residual[1].weight[0, 0, 1, 1] = 1
        gate_probe = head.gate(torch.zeros(1, 24, 2, 2, device=device))
        require(torch.allclose(gate_probe, head.gate.bias.view(1, 3, 1, 1).expand_as(gate_probe)),
                "Zero mix/gate weights do not reproduce the gate bias")
        feature = torch.zeros(2, 24, 5, 7, device=device)
        source = torch.zeros(2, 3, 5, 7, device=device)
        source[:, 1] = 0.4
        changed = source.clone()
        changed[:, 0, :, 4:] = 1
        open_source = head(feature, source)
        open_changed = head(feature, changed)
        green_source = head.branches[1](torch.cat((feature, source[:, 1:2]), dim=1))
        green_changed = head.branches[1](torch.cat((feature, changed[:, 1:2]), dim=1))
        require(torch.equal(green_source, green_changed), "Green branch feature changed while its input was fixed")
        own = F.gelu(F.gelu(torch.tensor(0.4, device=device)))
        cross = F.gelu(F.gelu(torch.tensor(1.0, device=device)))
        open_gate = torch.sigmoid(torch.tensor(20.0, device=device))
        require(torch.allclose(green_source, own.expand_as(green_source), atol=1e-5, rtol=1e-5),
                "Green branch does not match conv-GELU-conv-GELU")
        expected_changed = own.expand(2, 5, 7).clone()
        expected_changed[:, :, 4:] = own + open_gate * cross
        map_error = max(
            (open_source[:, 1] - own.expand(2, 5, 7)).abs().max().item(),
            (open_changed[:, 1] - expected_changed).abs().max().item(),
        )
        require(map_error < 1e-4, f"Open-gate cross-colour map mismatch: {map_error}")
        require(torch.equal(open_changed[:, 0], torch.zeros_like(open_changed[:, 0])),
                "Red destination changed without its residual weights")
        require(torch.equal(open_changed[:, 2], torch.zeros_like(open_changed[:, 2])),
                "Blue destination changed")
        head.gate.bias[1] = -40.0
        closed_source = head(feature, source)
        closed_changed = head(feature, changed)
    open_effect = (open_changed[:, 1] - open_source[:, 1]).abs().max().item()
    closed_effect = (closed_changed[:, 1] - closed_source[:, 1]).abs().max().item()
    own_level = closed_source[:, 1].abs().max().item()
    require(open_effect > 0.2, f"Mixed red branch did not affect green: {open_effect}")
    require((open_changed[:, 1, :, :4] - open_source[:, 1, :, :4]).abs().max().item() < 1e-6,
            "Open gate changed green where red was unchanged")
    require(closed_effect < 1e-6, f"Closing the green gate did not remove the cross effect: {closed_effect}")
    require(own_level > 1e-3, f"Closing the gate also removed green's own residual: {own_level}")
    result["interaction"] = {
        "open_cross_effect": open_effect,
        "closed_cross_effect": closed_effect,
        "open_map_max_error": map_error,
        "closed_own_residual_max": own_level,
        "destination_feature_unchanged": True,
    }

    if device.type == "cuda":
        torch.manual_seed(20260927)
        amp_errors = {}
        for output_mode in ("direct", "rgb_interaction"):
            module = build(device, "off", output_mode).eval()
            errors = []
            for shape in ((1, 3, 1, 1), (1, 3, 17, 29)):
                amp_sample = torch.rand(shape, device=device)
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    amp_output = module(amp_sample)
                finite(amp_output.float(), f"autocast {output_mode}")
                require(amp_output.shape == amp_sample.shape, "autocast changed shape")
                errors.append((amp_output.float() - amp_sample).abs().max().item())
            amp_errors[output_mode] = max(errors)
        require(amp_errors["rgb_interaction"] <= max(1e-4, amp_errors["direct"] + 1e-6),
                f"autocast identity diverged from direct: {amp_errors}")
        torch.manual_seed(20260928)
        amp_model = build(device, "off", "rgb_interaction").train()
        amp_sample = torch.rand((2, 3, 17, 29), device=device)
        amp_target = torch.rand_like(amp_sample)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            amp_loss = F.l1_loss(amp_model(amp_sample).float(), amp_target)
        finite(amp_loss, "autocast loss")
        amp_loss.backward()
        residual_peak = 0.0
        for name, parameter in amp_model.named_parameters():
            if parameter.grad is None:
                continue
            finite(parameter.grad, f"autocast gradient of {name}")
            if name.startswith("interaction_head.residual."):
                residual_peak = max(residual_peak, parameter.grad.abs().max().item())
        require(residual_peak > 0.0, "autocast backward produced no residual gradient")
        result["cuda_autocast"] = {
            "checked": True,
            "identity_max_error": amp_errors,
            "residual_grad_max": residual_peak,
        }
    else:
        result["cuda_autocast"] = {"checked": False, "reason": "cpu"}
    return result


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
    report["rgb_interaction"] = check_rgb_interaction(device)
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
