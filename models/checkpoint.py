from collections import OrderedDict

R3_VARIANT = "v2-bottom-interleaved-20261008"


def validate_model_config(config):
    expected = {
        "width": 24, "spectral_mode": "off", "output_mode": "direct",
        "fusion_mode": "gated", "bottleneck_attention": "mdta",
        "half_decoder": "gated", "half_encoder_frequency": "none",
        "bottleneck_prior": "none", "illumination_guidance": "none",
        "star_refinement": "multiscale", "lf_guidance": "none",
        "star_variant": "original", "wavelet_variant": "all-subband-first-mdta",
        "experiment_variant": R3_VARIANT,
    }
    for key, value in config.items():
        if key not in expected or value != expected[key]:
            raise ValueError(f"Checkpoint does not match this architecture: {key}={value}")

ROOT_NAMES = {
    "encoder0": "encoder_full_1", "star_encoder0": "encoder_full_2",
    "encoder1": "encoder_half_1", "star_encoder1": "encoder_half_2",
    "encoder2": "bottleneck_fusion_in", "spectral": "bottleneck_fusion_stages",
    "mdta": "lfgsi1", "mdta_second": "lfgsi2",
    "decoder1": "decoder_half_1", "star_decoder1": "decoder_half_2",
    "decoder0": "decoder_full_1", "star_decoder0": "decoder_full_2",
    "down1": "downsample_half", "down2": "downsample_quarter",
    "fuse1": "decoder_half_merge", "fuse0": "decoder_full_merge",
    "direct_head": "output_head",
}
SPATIAL_NAMES = {
    "norm1": "spatial_norm", "expand": "spatial_in", "depthwise": "spatial_dw",
    "channel_scale": "channel_gate", "project": "spatial_out", "norm2": "ffn_norm",
    "scale1": "fusion_scale", "scale2": "ffn_scale",
}
FUSION_NAMES = {"spatial": "spatial_path", "coarse": "coarse_processor", "fine": "fine_processor"}
INTERACTION_NAMES = {"norm": "low_norm", "attn": "low_attention", "gates": "high_gates"}
ATTENTION_NAMES = {"qkv_dwconv": "qkv_depthwise", "project_out": "output_project"}
OLD_FUSION_ROOTS = {
    "encoder0", "encoder1", "encoder2", "decoder1", "decoder0",
    "star_encoder0", "star_encoder1", "star_decoder1", "star_decoder0", "spectral",
}


def canonical_key(key: str) -> str:
    parts = key.split(".")
    root = parts[0]
    if root not in ROOT_NAMES:
        return key
    parts[0] = ROOT_NAMES[root]
    if root in OLD_FUSION_ROOTS:
        offset = 1
        if root == "spectral":
            if len(parts) < 2 or parts[1] != "block":
                return ".".join(parts)
            parts[1] = "blocks"
            offset = 3
        if len(parts) > offset:
            kind = parts[offset]
            parts[offset] = FUSION_NAMES.get(kind, kind)
            if kind == "spatial" and len(parts) > offset + 1:
                parts[offset + 1] = SPATIAL_NAMES.get(parts[offset + 1], parts[offset + 1])
    elif root in {"mdta", "mdta_second"} and len(parts) > 1:
        kind = parts[1]
        parts[1] = INTERACTION_NAMES.get(kind, kind)
        if kind == "attn" and len(parts) > 2:
            parts[2] = ATTENTION_NAMES.get(parts[2], parts[2])
    return ".".join(parts)


def canonical_state_dict(state_dict):
    result = OrderedDict()
    for name, tensor in state_dict.items():
        converted = canonical_key(name)
        if converted in result:
            raise ValueError(f"Checkpoint contains duplicate aliases for {converted}")
        result[converted] = tensor
    if hasattr(state_dict, "_metadata"):
        result._metadata = {canonical_key(name): value
                            for name, value in state_dict._metadata.items()}
    return result


