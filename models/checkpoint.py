"""Current native model config only; no historical checkpoint adapters."""
MODEL_VARIANT = 'v3-mswf-joint-subband-residual-20261019'


def validate_model_config(config):
    expected = {'width': 24, 'experiment_variant': MODEL_VARIANT}
    if not isinstance(config, dict) or type(config.get('width')) is not int or config != expected:
        raise ValueError(f'This package requires its own native model config: {expected}')
