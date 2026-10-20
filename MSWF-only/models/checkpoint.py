"""Current native model config only; no historical checkpoint adapters."""
MODEL_VARIANT = 'final-b-mswf-only-vgg0005-20261019'


def validate_model_config(config):
    expected = {'width': 24, 'experiment_variant': MODEL_VARIANT}
    if not isinstance(config, dict) or type(config.get('width')) is not int or config != expected:
        raise ValueError(f'This package requires its own native model config: {expected}')
