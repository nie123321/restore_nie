"""Prepare the frozen VGG19 loss prefix locally; weights are not published."""
from pathlib import Path
import uuid

import torch
from torchvision.models import VGG19_Weights


def main():
    destination = Path(__file__).resolve().parent / 'weights/vgg19_features16.pth'
    if destination.exists():
        raise FileExistsError(f'Weights already exist; leaving them unchanged: {destination}')
    state = VGG19_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
    prefix = {
        name.removeprefix('features.'): value
        for name, value in state.items()
        if name.startswith('features.') and int(name.split('.')[1]) < 16
    }
    from losses import VGGPerceptualMSE

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'.vgg.{uuid.uuid4().hex}.tmp')
    try:
        torch.save(prefix, temporary)
        # Validate the same layer/key contract used by training before publishing locally.
        VGGPerceptualMSE(temporary)
        # Exclusive destination creation prevents overwriting independently prepared weights.
        with temporary.open('rb') as source, destination.open('xb') as output:
            import shutil
            shutil.copyfileobj(source, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Prepared frozen VGG19 features[:16]: {destination}')


if __name__ == '__main__':
    main()
