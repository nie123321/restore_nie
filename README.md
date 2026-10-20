# restore_nie

## MSWF-only

The current **11-block MSWF-only model (688,707 parameters)** is available in [MSWF-only/](MSWF-only/), including training, inference, four-metric evaluation, and instructions for preparing the perceptual-loss weights locally. See its [usage instructions](MSWF-only/README.md).

## Original R3

The existing root-level implementation and `ours.pt` are the earlier R3 release.

```bash
# Install
pip install -r requirements.txt

# Train
python train.py --data-root xx --output xx

# Resume
python train.py --data-root xx --output xx --resume xx

# Inference (image or folder)
python test.py --checkpoint xx --input xx --output xx

# Evaluate (paired dataset)
python test.py --checkpoint xx --data-root xx --output xx
```
