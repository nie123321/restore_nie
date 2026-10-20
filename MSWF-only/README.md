# MSWF-only

Image enhancement network with **11 joint-subband MSWF blocks**, **688,707 parameters**, and no LFGSI blocks in its forward pass or trained state dictionary. The encoder/decoder widths are 24/48/96. This directory is a self-contained release of the version used for the current local seed100/seed101 runs.

## Installation

Use Python 3.10 or newer. Install a CUDA-compatible PyTorch build for your system, then:

```bash
cd MSWF-only
pip install -r requirements.txt
```

No VGG weights, dataset, or trained enhancement checkpoint is included in this directory. The training recipe still uses the frozen ImageNet VGG19 `features[:16]` perceptual loss. Before training, prepare its weights locally:

```bash
python prepare_vgg.py
```

This downloads the official full VGG19 checkpoint on first use (about 548 MB), then saves only the roughly 6.95 MB prefix as `weights/vgg19_features16.pth`. An existing torchvision cache is reused. Alternatively, copy an existing compatible prefix directly to that path. The weights directory is ignored by Git. VGG is not needed for enhancement inference.

## Dataset

Provide paired images with identical filenames and dimensions:

```text
DATA_ROOT/
  split_manifest.csv
  train/
    lowlight/
    gt/
  test/
    lowlight/
    gt/
```

Training requires `split_manifest.csv`. It records the split provenance; paired evaluation also uses its test entries. The required evaluation columns are:

```csv
sample_id,split,lowlight_relpath,gt_relpath
1,train,train/lowlight/0001.png,train/gt/0001.png
2,test,test/lowlight/0002.png,test/gt/0002.png
```

The reference split contains 1,700 training pairs and 300 test pairs. Supply your actual data root explicitly; `config.json` preserves the original local default path for provenance.

## Training

```bash
python -u train.py --data-root /path/to/DATA_ROOT --output outputs/seed100 --seed 100 --workers 4
python -u train.py --data-root /path/to/DATA_ROOT --output outputs/seed101 --seed 101 --workers 4
```

The commands above are separate runs; use separate output directories. The recipe is:

- 100,000 steps, batch size 8, random 128 x 128 crops.
- AdamW, learning rate 2e-4 to 2e-6 with a full-length cosine schedule, weight decay 1e-4, gradient clipping 1.
- Paired D4 augmentation and paired saturation augmentation (probability 0.25, factor 0.9-1.1).
- `GT-Mean L1(sigma=0.1) + 0.1 * complex FFT L1 + 0.005 * VGG19 relu3_3 MSE`.
- FFT uses `norm='backward'`; VGG features are frozen and use ImageNet mean/std normalization without image resizing.
- FP32; AMP and TF32 disabled. SSIM/HF are not training losses. FFT/VGG forwards are skipped when their weight is zero.
- Checkpointing and full-image RGB uint8 PSNR/SSIM evaluation every 500 steps. The configured evaluation split is `test`, and its scores select `best_psnr.pt` and `best_ssim.pt`.

The preserved config defaults to two workers; `--workers 4` overrides that setting without changing the model or loss. Training workers remain active for the single step-based training iterator. Evaluation workers are recreated at each evaluation because `persistent_workers` is not enabled in this release.

Linux background launcher:

```bash
bash train_server.sh --data-root /path/to/DATA_ROOT --output outputs/seed100 --seed 100 --workers 4
```

On Windows, use the **same environment's `pythonw.exe`** with the background launcher so that DataLoader workers do not open console windows:

```powershell
.\train_local.ps1 -PythonPath (Get-Command pythonw.exe).Source -DataRoot 'D:\dataset' -OutputPath outputs/seed100 -Workers 4
```

For a foreground run, use `python train.py` directly. Background launcher logs are under `launch_logs/`; training outputs include `train.jsonl`, `status.json`, `selection.json`, a code snapshot, and native checkpoints.

## Resume

```bash
python -u train.py --data-root /path/to/DATA_ROOT --output outputs/seed100 --seed 100 --workers 4 --resume outputs/seed100/last.pt
```

Resume using this release's own native checkpoints, unchanged training settings/data, and the matching output directory. Seed101 resumes must explicitly pass `--seed 101`. Resume verifies the source fingerprint, configuration, training image list, and split manifest. Model inference loads native MSWF-only checkpoints strictly; historical R3/full-B adapters are not included.

## Inference and evaluation

```bash
# A single image or a folder of images
python test.py --checkpoint /path/to/MSWF_only.pt --input /path/to/images --output outputs/inference

# Paired four-metric evaluation
python test.py --checkpoint /path/to/MSWF_only.pt --data-root /path/to/DATA_ROOT --output outputs/evaluation
```

Inference uses full-resolution FP32 with no self-ensemble or ground-truth adjustment. Outputs are clamped to [0,1], rounded to RGB uint8, and saved as PNGs. Paired evaluation writes per-image and mean PSNR, SSIM, LPIPS-Alex, and CIEDE2000; LPIPS may download its feature-extractor weights on first use. Each command requires a fresh output directory. For CPU inference, add `--device cpu`.

## Source identity

All training/model source files, config, variant metadata, and launch scripts are byte-identical to the active local MSWF-only package. This README, `prepare_vgg.py`, and repository packaging files are new; VGG weights are excluded. `source_manifest.json` records the exact training-source fingerprint:

```text
3e856379601a341b33820332aeacfbceaa8d020aaff51ed23709a31a04df0219
```

The model constructor preserves the reference B initialization order and then removes the two LFGSI blocks and their scaling parameters before training. It reads no trained reference checkpoint. This preserves same-seed initialization of shared layers; LFGSI is absent from the forward pass, optimizer, and saved model parameters.
