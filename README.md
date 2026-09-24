# ColorWaveletNet

空间自适应残差 LUT 加颜色引导的 Haar 小波 U 型重建。第一版候选，尚未做真实数据训练。规格见 `GROK_IMPLEMENTATION_SPEC.md`。不要修改 `endo_enhancement_demo`。

```powershell
$cwPython = "M:\Anaconda_envs\envs\retinexformer\python.exe"
$cwRoot = "M:\picture data\cholec80_t\code\cut\endo_color_wavelet"
$cwData = "M:\picture data\cholec80_t\train_test"
```

## CPU 自检

不读取真实图像。

```powershell
& $cwPython -u "$cwRoot\verify.py" --device cpu --threads 2
```

结果写入 `$cwRoot\checks\cpu_checks.json`。

## 可选 CUDA AMP / 整图形状自检

本命令需要 GPU。实现阶段没有执行。

```powershell
& $cwPython -u "$cwRoot\verify.py" --device cuda --full-shape
```

形状为 `B=8, C=3, H=224, W=448` 的一次 AMP 优化步。

## 建议的 30000 步训练

不要在未确认前启动。选权重为验证集 float PSNR。余弦按 `--steps` 拉满，短程 `--stop-after` 不改变周期。

```powershell
& $cwPython -u "$cwRoot\run.py" train `
  --run-dir "$cwRoot\runs\color_wavelet_30k_psnr_seed100" `
  --data-root $cwData `
  --steps 30000 --batch-size 8 `
  --lr 0.0002 --lr-schedule cosine --min-lr 0.000002 `
  --val-every 500 --save-every 1000 --log-every 50 `
  --seed 100 --best-metric psnr --device cuda `
  --lambda-coarse 0.2 --lambda-ab 0.05 --lambda-lut 0.0001
```

架构固定为 `color_wavelet_v1`，没有旧 demo 的 `--output-mode` 开关。

## 从 last.pt 恢复

使用与首次训练相同的标志，并指向已有 run 目录。

```powershell
& $cwPython -u "$cwRoot\run.py" train `
  --run-dir "$cwRoot\runs\color_wavelet_30k_psnr_seed100" `
  --data-root $cwData `
  --steps 30000 --batch-size 8 `
  --lr 0.0002 --lr-schedule cosine --min-lr 0.000002 `
  --val-every 500 --save-every 1000 --log-every 50 `
  --seed 100 --best-metric psnr --device cuda `
  --lambda-coarse 0.2 --lambda-ab 0.05 --lambda-lut 0.0001 `
  --resume "$cwRoot\runs\color_wavelet_30k_psnr_seed100\last.pt"
```

## 推理

单图：

```powershell
& $cwPython -u "$cwRoot\run.py" infer `
  --checkpoint "$cwRoot\runs\color_wavelet_30k_psnr_seed100\best_val.pt" `
  --input "M:\picture data\cholec80_t\train_test\val\lowlight\0001.png" `
  --output-dir "$cwRoot\runs\color_wavelet_30k_psnr_seed100\preview" `
  --device cuda
```

文件夹：

```powershell
& $cwPython -u "$cwRoot\run.py" infer `
  --checkpoint "$cwRoot\runs\color_wavelet_30k_psnr_seed100\best_val.pt" `
  --input "M:\picture data\cholec80_t\train_test\val\lowlight" `
  --output-dir "$cwRoot\runs\color_wavelet_30k_psnr_seed100\val_preview" `
  --device cuda
```

输出为量化 RGB PNG：`np.rint(clamp(Y,0,1)*255)`。无 GT 均值校正。请把 `0001.png` 换成实际存在的文件名。

## 验证集 200 对四指标

不会跟在训练后自动跑。使用已有 `HVI_CIDNet\evaluate_four_metrics.py`。

```powershell
& $cwPython -u "$cwRoot\eval_val.py" `
  --checkpoint "$cwRoot\runs\color_wavelet_30k_psnr_seed100\best_val.pt" `
  --experiment-root "$cwRoot\runs\color_wavelet_30k_psnr_seed100\val_metrics" `
  --data-root $cwData `
  --method-label ColorWaveletNet-30k-best-psnr `
  --device cuda
```
