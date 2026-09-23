# Deep-A

在 A 的 direct / spatial-off 计算方式上增加 H/8、192 通道瓶颈。规格见 `GROK_DEEP_A_SPEC.md`。不要修改 `endo_enhancement_demo` 或 `endo_color_wavelet`。

```powershell
$deepAPython = "M:\Anaconda_envs\envs\retinexformer\python.exe"
$deepARoot = "M:\picture data\cholec80_t\code\cut\endo_deep_a"
$deepAData = "M:\picture data\cholec80_t\train_test"
```

## CPU 自检

不读取真实图像。

```powershell
& $deepAPython -u "$deepARoot\verify.py" --device cpu --threads 2
```

结果写入 `$deepARoot\checks\cpu_checks.json`。

## 可选 CUDA 整图自检

本命令需要 GPU。实现阶段没有执行。输入形状 `[8,3,224,448]`，少量 AMP 步。

```powershell
& $deepAPython -u "$deepARoot\verify.py" --device cuda --full-shape --timed-steps 3
```

## 建议的 30000 步训练

不要在未确认前启动。选权重为验证集 float PSNR。

```powershell
& $deepAPython -u "$deepARoot\run.py" train `
  --run-dir "$deepARoot\runs\deep_a_30k_psnr_seed100" `
  --data-root $deepAData `
  --steps 30000 --batch-size 8 `
  --lr 0.0002 --lr-schedule cosine --min-lr 0.000002 `
  --val-every 500 --save-every 1000 --log-every 50 `
  --seed 100 --best-metric psnr --device cuda
```

## 从 last.pt 恢复

使用与首次训练相同的标志。

```powershell
& $deepAPython -u "$deepARoot\run.py" train `
  --run-dir "$deepARoot\runs\deep_a_30k_psnr_seed100" `
  --data-root $deepAData `
  --steps 30000 --batch-size 8 `
  --lr 0.0002 --lr-schedule cosine --min-lr 0.000002 `
  --val-every 500 --save-every 1000 --log-every 50 `
  --seed 100 --best-metric psnr --device cuda `
  --resume "$deepARoot\runs\deep_a_30k_psnr_seed100\last.pt"
```

## 推理

把输入换成实际存在的文件或文件夹。

```powershell
& $deepAPython -u "$deepARoot\run.py" infer `
  --checkpoint "$deepARoot\runs\deep_a_30k_psnr_seed100\best_val.pt" `
  --input "$deepAData\val\lowlight" `
  --output-dir "$deepARoot\runs\deep_a_30k_psnr_seed100\val_preview" `
  --device cuda
```

单图把 `--input` 换成一张 PNG。输出为 `np.rint(clamp(Y,0,1)*255)` 的 RGB PNG。

## 验证集 200 对四指标

不会跟在训练后自动跑。

```powershell
& $deepAPython -u "$deepARoot\eval_val.py" `
  --checkpoint "$deepARoot\runs\deep_a_30k_psnr_seed100\best_val.pt" `
  --experiment-root "$deepARoot\runs\deep_a_30k_psnr_seed100\val_metrics" `
  --data-root $deepAData `
  --method-label DeepA-30k-best-psnr `
  --device cuda
```
