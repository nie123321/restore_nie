# Star-A

在 A 的三尺度 U 型框架里，瓶颈和半分辨率解码各放一个完整 Star Block。规格见 `GROK_STAR_A_SPEC.md`。不要修改 `endo_enhancement_demo`、`endo_deep_a` 或 `endo_color_wavelet`。

```powershell
$starAPython = "M:\Anaconda_envs\envs\retinexformer\python.exe"
$starARoot = "M:\picture data\cholec80_t\code\cut\endo_star_a"
$starAData = "M:\picture data\cholec80_t\train_test"
```

## CPU 自检

不读取真实图像。

```powershell
& $starAPython -u "$starARoot\verify.py" --device cpu --threads 2
```

结果写入 `$starARoot\checks\cpu_checks.json`。

## 可选 CUDA 整图自检

本命令需要 GPU。实现阶段没有执行。输入 `[8,3,224,448]`，首步后再跑至少 3 个 AMP 优化步。

```powershell
& $starAPython -u "$starARoot\verify.py" --device cuda --full-shape --timed-steps 3
```

## 建议的 10000 步训练

不要在未确认前启动。选权重默认是最低验证 L1。

```powershell
& $starAPython -u "$starARoot\run.py" train `
  --run-dir "$starARoot\runs\star_a_10k_l1_seed100" `
  --data-root $starAData `
  --steps 10000 --batch-size 8 `
  --lr 0.0002 --lr-schedule cosine --min-lr 0.000002 `
  --weight-decay 0.0001 --grad-clip 1.0 `
  --val-every 500 --save-every 1000 --log-every 50 `
  --seed 100 --workers 0 --best-metric l1 --device cuda
```

## 从 last.pt 恢复

使用与首次训练相同的标志。

```powershell
& $starAPython -u "$starARoot\run.py" train `
  --run-dir "$starARoot\runs\star_a_10k_l1_seed100" `
  --data-root $starAData `
  --steps 10000 --batch-size 8 `
  --lr 0.0002 --lr-schedule cosine --min-lr 0.000002 `
  --weight-decay 0.0001 --grad-clip 1.0 `
  --val-every 500 --save-every 1000 --log-every 50 `
  --seed 100 --workers 0 --best-metric l1 --device cuda `
  --resume "$starARoot\runs\star_a_10k_l1_seed100\last.pt"
```

## 推理

把 `--input` 换成实际存在的文件或文件夹。

```powershell
& $starAPython -u "$starARoot\run.py" infer `
  --checkpoint "$starARoot\runs\star_a_10k_l1_seed100\best_val.pt" `
  --input "$starAData\val\lowlight" `
  --output-dir "$starARoot\runs\star_a_10k_l1_seed100\val_preview" `
  --device cuda
```

## 验证集 200 对四指标

不会跟在训练后自动跑。`--device cpu` 会把子进程的 `CUDA_VISIBLE_DEVICES` 置空。

```powershell
& $starAPython -u "$starARoot\eval_val.py" `
  --checkpoint "$starARoot\runs\star_a_10k_l1_seed100\best_val.pt" `
  --experiment-root "$starARoot\runs\star_a_10k_l1_seed100\val_metrics" `
  --data-root $starAData `
  --method-label StarA-10k-best-l1 `
  --device cuda
```
