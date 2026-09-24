# Prior-A

在冻结 A（`direct`、`spectral off`、`gated` Spatial Fusion）上增加六通道亮度先验和固定 CIConv-W 结构先验。规格见 `GROK_PRIOR_A_SPEC.md`。不要修改 `endo_enhancement_demo` 或其他候选项目。

```powershell
$priorAPython = "M:\Anaconda_envs\envs\retinexformer\python.exe"
$priorARoot = "M:\picture data\cholec80_t\code\cut\endo_prior_a"
$priorAData = "M:\picture data\cholec80_t\train_test"
```

## CPU 自检

不读取真实图像，不加载历史训练权重。

```powershell
& $priorAPython -u "$priorARoot\verify.py" --threads 2
```

结果写入 `$priorARoot\checks\cpu_checks.json`。

## 建议的 10000 步训练

本轮不执行。选权重是最低验证 L1。

```powershell
& $priorAPython -u -X utf8 "$priorARoot\run.py" train `
  --data-root $priorAData `
  --run-dir "$priorARoot\runs\prior_a_10k_seed100" `
  --steps 10000 --batch-size 8 --seed 100 --workers 0 `
  --lr 0.0002 --lr-schedule cosine --min-lr 0.000002 `
  --weight-decay 0.0001 --grad-clip 1 --amp --device cuda `
  --val-every 500 --best-metric l1 --save-every 1000 --log-every 50
```

## 从 last.pt 恢复

使用与首次训练相同的标志。

```powershell
& $priorAPython -u -X utf8 "$priorARoot\run.py" train `
  --data-root $priorAData `
  --run-dir "$priorARoot\runs\prior_a_10k_seed100" `
  --steps 10000 --batch-size 8 --seed 100 --workers 0 `
  --lr 0.0002 --lr-schedule cosine --min-lr 0.000002 `
  --weight-decay 0.0001 --grad-clip 1 --amp --device cuda `
  --val-every 500 --best-metric l1 --save-every 1000 --log-every 50 `
  --resume "$priorARoot\runs\prior_a_10k_seed100\last.pt"
```

## 推理

把 `--input` 换成实际存在的文件或文件夹。

```powershell
& $priorAPython -u -X utf8 "$priorARoot\run.py" infer `
  --checkpoint "$priorARoot\runs\prior_a_10k_seed100\best_val.pt" `
  --input "$priorAData\val\lowlight" `
  --output-dir "$priorARoot\runs\prior_a_10k_seed100\val_preview" `
  --device cuda
```

## 验证集 200 对四指标

不会跟在训练后自动跑。`--device cpu` 会把子进程的 `CUDA_VISIBLE_DEVICES` 置空。

```powershell
& $priorAPython -u -X utf8 "$priorARoot\eval_val.py" `
  --checkpoint "$priorARoot\runs\prior_a_10k_seed100\best_val.pt" `
  --experiment-root "$priorARoot\runs\prior_a_10k_seed100\val_metrics" `
  --data-root $priorAData `
  --method-label PriorA-10k-best-l1 `
  --device cuda
```
