# Prior-Fusion-U3

三尺度 U 型主干，宽度 64/128/256。亮度先验和结构先验各有一条独立编码器。瓶颈是两个完整先验融合块和两个空间注意力块交替。没有 FFT。规格见 `GROK_PRIOR_FUSION_U3_SPEC.md`。

实测可训练参数量：**7,145,363**，与规格静态表一致。

2026-09-25 已完成显存执行优化，宽度、块数和参数量不变：

- AMP 残差缩放保持激活的低精度 dtype；主参数保留 FP32。
- 原生 LayerNorm 替换保存多份全尺寸 FP32 特征的展开计算。
- 空间注意力 Q/K/V 最后一维连续，满足融合 SDPA 的布局条件。
- 训练前向默认按块激活重算（`activation_checkpointing=true`），以额外计算换显存；eval/no_grad 自动旁路。

一次有界合成检查：RTX 5070 Ti Laptop 12 GB，224×448，物理 batch 2，FP16，前向 + 反向；峰值 allocated **1,809.6 MiB**，reserved **2,180 MiB**，输出及全部参数梯度有限。没有创建优化器、更新参数或读取真实数据。此结果不等于完整训练步的显存/速度，Adam 两个 FP32 动量状态另约 54.5 MiB，完整更新还可能有临时开销；单次冷启动耗时不用于稳态速度结论。未启动正式训练。

```powershell
$fusionPython = "M:\Anaconda_envs\envs\retinexformer\python.exe"
$fusionRoot = "M:\picture data\cholec80_t\code\cut\endo_prior_fusion_u3"
$fusionData = "M:\picture data\cholec80_t\train_test"
```

## CPU 验收

不读真实图像，不训练。

```powershell
& $fusionPython -u "$fusionRoot\verify.py" --threads 2
```

## 训练

`--steps` 和 `--batch-size` 必须显式给出。下面只是示例，本轮没有执行。

```powershell
& $fusionPython -u -X utf8 "$fusionRoot\run.py" train `
  --data-root $fusionData `
  --run-dir "$fusionRoot\runs\prior_fusion_u3_main" `
  --steps 10000 --batch-size 2 `
  --seed 100 --workers 0 `
  --lr 0.0002 --lr-schedule cosine --min-lr 0.000002 `
  --weight-decay 0.0001 --grad-clip 1 --amp --device cuda `
  --val-every 500 --best-metric l1 --save-every 1000 --log-every 50 `
  --luma-input-mode real --structure-input-mode real
```

恢复时使用同一套参数，并加上 `--resume` 指向该目录的 `last.pt`。

## 推理与验证集评价

```powershell
& $fusionPython -u -X utf8 "$fusionRoot\run.py" infer `
  --checkpoint "$fusionRoot\runs\prior_fusion_u3_main\best_val.pt" `
  --input "$fusionData\val\lowlight" `
  --output-dir "$fusionRoot\runs\prior_fusion_u3_main\val_preview" `
  --device cuda

& $fusionPython -u -X utf8 "$fusionRoot\eval_val.py" `
  --checkpoint "$fusionRoot\runs\prior_fusion_u3_main\best_val.pt" `
  --experiment-root "$fusionRoot\runs\prior_fusion_u3_main\val_metrics" `
  --data-root $fusionData `
  --method-label PriorFusionU3 `
  --device cuda
```

以上 batch 2 示例参考合成前向/反向检查，不代表已执行训练；其他 batch、完整参数更新、稳态速度和真实指标尚未验证。后续对照必须统一有效 batch，不能把缩小物理 batch 后的结果直接当作同训练设置。
