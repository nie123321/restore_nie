# A + U3 feature priors

独立版本 `prior_a_u3_v1`。在本轮 A 系列三尺度骨架上恢复 U3 的特征级先验接法，从头训练。

- 主干 `32/64/128`；两路先验编码器 `16/32/64`，每尺度 `1/2/2` 个残差块。
- 亮度输入恢复原 U3 六种描述；结构输入恢复原 U3 单张 `CIConv-W(scale=0.9)`，包括原来的归一化与边界处理。
- H/4 瓶颈两个 `LocalPriorBlock`，H/2 解码一个；计算顺序为亮度幅度调制、局部 Spatial Fusion、结构通道交叉注意力、FFN。Query 来自主特征，Key/Value 来自结构特征。
- H/2、H 的两条 skip 恢复 `PriorSkipFusion`，由主干和两种先验共同调节跳连特征。
- 保留 A 的卷积编码与解码结构。输出 `I + R`，最终残差头零初始化。没有 G/B 图像预校正、动态邻域聚合、梯度结构图、FFT 或全局空间注意力。
- 损失仅为最终输出的 raw RGB L1。该试验评估整套 U3 接法迁移到较小骨架后的表现，不是仅移除 G/B 的单因素消融。
- 亮度编码器与 gamma 调制头采用 FP32，主干继续 AMP。小幅初始化的 gamma 头在原先全 AMP 检查中使深层亮度梯度下溢；这个精度调整保留模块计算形式，CPU 同权重输出与原 U3 一致。

训练设置：batch 8、成对 `192×384` 随机裁剪、水平/垂直翻转各 0.5、AMP、seed 100、AdamW `weight_decay=1e-4`、cosine `2e-4→2e-6`。40 epoch，共 7,520 步，每 1,000 步验证和保存。增强采样规则与上一轮完全相同。best 按完整图像验证的最低 raw L1 选择，同时记录 clamped RGB float PSNR。

在 `M:\picture data\cholec80_t\code\cut` 执行：

```powershell
$py = 'M:\Anaconda_envs\envs\retinexformer\python.exe'
& $py -m endo_prior_a_u3.run verify --gpu-smoke --gpu-steps 3
& $py -m endo_prior_a_u3.run train --run-dir endo_prior_a_u3/runs/prior_a_u3_b8_e40_seed100_20260926
& $py -m endo_prior_a_u3.run resume --checkpoint endo_prior_a_u3/runs/prior_a_u3_b8_e40_seed100_20260926/last.pt
```

模型及 checkpoint 版本独立于原 A、U3 和 Prior-A v2。暂停机制沿用 run 目录中的 `STOP_REQUESTED` 文件。CPU 检查包含与旧 U3 模块加载相同权重后的输出一致性；GPU 检查限真实 batch 8 的少量更新。结果位于 `checks/`。
