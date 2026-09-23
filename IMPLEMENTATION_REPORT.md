# Deep-A 实现报告

日期：2026-09-23。本轮只实现模型和 CPU 自检，没有启动真实训练。

## 创建的文件

- `model.py`：`DeepA`、`SpatialFusion`；`LayerNorm2d` / `GatedConvBlock` 复制自 `endo_enhancement_demo/model.py`。
- `run.py`：`train` / `infer`，含恢复、余弦、数据预检、manifest SHA256。
- `eval_val.py`：手动 val 四指标，严格 200 个唯一 sample_id。
- `verify.py`、`requirements.txt`、`README.md`、`checks/cpu_checks.json`。
- 规格 `GROK_DEEP_A_SPEC.md` 未改。

未修改 `endo_enhancement_demo`、`endo_color_wavelet`。

实际可训练参数量：**1,253,715**。架构版本 `deep_a_v1`。

## 结构

- 宽度 24/48/96/192，编码块 `[1,2,2,2]`，解码块 `[2,2,1]`（H/4→H/2→H）。
- 单个 `SpatialFusion` 在 192 通道 H/8；计算形式与 A 的 `spectral_mode=off` 空间路径一致，无 Fourier/router。
- 输出 `Y = X.float() + Conv3x3(D0)`，头零初始化。
- 损失：未截断 RGB L1。默认 30000 步、batch 8、余弦 2e-4→2e-6、seed 100、`--best-metric psnr`。

## 已执行

```text
M:\Anaconda_envs\envs\retinexformer\python.exe -u verify.py --device cpu --threads 2
```

`checks/cpu_checks.json` 全部 `passed: true`：层数/通道/skip 尺寸、形状与有限值、初始恒等与非截断、Spatial Fusion 同权重前向/反向、输出头更新后关键模块梯度、序列化恢复、合成 manifest 报错、三个入口 `--help`。

## 未执行（待验证）

- `verify.py --device cuda --full-shape`
- 真实图像训练、推理、val 四指标
- 显存与速度未测，这里不估计

## 偏离

无方法替换。`return_debug=True` 仅用于自检，默认不写盘。

## 训练与恢复命令（未执行）

```powershell
$deepAPython = "M:\Anaconda_envs\envs\retinexformer\python.exe"
$deepARoot = "M:\picture data\cholec80_t\code\cut\endo_deep_a"
$deepAData = "M:\picture data\cholec80_t\train_test"

& $deepAPython -u "$deepARoot\run.py" train `
  --run-dir "$deepARoot\runs\deep_a_30k_psnr_seed100" `
  --data-root $deepAData `
  --steps 30000 --batch-size 8 `
  --lr 0.0002 --lr-schedule cosine --min-lr 0.000002 `
  --val-every 500 --save-every 1000 --log-every 50 `
  --seed 100 --best-metric psnr --device cuda

& $deepAPython -u "$deepARoot\run.py" train `
  --run-dir "$deepARoot\runs\deep_a_30k_psnr_seed100" `
  --data-root $deepAData `
  --steps 30000 --batch-size 8 `
  --lr 0.0002 --lr-schedule cosine --min-lr 0.000002 `
  --val-every 500 --save-every 1000 --log-every 50 `
  --seed 100 --best-metric psnr --device cuda `
  --resume "$deepARoot\runs\deep_a_30k_psnr_seed100\last.pt"
```
