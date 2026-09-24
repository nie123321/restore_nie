# ColorWaveletNet 实现报告

日期：2026-09-23。本轮只实现模型和 CPU 自检，没有启动真实训练。

## 创建的文件

- `model.py`：`ColorWaveletNet`，上下文编码器，GuidedLow/High，Haar U 型重建。`LayerNorm2d` 与 `GatedConvBlock` 从旧 `endo_enhancement_demo/model.py` 复制，不导入旧模块。
- `lut.py`：残差 3D LUT 初始化、三线性采样与权重混合。
- `wavelet.py`：固定 Haar DWT/IWT，高频通道顺序 `[LH, HL, HH]`。
- `losses.py`：`L_rgb`、`L_coarse`、`L_ab`、`L_lut` 与可微 sRGB→CIE Lab D65。
- `run.py`：`train` / `infer`，含恢复、余弦、AMP 跳步记录、非有限值停止。
- `eval_val.py`：手动 val 四指标入口，调用已有 `evaluate_four_metrics.py`。
- `verify.py`：CPU 合成自检；可选 `--device cuda --full-shape`。
- `requirements.txt`、`README.md`、`checks/cpu_checks.json`。
- 规格原文 `GROK_IMPLEMENTATION_SPEC.md` 未改。

实际可训练参数量以 `verify.py` 最新一次为准。架构版本 `color_wavelet_v1`。默认已改为瓶颈 **2** 块、高频 **1** 层，LUT 仍为 K=4。

## 与规格的对应

- 第一阶段：area 下采样到 H/4，C0/C1/C2 宽度 32/64/128，K=4 空间 softmax 权重，全分辨率查 17³ 残差 LUT，FP32 混合，Z 不硬截断。
- 第二阶段：两级 Haar，宽度 32/64/128，编码 2/2、瓶颈 6、解码 2/2，高频每级 2 个 GatedConvBlock；Y = Z + R。
- 损失：`L_total = L_rgb + 0.2 L_coarse + 0.05 L_ab + 1e-4 L_lut`，可用命令行覆盖。
- 训练契约与旧 runner 对齐：空目录、显式 `--resume`、核对架构/损失权重/文件名/步数/选权重规则；默认 30000 步、batch 8、余弦 2e-4→2e-6、seed 100、`--best-metric psnr`。
- 未叠加 FFT、Mamba、扩散、独立 RGB 分支或旧 A 旁路。

## 已执行

```text
M:\Anaconda_envs\envs\retinexformer\python.exe -u verify.py --device cpu --threads 2
```

`checks/cpu_checks.json` 中全部检查 `passed: true`：尺寸/有限值、DWT/IWT、LUT 轴顺序（RGB→GBR 残差表）、空间权重单纯形与公式、关键梯度与参数更新、仅从 Y 的 L1 反传到 LUT/上下文、引导扰动改变高频、Lab 参考值、序列化与恢复、train/infer/eval_val 的 `--help`。

## 已执行的短测与训练

- `verify.py --device cuda --full-shape`（瓶颈 2、高频 1、batch 8）通过，峰值 **11249 MiB**。参数量 **1,838,455**。
- 正式训练用物理 batch **4**，目录 `runs/color_wavelet_30k_psnr_bs4_seed100`。2026-09-24 停在第 **27200 / 30000** 步。`best_val.pt` 为第 **4500** 步，验证 float PSNR **25.307**。
- 300 对测试（与 A/D 同一 manifest、量化 RGB PNG）：PSNR **25.431**、SSIM **0.865**、LPIPS **0.208**、CIEDE **5.335**。浅层 A 为 25.425 / 0.865 / 0.201 / 5.093。PSNR 几乎相同，LPIPS 和 CIEDE 更差。
- 权重、输出图和 `RESULT.md` 在 `runs/`，不进入 Git。

## Codex 复查后的修改（未重做结构）

1. Lab：`t^{1/3}` 只在 `t ≥ (6/29)^3` 上求值，黑色不再出现 NaN 梯度；CPU 增加 `black_lab_grad`。
2. 损失在 `color_wavelet_losses` 内关闭 autocast 并强制 FP32；`run.py` 在 autocast 外计算损失。
3. `verify.py --full-shape` 必须确认梯度有限、AMP 未跳步、参数有更新。瘦身默认（瓶颈 2、高频 1）后已跑通，峰值 11249 MiB。
4. `eval_val.py` 只读 `split_manifest.csv` 的 val 行，必须 200 对且 sample_id / 文件名唯一；不再按目录凑名单。

## 偏离

无方法替换。实现细节：

- 低分辨率上下文用 `scale_factor=0.25` 的 area 下采样，输入已 pad 成 4 的倍数，等价于 H/4×W/4。
- `return_debug=True` 额外返回各尺度 LL/HH 与条件，仅供自检，默认不写盘。
- README 里的单图路径需换成实际 val 文件名；实现阶段没有扫描数据目录。

## 训练与恢复命令（未执行）

```powershell
$cwPython = "M:\Anaconda_envs\envs\retinexformer\python.exe"
$cwRoot = "M:\picture data\cholec80_t\code\cut\endo_color_wavelet"
$cwData = "M:\picture data\cholec80_t\train_test"

& $cwPython -u "$cwRoot\run.py" train `
  --run-dir "$cwRoot\runs\color_wavelet_30k_psnr_seed100" `
  --data-root $cwData `
  --steps 30000 --batch-size 8 `
  --lr 0.0002 --lr-schedule cosine --min-lr 0.000002 `
  --val-every 500 --save-every 1000 --log-every 50 `
  --seed 100 --best-metric psnr --device cuda `
  --lambda-coarse 0.2 --lambda-ab 0.05 --lambda-lut 0.0001

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

本轮没有启动正式训练，没有修改 `endo_enhancement_demo`，没有读取实验结果或 GPU/进程状态。
