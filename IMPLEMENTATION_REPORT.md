# Star-A 实现报告

日期：2026-09-23。本轮只实现模型和 CPU 自检，没有启动真实训练。

## 创建的文件

- `model.py`：`StarA`。`LayerNorm2d` / `GatedConvBlock` 复制自 `endo_enhancement_demo/model.py`。
- `star_blocks.py`：完整 Star Block（FMB、SMB、乘法融合、CMU、DFFN）。
- `run.py`、`eval_val.py`、`verify.py`、`requirements.txt`、`README.md`、`checks/cpu_checks.json`。
- 规格 `GROK_STAR_A_SPEC.md` 未改。未修改旧项目和 `endo_color_wavelet`、`endo_deep_a`。

可训练参数量：**263,331**。架构版本 `star_a_v1`。

## 替换位置

- 瓶颈：`Down3x3(48→96)` 后一个 `StarBlock(96)`，替代旧 A 的 96 通道 GatedConvBlock 和独立 Spatial Fusion。
- Decoder 1：跳接 1×1 融合后一个 `StarBlock(48)`，替代旧 `decoder1`。
- Encoder 0、Encoder 1、Decoder 0 仍是 A 的 GatedConvBlock，共三个。
- 输出 `Y = X.float() + Conv3x3(D0)`，头零初始化。宽度 24/48/96，只下采样两次。

两个 Star Block 各自有独立的 `fft_FSAS` 和 DFFN `fft`，共四个频域参数，全 1 初始化。CMU 为 `AdaptiveAvgPool2d(1)` 加无激活 1×1，`basesize=None`。

## 来源

官方文件：`https://github.com/c-yn/StarIR/blob/main/basicsr/models/archs/StarIR_arch.py`  
获取日期：2026-09-23。SHA256：`9ee5e3644df5ebb65b1ddf9b488d2a0832c00c879ffcac75e9144d519ad5d6e0`。  
对照 `LayerNorm(WithBias)`、`SpatialOperation`、`StarModule`、`DFFN`、`StarBlock`，`ffn_expansion_factor=3`。未复制整个仓库，也不是 StarIR 整网。

在 16×24、通道 16 的特征上，同一权重、非全 1 的频域参数，前向最大误差和输入梯度最大误差都低于 `1e-5`。

## 已执行

```text
M:\Anaconda_envs\envs\retinexformer\python.exe -u verify.py --device cpu --threads 2
```

`checks/cpu_checks.json` 全部通过：结构与尺寸、奇数/1×1、初始恒等与非截断、官方模块对照、头更新后的频域/CMU/主干梯度、保存恢复、合成 manifest 错误、三个入口 `--help`。

## 未执行

- `verify.py --device cuda --full-shape`
- 真实图像训练、推理、val 四指标
- 不估计显存、速度或指标

## 偏离

无方法替换。非 8 的倍数只在 `window_fft_filter` 内做右侧和底部 replicate padding，IFFT 后裁回；可被 8 整除时与官方 rearrange 路径一致。
