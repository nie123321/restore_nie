# Deep-A 实现说明：在 A 基础上增加深层恢复能力

日期：2026-09-23。状态：待实现；没有启动训练或测量新模型资源占用。

## 1. 任务与交付范围

请实现本文规定的一版 Deep-A，作为后续寻找更强基础模型的完整候选。用户负责向 Grok 下命令，完成后会请 Codex 核查。

- 新项目目录：`M:\picture data\cholec80_t\code\cut\endo_deep_a`。
- 只读参考项目：`M:\picture data\cholec80_t\code\cut\endo_enhancement_demo`。
- 保留原 A、RGB 分支版本和 `endo_color_wavelet`。不要修改它们的代码、运行目录、权重或启动脚本。
- 此次完成新代码、使用说明、轻量 CPU 合成自检。不要启动正式训练、真实图像推理、四指标评价、GPU 自检或自动监控；这些由用户随后启动。
- 不检查现有实验、日志、checkpoint、GPU 占用或训练进程。实现阶段可以阅读指定的参考源码，不需要读取真实图像。
- 不调用其他 AI，不建立多代理或实验队列。不要自动安装/升级现有 PyTorch 环境。

这版的研究问题是：**沿用 A 的基本计算方式，增加一个深层尺度并增加深层块数，能否获得更好的恢复结果。**

它同时改变了容量、感受野和尺度组织，因此不能称为严格的“只改变参数量”实验，也不能将一次结果直接解释成容量是唯一瓶颈。这版不承担颜色模块创新的任务。

## 2. 已核对的 A 结构与参考文件

A 对应旧模型的 `width=24, output_mode="direct", spectral_mode="off"`。

旧文件：

- `M:\picture data\cholec80_t\code\cut\endo_enhancement_demo\model.py`
  - `LayerNorm2d`、`GatedConvBlock`。
  - `ConditionalSpectralBlock` 的 `mode="off"` 路径。
  - `EnhancementDemo` 的下采样、插值、跳接融合和 direct 输出路径。
- `M:\picture data\cholec80_t\code\cut\endo_enhancement_demo\run_demo.py`
  - 数据读取、`StepBatches`、学习率计算、训练验证、保存恢复和推理格式。
- `M:\picture data\cholec80_t\code\cut\endo_enhancement_demo\run_ablation_queue.py`
  - 仅参考 `evaluate` 中验证集 manifest 与四指标脚本的调用方式，不复制或启动队列。

旧 A 的尺度为 24/48/96，每个编码和解码尺度一个 GatedConvBlock，Spatial Fusion 在 96 通道瓶颈处。`off` 只关闭 Fourier 滤波，**没有关闭 Spatial Fusion**。

新项目可复制必要的小模块和训练代码并注明来源。正式运行不要通过导入旧项目来依赖它的可变源码。不要复制旧 runs、缓存、权重或整套其他版本。

## 3. 固定的 Deep-A 第一版结构

原分辨率的编码/解码层不增加块数，新增计算集中在较低分辨率。

| 位置 | 输入 224×448 时的空间尺寸 | 通道数 | GatedConvBlock 数量 |
| --- | --- | ---: | ---: |
| Stem 后的 Encoder 0 | 224×448 | 24 | 1 |
| Encoder 1 | 112×224 | 48 | 2 |
| Encoder 2 | 56×112 | 96 | 2 |
| 新增 Encoder 3 / 瓶颈 | 28×56 | 192 | 2 |
| Spatial Fusion | 28×56 | 192 | 单个原 A 空间模块 |
| Decoder 2 | 56×112 | 96 | 2 |
| Decoder 1 | 112×224 | 48 | 2 |
| Decoder 0 | 224×448 | 24 | 1 |
| RGB 残差头 | 224×448 | 24→3 | 单个 3×3 卷积 |

### 3.1 准确的数据流

`GB(c,n)` 表示串联 n 个 c 通道的旧 GatedConvBlock。

```text
X: [B,3,H,W]

E0 = GB(24,1)(Conv3x3(3→24)(X))
E1 = GB(48,2)(Down3x3(24→48)(E0))
E2 = GB(96,2)(Down3x3(48→96)(E1))
E3 = GB(192,2)(Down3x3(96→192)(E2))
B  = SpatialFusion(192)(E3)

D2 = GB(96,2)(Conv1x1(288→96)(concat(resize(B, size(E2)), E2)))
D1 = GB(48,2)(Conv1x1(144→48)(concat(resize(D2, size(E1)), E1)))
D0 = GB(24,1)(Conv1x1(72→24)(concat(resize(D1, size(E0)), E0)))

R = Conv3x3(24→3)(D0)
Y = X.float() + R.float()
```

- Stem 和输出头：3×3，stride=1，padding=1。
- 三次 Down：3×3，stride=2，padding=1，与 A 一致。
- resize：bilinear，`align_corners=False`，明确指定对应 skip 的空间尺寸。不要换成转置卷积、PixelShuffle 或小波。
- 奇数尺寸允许在下采样时向上取整，通过对应 skip 的尺寸恢复。无需强行把输入 resize 或 pad 成 8 的倍数。
- 最终返回与输入同形状的 FP32、未截断 RGB。训练中不 clamp 最终输出，不接 sigmoid/tanh。

### 3.2 Spatial Fusion 的保留方式

保留原 A `mode="off"` 的空间计算：

```text
q, v = split(DWConv3x3(Conv1x1(C→2C)(LayerNorm2d(F))))
mixed = LayerNorm2d(q) * (v * sigmoid(DWConv3x3(v)))
out = F + scale * Conv1x1(C→C)(mixed)
```

- `scale` 初值与 A 一致为 0.1；其余计算和初始化沿用 A。
- 新版只有一个 Spatial Fusion，放在新增 192 通道的 H/8 瓶颈；不要同时在 H/4 再保留一个。
- “保留”指保留计算形式：由于位置和通道数改变，它不是原先 96 通道模块的同一组参数。
- 可将这条路径单独整理为 `SpatialFusion` 类，去掉未使用的 Fourier/router/condition 逻辑。整理前后同权重下的前向值与梯度应一致。
- 不额外加入全局注意力、颜色先验、动态卷积、RGB 三分支、LUT、小波、Retinex 增益或任何额外辅助头。

### 3.3 初始化和接口

- 主干沿用 A 的初始化和 GatedConvBlock 残差缩放，不顺便替换激活、归一化或门控公式。
- direct 输出头的权重和 bias **零初始化**，与 A 一致，初始输出为输入图像。
- 因输出头零初始化，第一次反向传播中主干梯度为零是预期现象；关键梯度验收应在输出头发生更新后检查。
- 全部从头训练，不加载 A 权重做局部迁移，不用 `strict=False` 静默忽略参数。
- 建议类名 `DeepA`，接口 `y = model(x)`；额外 debug 特征只在显式选项下返回，不在默认路径保存到磁盘。
- 输入必须为非空 NCHW 三通道张量；支持正常尺寸、奇数尺寸、1×1。
- 架构版本建议 `deep_a_v1`，配置完整保存 `widths=[24,48,96,192]`、`encoder_blocks=[1,2,2,2]`、`decoder_blocks=[2,2,1]`。decoder 列表按执行顺序 H/4→H/2→H 排列。
- 固定 `output_mode="direct"`、`spectral_mode="off"`，不要把旧模型的 structured 或 RGB head 默认值带入。
- 报告实际可训练参数量。不要先写“约 1M”“与 A 一样快”之类未经统计/测量的结论。

## 4. 训练方案与数据协议

以下是为下一次完整训练准备的默认配置，**本轮不执行**。

| 项目 | 设置 |
| --- | --- |
| Python | `M:\Anaconda_envs\envs\retinexformer\python.exe` |
| 数据根目录 | `M:\picture data\cholec80_t\train_test`，只读 |
| 配对路径 | `train/lowlight` ↔ `train/gt`；`val/lowlight` ↔ `val/gt`，按同名文件匹配 |
| 划分 | 既有 `split_manifest.csv`；train 1500 对，val 200 对；不读取 test 图像 |
| 图像 | 整图 encoded RGB `[0,1]`，现有协议宽448×高224，不裁剪、不 resize、不颜色增强、不 GT 均值校正 |
| 损失 | 仅 `mean(abs(Y.float() - GT.float()))`，未截断 RGB L1 |
| 优化器 | AdamW，lr=2e-4，weight_decay=1e-4，betas=(0.9,0.999) |
| Batch | 8，沿用 A 最后不足一个 batch 时保留短批次的行为 |
| 学习率计划 | 余弦 2e-4→2e-6，依据目标总步数与绝对 step 计算 |
| 总步数 | 默认 30000，提供 `--steps` 参数 |
| seed / workers | 100 / 0 |
| AMP / 梯度裁剪 | CUDA AMP；CPU FP32；grad_clip=1.0 |
| 验证 | 每500步以及最后一步，完整 val，batch=1 |
| 保存 / 日志 | 每1000步另存权重；last、best；每50步打印日志 |
| 选权重 | 默认 `--best-metric psnr`，同时支持 `l1`；在记录中明确所用规则 |

30k/max-PSNR 是此次建议配置，不代表已核实某份历史 A 结果采用同样配置。比较时必须使用训练预算、数据划分、选权重规则和评价口径一致的 A 结果。实现阶段不为此扫描旧 runs，也不自动重跑 A。

训练验证沿用旧口径：逐图未截断 L1，以及逐图 clamp 到 `[0,1]` 后的 RGB float PSNR，随后图像等权平均；PSNR 的 MSE 下限为 1e-12。不能把训练期 float PSNR 和最终量化 PNG 的 PSNR 混作同一结果。

真实实验启动时才做数据预检：核对 manifest 中 train/val 数量、sample_id 唯一、配对路径与目录文件集合一致，记录 manifest SHA256 与使用的文件名；manifest 缺失或不一致应报错，不按目录临时凑出新的划分。不选择失败图或为反光/配准边缘设计掩码。

## 5. 运行与恢复契约

沿用旧 runner 的简单结构，提供 `train`、`infer`。不要搭建通用训练框架或实验调度系统。

1. 新实验必须使用独立的空 run 目录。已有输出时拒绝覆盖，恢复需要显式 `--resume`；输出不能落入数据或旧项目目录。
2. checkpoint 保存模型、optimizer、GradScaler、完整配置、step、RNG、best score、训练/验证文件名及 manifest 摘要。
3. 恢复时严格核对架构版本、层数/宽度、数据划分、batch、seed、损失、学习率计划和选权重规则。余弦训练必须同时核对原总步数与 min_lr。
4. 支持 `--stop-after N`，只限制本次追加执行的步数，不改变原 `--steps` 及余弦周期。
5. 不静默重置优化器、scaler 或训练步数，不把 A/ColorWavelet 的 checkpoint 当 Deep-A 恢复。
6. loss 非有限时停止并记录错误；保留梯度范数、AMP scale、`amp_skipped_step` 的记录。自检时不能把 AMP 跳过更新当作有效优化。
7. 所有输出放在新项目指定 run 中；至少有 `config.json`、`loss.jsonl`、`status.json`、`last.pt`、`best_val.pt`、`step_XXXXXX.pt`。
8. 不因 OOM 或异常自动更改 batch、层数、宽度、损失或训练总步数。报告实际问题，由用户决定后续配置。

推理只需要 checkpoint 和输入图像，严格加载权重，保持原分辨率。保存 PNG 使用：

```python
np.rint(y.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
```

## 6. 验证集四指标入口

准备独立手动入口 `eval_val.py`，本轮不要运行，也不要默认跟在训练后执行。

- 四指标继续调用 `M:\picture data\cholec80_t\train_test\HVI_CIDNet\evaluate_four_metrics.py`，不要重写其 PSNR/SSIM/LPIPS/CIEDE2000 数值口径。
- 命令参数可参考旧队列中的 `--experiment-root`、`--method-label`、`--dataset-root`。
- 从既有 split manifest 严格选取 val 的 200 个唯一 sample_id，保存 `sample_id,video,lowlight_relpath,gt_relpath,output_filename`，预测写入本次新评价目录的 `enhanced`。
- 核对路径、文件名集合、预测数量，结果 CSV 必须恰有同一组 200 个唯一 ID，指标全部有限。缺失、重复、非有限就报错，不能只做集合比较而忽略重复行。
- 汇总字段沿用 `output_psnr`、`output_ssim`、`output_lpips_alex`、`output_ciede2000`，图像等权平均；记录 checkpoint SHA256、step、选权重规则、数据 manifest 摘要及评价口径。
- 非空评价目录不得覆盖。不要追加 test 评价或下载新数据/模型；依赖缺失时明确报告。

## 7. 最小实现与验收

建议文件：

```text
endo_deep_a/
  GROK_DEEP_A_SPEC.md       # 本文保留为规格
  model.py                 # DeepA 与复制自 A 的必要模块
  run.py                   # train / infer / resume
  eval_val.py              # 手动 val 四指标入口
  verify.py                # 轻量合成自检
  requirements.txt
  README.md
  IMPLEMENTATION_REPORT.md
  checks/                  # 验收结果
  runs/                    # 用户后续启动的实验
```

现有依赖 torch/numpy/Pillow 足够实现主干，不新增自定义 CUDA 算子或预训练模型。代码只实现这一版 Deep-A，不同时做等参数对照、多个深度变体或颜色模块消融。

本轮运行的 CPU 检查使用小型合成输入，并控制线程数：

- 核对实际层数、各尺度通道、skip/fuse 尺寸、单个 Spatial Fusion 和参数量；正常小图、奇数尺寸、1×1 的形状与有限值正确。
- 确认 direct head 零初始化，初始输出等于输入；在独立测试实例里改变 head 后，确认输出没有被硬截断。
- 整理后的 Spatial Fusion 与原 `off` 路径同权重下前向/反向一致。不要重新跑旧模型的全套历史验证。
- 用不同于输入的合成 GT 做少量（3～5）L1 优化步，确认输出头更新后，新增第三次下采样、192通道块、Spatial Fusion 和各解码级都有有限非零梯度且参数更新。
- 严格加载保存的 checkpoint 后输出一致；微型合成连续训练与保存恢复后的下一步学习率、采样顺序及 CPU 参数结果相符。检查实际序列化，不只复制内存对象。
- 用合成 manifest 检查缺失/重复/数量不符会报错，不能打开真实图像来完成这项自检。
- train、infer、eval_val 的 help 正常。

CPU 结果写入 `checks/cpu_checks.json`，不要打印所有参数的逐项梯度数组；报告各模块是否通过及必要最大误差即可。

另提供用户手动运行的 CUDA 自检命令，实际输入为 `[8,3,224,448]`，执行少量 AMP 优化步。必须验证 loss/梯度有限、AMP 没跳过对应有效步、主干与输出头参数实际更新；报告峰值显存和同步计时。零初始化输出头的首步特性仍按上文处理。**本轮只实现这条命令，不执行。**

如 CPU 检查通过而 CUDA/真实数据尚未执行，在报告里明确写“待验证”，不要写成全部运行通过。

## 8. 完成后交付

`README.md` 提供与实际 argparse 完全一致的 PowerShell 命令：CPU 自检、可选 CUDA 整图自检、建议 30k 训练、同配置恢复、推理、val 四指标。包含空格的路径均正确引用。

命令中建议使用：

```powershell
$deepAPython = "M:\Anaconda_envs\envs\retinexformer\python.exe"
$deepARoot = "M:\picture data\cholec80_t\code\cut\endo_deep_a"
$deepAData = "M:\picture data\cholec80_t\train_test"
# 建议后续 run：$deepARoot\runs\deep_a_30k_psnr_seed100
```

不要覆盖 HOME/CODEX_HOME，不启动后台训练或定时检查。

`IMPLEMENTATION_REPORT.md` 简短说明实际文件、结构与参数量、已执行命令及检查结果、尚未验证事项，以及任何偏离规格的地方。附可直接复制的训练与恢复命令。效果和显存没有实测就不要预测数字。

完成后停止，等待用户命令。此时不需要联系 Codex 或其他代理；用户会另行安排核查。
