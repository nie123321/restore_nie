# Star-A 实施说明：在 A 的框架内使用完整 Star Block

状态：实施规格，尚未实现或启动训练。用户将本文交给 Grok 编写代码，完成后再由 Codex 核查。

## 1. 本次任务和研究问题

实现一个独立候选模型 **Star-A**：保留原 A 的三个尺度、下采样、跳接融合和直接残差输出，在瓶颈与半分辨率解码层使用完整 StarIR Star Block。

完整块必须包含 **频域调制 FMB、空间调制 SMB、乘法融合、通道调制 CMU，以及双域前馈网络 DFFN**。仅给旧 Spatial Fusion 打开 FFT、增加一个通道注意力或者把卷积堆得更深，都不符合本规格。

这次要回答的问题是：在 A 的小型 U 型框架内，改变特征处理方式，能否改善恢复效果。它同时改变模块、使用位置和参数量，属于新候选模型探索；结果不能归因于某一个组件，也不能直接称为原创方法或完整 StarIR 复现。

### 1.1 目录与执行边界

- 新项目：`M:\picture data\cholec80_t\code\cut\endo_star_a`，本文已经放在这里。
- 原 A：`M:\picture data\cholec80_t\code\cut\endo_enhancement_demo`，只读参考。
- Deep-A：`M:\picture data\cholec80_t\code\cut\endo_deep_a`，只读参考训练与评价入口。
- `endo_color_wavelet`、旧模型、运行目录、权重和数据均保持原样。
- 本轮交付代码、README、实现报告及少量 CPU 合成检查。正式训练、GPU 检查、真实图像推理和四指标评价，由用户随后手动启动。
- 不扫描现有实验、训练日志、checkpoint、GPU 占用或进程；不读取真实图像来完成实现自检。
- 不建立自动训练队列、监控或其他 AI 调用；不自动安装、升级现有环境。

## 2. 来源与当前 A 的准确含义

原 A 是 `EnhancementDemo(width=24, output_mode="direct", spectral_mode="off")`。

参考源码：

1. `endo_enhancement_demo/model.py`：复制必要的 `LayerNorm2d`、`GatedConvBlock`，参考 A 的卷积、跳接与 direct 输出。
2. `endo_enhancement_demo/run_demo.py`：参考原 A 的采样、训练、保存恢复及 PNG 格式。
3. `endo_deep_a/run.py`、`eval_val.py`：可复制并调整独立运行入口，保留已加入的配对路径与 manifest 哈希检查；不要照搬 Deep-A 架构、30k 默认值或架构标识。
4. StarIR 官方实现：<https://github.com/c-yn/StarIR/blob/main/basicsr/models/archs/StarIR_arch.py>。
   原始文件：<https://raw.githubusercontent.com/c-yn/StarIR/main/basicsr/models/archs/StarIR_arch.py>。
   对照 `LayerNorm`、`SpatialOperation`、`StarModule`、`DFFN`、`StarBlock`，选择 `basesize=None`。
5. 论文方法第 III 节、图 4：`C:\Users\聂骏鹏\Zotero\storage\YBD4SAGG\Cui 等 - 2026 - StarIR Convolutional image restoration with spatial-frequency fusion.pdf`。

实现报告记录实际参考的官方源码版本（commit 或获取日期与源码 SHA256），保留所复制源码的许可与署名。若拿不到官方源码，明确报告，不能把自行写的近似版本标成已经与官方核对。

A 的 `off` 只关闭频域滤波，仍保留空间门控与乘法融合。旧 A/B/C/D 中，C、D 使用结构化输出头；它们没有覆盖“直接输出头＋完整 Star Block”的组合。A 的普通 GatedConvBlock 本身已有简单通道缩放，因此不能宣称 A 完全没有全局统计或通道交互。

正式模型和运行入口应在新目录自包含，不在运行时导入会继续变化的旧项目，不复制旧 runs、权重或整个 StarIR 仓库。

## 3. 第一版结构固定如下

宽度保持 **24/48/96**，只做两次下采样。全模型恰好 **两个完整 Star Block**：96 通道瓶颈一个、48 通道解码层一个。

| 位置 | 224×448 输入对应尺寸 | 处理方式 |
| --- | --- | --- |
| Stem / Encoder 0 | 224×448，24 通道 | 3×3 卷积＋原 A 的一个 GatedConvBlock |
| Encoder 1 | 112×224，48 通道 | 原 A 下采样＋一个 GatedConvBlock |
| 瓶颈 | 56×112，96 通道 | 原 A 下采样＋一个完整 Star Block |
| Decoder 1 | 112×224，48 通道 | 插值、拼接、1×1 融合＋一个完整 Star Block |
| Decoder 0 | 224×448，24 通道 | 插值、拼接、1×1 融合＋原 A 的一个 GatedConvBlock |
| 输出 | 224×448，3 通道 | 3×3 残差头＋输入图像 |

### 3.1 替换位置必须明确

- 瓶颈的一个完整 Star Block **整体替代旧 A 的 `encoder2` 和后接的 `spectral`**。不保留旧的 96 通道 GatedConvBlock 或独立 Spatial Fusion 再叠加。
- Decoder 1 的一个完整 Star Block 替代旧 `decoder1`。
- Encoder 0、Encoder 1、Decoder 0 仍使用原 A 的 GatedConvBlock，共三个。
- 不增加 H/8、192 通道、额外精修阶段或其他模块。

```text
X: [N,3,H,W]

E0 = GatedConvBlock(24)(Conv3x3(3 -> 24)(X))
E1 = GatedConvBlock(48)(Down3x3(24 -> 48)(E0))
B  = StarBlock(96)(Down3x3(48 -> 96)(E1))

D1 = StarBlock(48)(Conv1x1(144 -> 48)(concat(resize(B, E1.size), E1)))
D0 = GatedConvBlock(24)(Conv1x1(72 -> 24)(concat(resize(D1, E0.size), E0)))

R = Conv3x3(24 -> 3)(D0)
Y = X.float() + R.float()
```

Stem/head 均为 stride=1、padding=1；Down 为 3×3、stride=2、padding=1，与 A 一致。插值使用 bilinear、`align_corners=False`，尺寸明确取自对应 skip。不改成原 StarIR 整网的插值下采样方式。

建议类名 `StarA`，`forward(x)` 返回同尺寸、未截断的 FP32 RGB。输出头权重与 bias 零初始化，初始输出等于输入。保留的 GatedConvBlock 计算与 0.1 残差缩放沿用 A。

架构标识固定 `star_a_v1`，配置至少保存 `widths=[24,48,96]`、`star_locations=["bottleneck","decoder1"]`、两个 Star Block 的配置和 `output_mode="direct"`。全部从头训练，不加载 A、Deep-A 或官方预训练权重，不用 `strict=False` 混用 checkpoint。

## 4. 完整 Star Block 的实现细节

### 4.1 固定参数

| 项目 | 设置 |
| --- | --- |
| Star Block 数量 | 两个，各自独立参数 |
| 窗口大小 | 8×8 |
| DFFN expansion | 3.0，隐藏通道数为 `int(3.0*C)` |
| Star 中的 LayerNorm | 官方 WithBias 形式，逐像素沿通道归一化，`eps=1e-5`，有 weight/bias |
| Star 的卷积 bias 参数 | `bias=False`；SMB 内部 depthwise 卷积按官方 `SpatialOperation` 保留默认 `bias=True` |
| CMU 池化 | `AdaptiveAvgPool2d(1)`，即 `basesize=None` |
| 频域滤波器 | 实数可学习参数，全 1 初始化；每个模块独立，窗口间和样本间共享 |
| 滤波器约束 | 沿用官方自由参数；不加正值约束、tanh、0.5～1.5 限幅或条件路由 |
| FFT | 空间窗口的 `rfft2/irfft2`，采用官方默认 normalization（backward） |

保留的 A GatedConvBlock 使用其原有 `eps=1e-6`。两种归一化要分清，不能为统一类名悄悄改变其中一种。

### 4.2 Star Module：顺序不能省略

以下 `U` 为外层 LayerNorm 后的输入，`C` 为所在层的通道数：

```text
q, v = split(DWConv3x3(Conv1x1(C -> 2C)(U)))

# FMB：频域分支
qf = WindowFFTFilter(q, A_star)
qf = LayerNormWithBias(qf)

# SMB：空间分支
vs = v * sigmoid(DWConv3x3(v))

# Star 乘法融合
fused = qf * vs

# CMU：在融合结果上计算通道缩放
channel_scale = Conv1x1(C -> C)(GlobalAveragePool(fused))
mixed = fused * channel_scale
StarModule(U) = Conv1x1(C -> C)(mixed)
```

CMU 的 1×1 卷积后不加 sigmoid、softmax 或其他激活；它调制学习到的特征通道，不能描述成直接对 RGB 做独立校色。`A_star` 对应官方 `fft_FSAS`，形状可沿用 `[C,1,1,8,5]`。

### 4.3 DFFN：保留它自己的频域调制

```text
a, b = split(DWConv3x3(Conv1x1(C -> 2*hidden)(V)))
gated = GELU(a) * b
projected = Conv1x1(hidden -> C)(gated)
DFFN(V) = WindowFFTFilter(projected, A_ffn)
```

- `A_ffn` 对应官方 DFFN 的 `fft`，与 `A_star` 完全独立，同样全 1 初始化。
- 投影到 C 通道发生在 FFT 之前。
- GELU 只作用于其中一支，不能改成 A 的无激活乘法门控。
- 一个完整 Star Block 因而包含两次窗口频域调制，不是仅 Star Module 做 FFT。

### 4.4 外层残差

```text
T = F + StarModule(LN1(F))
O = T + DFFN(LN2(T))
```

沿用官方 Star Block 的两次直接残差相加，不给新 Star Block 加原 A 的 0.1 缩放、额外 beta/gamma 或零初始化门控。LN1、LN2、FMB 内的 LN 独立。两处 Star Block 也不共享权重。

### 4.5 窗口、尺寸与数值精度

1. 只沿 H/W 切不重叠 8×8 窗口。FFT 在每个窗口最后两维执行，不沿通道做 FFT。
2. 对 `[N,C,H,W]` 进行正确的 reshape/permute，得到 `[N,C,nH,nW,8,8]`；滤波参数在 nH/nW 上广播。逆变换后恢复原排列。
3. `irfft2` 明确指定 `s=(8,8)`。FFT、复数乘法和 IFFT 在关闭 autocast 的 FP32 区域执行，保持梯度。不要为提速 `.detach()` 或切换 half FFT。
4. 正常整图训练中两处特征尺寸均是 8 的倍数，运算应与官方模块一致。
5. 对奇数尺寸、小图和 1×1 输入，只在各个 `WindowFFTFilter` 内部、FFT 之前给特征右侧和底部做最少量 replicate padding；IFFT 后立即裁回进入该滤波函数时的 H/W。FMB 的 LayerNorm、CMU 的全局平均在裁回后的特征上计算，避免把填充区域计入统计。不要把整张输入图 resize。
6. 输出不 clamp、不接 sigmoid；只在保存图像和计算规定的验证指标时 clamp。

## 5. 训练与评价协议

第一轮默认回到原 A 的 **10000 步、batch 8、2e-4 余弦衰减** 配方。这里有意采用旧 A 的 10k 配置，不沿用 Deep-A 的 30k 默认值，也不照搬 StarIR 论文的训练预算或频域损失。

| 项目 | 默认值 |
| --- | --- |
| Python | `M:\Anaconda_envs\envs\retinexformer\python.exe` |
| 数据根目录 | `M:\picture data\cholec80_t\train_test`，只读 |
| 划分 | 既有 `split_manifest.csv`，train 1500 对、val 200 对 |
| 图像 | 宽448×高224整图，encoded RGB `[0,1]`，不裁剪、resize、增强或 GT 均值校正 |
| 损失 | 单一未截断 RGB L1：`mean(abs(Y.float()-GT.float()))` |
| 优化器 | AdamW，betas=(0.9,0.999)，weight_decay=1e-4 |
| batch / seed / workers | 8 / 100 / 0；每个 epoch 最后不足8张时保留短批次 |
| 学习率 | 2e-4 余弦衰减到 2e-6，按原总步数与绝对 step 计算 |
| 总步数 | 默认10000，支持 `--steps` |
| AMP / 梯度裁剪 | CUDA AMP、CPU FP32，grad_clip=1.0 |
| 验证 | 每500步及最后一步，全部200对，batch=1 |
| 选权重 | 默认 `--best-metric l1`，最低验证 L1；可显式选择 psnr，但必须在记录中注明 |
| 保存 / 日志 | 每1000步另存 checkpoint，另有 last/best；每50步打印 |

训练验证同时记录逐图未截断 L1 和逐图 clamp 后 RGB float PSNR，图像等权平均，PSNR 的 MSE 下限为1e-12。最终四指标基于量化 PNG，不能与训练期 float PSNR 混用。

默认最低 L1 选权重与既有 A 消融计划一致；不据此推定用户后来截图中的所有权重也采用了同样规则。后续正式比较需注明各模型训练预算、选择规则和评价划分。本轮不查旧 runs 或重跑 A 来补比较。

### 5.1 数据校验必须覆盖路径

真实运行开始时才进行预检；实现阶段用合成目录和 manifest 测试。

- 检查 train/val 数量、sample_id 唯一、输出文件名唯一、train/val ID 不重叠。
- 规范化分隔符后，配对必须为 `train/lowlight/同名文件` 与 `train/gt/同名文件`，或对应 val 路径；解析后的实际路径也必须落在对应目录内。
- 核对 manifest 路径与实际目录中的完整文件集合、两侧文件存在性。GT 换成别的同目录文件、指向 test 或不存在文件均应报错。
- 不得只取 lowlight 的 basename 后忽略 manifest 的 GT 路径；训练和评价使用同一配对检查。
- manifest 缺失、不一致时直接报错，不根据目录临时生成新划分。记录实际 manifest SHA256、训练与验证文件名顺序。
- 训练、推理和评价输出不能覆盖数据或旧项目。推理单图/目录输入只读，不依赖 GT。

### 5.2 保存与恢复

复用简单 runner，不搭建调度系统。checkpoint 保存模型、optimizer、GradScaler、完整配置、step、RNG、best score、训练/验证文件名和 manifest 哈希。

恢复严格核对 `star_a_v1`、模型完整配置、数据、batch、seed、损失、学习率计划与选权重规则；余弦计划还必须核对原总步数与 min_lr。支持 `--stop-after N` 只限制本次追加步数，不改变余弦周期。拒绝静默重置优化器或加载其他模型权重。

保持原 A 的采样次序和专用 DataLoader generator，保存随机状态。新训练使用空 run 目录；显式 resume 也不能把另一个实验的记录混入已有 run。非有限 loss 停止并写错误状态，记录梯度范数、AMP scale 与跳过更新情况；不因 OOM 自动改变训练配方。

### 5.3 手动 val 四指标入口

提供 `eval_val.py`，调用既有 `M:\picture data\cholec80_t\train_test\HVI_CIDNet\evaluate_four_metrics.py`，沿用其数值口径，不重写四指标。

- 只处理 val 200 对，保存 `sample_id,video,lowlight_relpath,gt_relpath,output_filename` 的本次 manifest。
- 预测 PNG 使用 `np.rint(clamp(Y,0,1)*255).astype(np.uint8)`，保持尺寸。
- 核对预测文件集合、结果 CSV 的200个唯一 ID 与指标有限性；读取 `output_psnr/output_ssim/output_lpips_alex/output_ciede2000` 并按图等权汇总。
- 评价前重新计算实际数据 manifest 哈希，并与 checkpoint 中的哈希核对；不一致即报错。记录实际使用的数据 manifest 哈希、本次评价 manifest 哈希、checkpoint SHA256、step、选权重规则和 PNG 口径，不能沿用旧哈希充当本次记录。
- 非空评价目录拒绝覆盖，不自动追加 test 或在训练结束后启动评价。
- 既有四指标脚本会自行选择 CUDA；若入口提供 CPU 选项，须通过子进程环境明确限制设备，不能声称 CPU 模式却启动 GPU 评价。依赖缺失时报错，不自动下载权重。

## 6. 最小验收

只执行少量 CPU 合成检查，线程数限制为2，使用本项目独立临时目录并清理。不要运行整套旧模型测试或读取真实数据。

1. **结构、尺寸、初始化**：核对三个尺度、三个原 A GatedConvBlock、两个 Star Block、四个独立频域滤波器及两套 CMU。用小尺寸检查各阶段；确认普通、奇数和1×1输入同尺寸输出。零初始化 head 时输出等于输入；在独立实例改变 head 后确认输出可超出 `[0,1]`。
2. **与官方模块数值对照**：在可被8整除的小特征图上，将同一组权重加载到官方参考 Star Block 与新实现；对照前向和输入梯度。频域滤波器使用固定种子的非全1扰动值，避免恒等滤波掩盖实现错误；CMU 也必须参与对照。只比较必要模块，不运行完整 StarIR。FP32 可采用 `atol=1e-5, rtol=1e-4`，报告实际误差。无法获得官方参考时把该项写成未完成，不拿另一份自写公式作为“官方一致”证据。
3. **可训练性**：不同于输入的随机 GT，3～5个小尺寸 L1 优化步。输出头零初始化使首步主干梯度为零，这是预期；头更新后，检查两处 FMB/DFFN 的频域参数、CMU、主干和 head 有有限非零梯度与实际更新。不能仅靠 AdamW 权重衰减引起的参数变化判为梯度通过。
4. **保存恢复**：微型合成连续训练与实际序列化恢复后的下一步，对照模型输出、学习率、采样顺序和参数；测试使用实际 runner 的保存/恢复辅助函数，包含 optimizer/scaler/RNG，不只复制内存对象。
5. **数据与入口**：合成 manifest 覆盖缺失、重复、数量不符、GT 配错、跨 split 路径、哈希改变等错误；检查 train/infer/eval 的 `--help`。避免为这些检查生成真实尺寸图像或跑 LPIPS。

若检查受文件系统权限或环境限制，记录失败位置与未完成项，不将它写成通过，也不持续重试或监视进程。

另提供但本轮不执行一个 CUDA 整图自检入口：输入 `[8,3,224,448]`，采用正式训练的 AMP/裁剪与优化器设置，执行首步及其后至少3个合成优化步。检查每一步 loss/梯度有限、AMP 是否跳过（首步也要记录）、输出头与频域参数实际更新，并报告同步计时和峰值显存。不要以 loss 有数值就判定训练有效。CPU 检查不能代替这一项。

## 7. 文件与交付

```text
endo_star_a/
  GROK_STAR_A_SPEC.md
  model.py
  star_blocks.py              # 可独立放完整 Star Block 与窗口滤波
  run.py                      # train / infer / resume
  eval_val.py                 # 手动 val 四指标入口
  verify.py                   # CPU 检查与可选 CUDA 检查
  README.md
  IMPLEMENTATION_REPORT.md
  requirements.txt
  checks/
  runs/                       # 用户后续实际启动时创建
```

实现不依赖新的自定义 CUDA 算子或预训练模型。窗口操作可用 torch reshape/permute；如复用 einops，确认现有环境已安装并记录依赖。

`IMPLEMENTATION_REPORT.md` 说明实际替换位置、配置、实测可训练参数量、来源版本、执行过的检查、尚未执行项和任何偏离规格之处。不要预先猜测参数量、显存、速度或指标提升，不把它写成原版 StarIR 整网。

README 提供 CPU、CUDA 合成检查、训练、恢复、推理、val 四指标的可复制 PowerShell 命令，必须与实际 argparse 一致。训练示例按下面配置提供，但本轮不执行：

```powershell
$starAPython = 'M:\Anaconda_envs\envs\retinexformer\python.exe'
$starARoot = 'M:\picture data\cholec80_t\code\cut\endo_star_a'
$starAData = 'M:\picture data\cholec80_t\train_test'

& $starAPython -u "$starARoot\run.py" train `
  --run-dir "$starARoot\runs\star_a_10k_l1_seed100" `
  --data-root $starAData `
  --steps 10000 --batch-size 8 `
  --lr 0.0002 --lr-schedule cosine --min-lr 0.000002 `
  --weight-decay 0.0001 --grad-clip 1.0 `
  --val-every 500 --save-every 1000 --log-every 50 `
  --seed 100 --workers 0 --best-metric l1 --device cuda
```

最终交付聚焦这一版 Star-A。无需额外实现多个变体、颜色先验、RGB 分支、损失组合或完整 StarIR 整网对照。
