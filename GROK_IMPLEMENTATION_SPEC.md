# Grok 实现任务：空间自适应颜色映射与小波重建网络

日期：2026-09-23。状态：待实现的第一版设计，尚未验证效果。

## 1. 本轮任务与交付边界

请根据本文实现一个独立、可训练、可恢复训练、可推理的新模型。目标是让用户随后启动一次完整候选实验。此次只实现模型、运行入口和必要自检，不开展模块消融或超参数搜索。

- 所有新代码、说明和后续输出放在 `M:\picture data\cholec80_t\code\cut\endo_color_wavelet`。
- 旧项目 `M:\picture data\cholec80_t\code\cut\endo_enhancement_demo` 是只读参考，原来的 A 和 RGB 分支版本保留。不要修改旧代码、旧运行目录、checkpoint 或旧实验队列。
- 用户负责操作 Grok，Codex 之后做一次核查。本轮不调用其他 AI，不创建监控或自动任务。
- 用户尚未在这条实现指令中授权启动训练。完成实现和轻量 CPU 自检后，提供命令并停止；真实图像训练、验证集推理、四指标计算、GPU 自检均等用户明确启动。不要检查正在运行的实验或 GPU/进程状态。
- 实现阶段不读取真实图像或已有实验结果。可以读取本文指定的旧代码来复用训练、数据与评价接口。
- 数据目录只读。不要选择问题图、制作反光/饱和/配准边缘掩码，也不设计这些区域的专门模块。

这里“重新设计”指保留 U 型编码—解码组织方式，改变前端、各尺度恢复和引导路径。不能只在旧模型输出头后追加一个模块就宣布完成。

## 2. 目标与设计逻辑

整体流程：

```text
低光 RGB 图像 X
  ├─ 低分辨率上下文编码器 → 局部/全局上下文 → 4 张颜色映射权重图
  ├─ 对全分辨率 X 查询 4 个可学习残差 LUT，并按权重混合 → 初步增强图 Z
  └─ 拼接 [X, Z] → 两级小波分解的 U 型重建网络
                      ↑ 每级接收颜色上下文和 Z-X 的引导
                      ↑ 恢复后的低频引导对应高频的修正
                  → 重建残差 R → 最终结果 Y = Z + R
```

两个阶段联合训练：第一阶段负责主要的颜色与明暗映射；第二阶段恢复纹理、处理噪声并修正剩余误差。它们不是严格可辨识的物理分解，不能宣称第一阶段“只改颜色”、第二阶段“只改纹理”。

输入仍然是三通道图像，但不建立 R/G/B 三个独立恢复分支。LUT 联合使用三个输入通道，后续网络在混合特征空间工作。

已有通道残差差异只是设计动机，不能据此固定增强某一通道，或宣称已证明某个成像机制。第一版是工程候选，不预先声明论文创新或预计提升幅度。

## 3. 本地参考接口

以下路径已根据旧代码核对；本文没有读取实时训练状态或结果。

| 用途 | 路径/接口 |
| --- | --- |
| 旧模型基础卷积块 | `M:\picture data\cholec80_t\code\cut\endo_enhancement_demo\model.py` 中 `LayerNorm2d`、`GatedConvBlock` |
| 数据与训练逻辑 | `M:\picture data\cholec80_t\code\cut\endo_enhancement_demo\run_demo.py` |
| 值得复用的函数 | `PairedImages`、`StepBatches`、`learning_rate`、`atomic_save`、RNG 保存恢复、输出目录保护、验证/推理的数值口径 |
| 验证集评价封装参考 | `M:\picture data\cholec80_t\code\cut\endo_enhancement_demo\run_ablation_queue.py` 中 `evaluate`；只参考封装，不启动该队列 |
| 四指标计算脚本 | `M:\picture data\cholec80_t\train_test\HVI_CIDNet\evaluate_four_metrics.py` |
| Python 环境 | `M:\Anaconda_envs\envs\retinexformer\python.exe` |
| 数据根目录 | `M:\picture data\cholec80_t\train_test` |

旧读取器使用 `train/lowlight` 对 `train/gt`、`val/lowlight` 对 `val/gt`，按相同文件名配对。现有协议是 train 1500 对、val 200 对，另有 test 300 对；数量和划分在真实实验启动时核对 `split_manifest.csv`，本轮不要扫描图像。

旧模型源码可以复制必要的小块到新目录并注明来源；不要用修改 `sys.path` 后导入旧 `model.py` 的方式建立运行依赖。后续改旧项目不应悄悄改变这个新模型。

## 4. 第一阶段：空间自适应残差 LUT

### 4.1 输入和上下文编码器

- 输入 `X`：`[B,3,H,W]`，encoded RGB，数值范围 `[0,1]`。不把它冒称线性传感器数据。
- 对整个模型先做右侧、下侧 replicate padding，使 H/W 为 4 的倍数。最后所有图像输出裁回原尺寸。不要用会在 1×1 图像上失败的 reflect padding。
- 将补齐后的输入用 area 下采样到 `H/4 × W/4`，经 `3×3 Conv(3→32)` 和 2 个 GatedConvBlock，得到 `C0`。
- 用两次 `3×3 stride=2, padding=1` 卷积，分别得到 64 通道 `C1`、128 通道 `C2`，每次后面各接 2 个 GatedConvBlock。
- 允许 C1/C2 的空间尺寸向上取整，后续使用明确的目标尺寸插值。
- 不使用 BatchNorm 或依赖整批图像统计的颜色标准化。

### 4.2 权重图

在 C0 尺度融合：

```text
Cmix = C0 + resize(Conv1x1(C1→32), size(C0))
           + resize(Conv1x1(C2→32), size(C0))
local_logits = Conv3x3(GELU(Cmix), 32→K)
global_logits = Linear(GlobalAvgPool(C2), 128→K)
logits = local_logits + global_logits[:, :, None, None]
w = softmax(resize(logits, full_size), dim=1)
```

- 默认 `K=4`；空间插值使用 bilinear、`align_corners=False`。
- 每个像素的 4 个权重非负且和为 1；权重由输入上下文预测，不用 GT、视频 ID 或人工区域标注。
- 权重图是连续混合系数，不解释成组织分割或问题区域标签。

### 4.3 LUT 定义与应用

默认 4 个可学习残差 LUT，每个分辨率 `D=17`。LUT 是跨训练样本共享的模型参数，当前图像通过权重图选择/混合它们。

```text
delta_luts: [K, 3, D, D, D]
delta_k(p) = trilinear_sample(delta_luts[k], X(p))
Z(p) = X(p) + sum_k w_k(p) * delta_k(p)
```

必须对全分辨率 X 查表；只有上下文/权重预测使用低分辨率。不要把低分辨率颜色图直接放大作为 Z。

采用 PyTorch 原生 5D `grid_sample`，无需第三方 CUDA 扩展。明确规定 LUT 的存储顺序：

```text
delta_luts[k, output_channel, b_index, g_index, r_index]
grid 最后一维 = (2*R-1, 2*G-1, 2*B-1)
align_corners=True，padding_mode='border'，mode='bilinear'
```

5D 输入下 `bilinear` 执行三线性插值。一个可行的调用形状是表 `[B,3,D,D,D]`、grid `[B,1,H,W,3]`，去掉输出的长度为 1 的深度维。可以遍历 4 张 LUT，避免不必要的大张量复制。

查表、权重 softmax 和颜色加法使用 FP32，可在局部关闭 autocast；必须保留对 LUT 和权重预测器的梯度。

初始化使用幅度约 `1e-3` 的、彼此不同的平滑随机残差表，例如将独立随机 `4×4×4` 小表三线性放大到 `17³`。不要让所有 LUT 和路由参数完全对称，导致各表一直学成同一个结果。用固定种子保证可复现。

Z 不做硬 clamp，不限制只能提亮，也不强制各输出通道对所有输入通道单调。训练通过原始 RGB 损失约束范围；可视化时才截断到 `[0,1]`。

## 5. 第二阶段：颜色引导的小波 U 型重建

### 5.1 默认容量与明确数据流

- 两次小波下采样，三个尺度，宽度 `[32,64,128]`。
- 编码器每级 2 个 GatedConvBlock；最低尺度 6 个；两级解码器各 2 个。
- 每一级高频修正分支使用 2 个 GatedConvBlock，通道数为该级 3 个高频子带拼接后的通道数。
- 这是允许达到百万级参数的新候选，报告实际统计值即可，不为凑参数量添加无关层。

准确的数据流如下，`B_c^n` 表示 n 个 c 通道 GatedConvBlock：

```text
F0 = B_32^2(Conv3x3(concat(X,Z), 6→32))            # H × W
L0, H0 = DWT(F0)                                 # H/2 × W/2，32 / 96 通道
F1 = B_64^2(Conv1x1(L0, 32→64))
L1, H1 = DWT(F1)                                 # H/4 × W/4，64 / 192 通道
F2 = B_128^6(Conv1x1(L1, 64→128))

L1_hat = GuidedLowBlock(Conv1x1(F2, 128→64), L1, guide_level1)
H1_hat = GuidedHighBlock(H1, L1_hat, guide_level1)
U1 = IWT(L1_hat, H1_hat)                         # H/2 × W/2，64 通道
U1 = B_64^2(U1)

L0_hat = GuidedLowBlock(Conv1x1(U1, 64→32), L0, guide_level0)
H0_hat = GuidedHighBlock(H0, L0_hat, guide_level0)
U0 = IWT(L0_hat, H0_hat)                         # H × W，32 通道
U0 = B_32^2(U0)
R = Conv3x3(U0, 32→3)
Y = Z + R
```

原始 L0/L1 经 GuidedLowBlock 有条件地融合；不要额外引入未经说明的整条旧 A 网络旁路。最后的 R 不限制为高频，使网络可以修正前端剩余的低频错误。

### 5.2 Haar DWT/IWT

用纯 PyTorch 固定 Haar 变换，无需安装小波库。DWT 作用于学习特征，而不是分别建立 RGB 网络。

对每个通道的 2×2 块，设 `a=左上，b=右上，c=左下，d=右下`：

```text
LL = ( a + b + c + d) / 2
LH = (-a - b + c + d) / 2
HL = (-a + b - c + d) / 2
HH = ( a - b - c + d) / 2

a = (LL - LH - HL + HH) / 2
b = (LL - LH + HL - HH) / 2
c = (LL + LH - HL - HH) / 2
d = (LL + LH + HL + HH) / 2
```

高频统一按 `[LH, HL, HH]` 沿通道拼接，IWT 严格对应这个顺序。子带在可逆变换后进入可学习处理；不要宣称整个网络无损或小波完全分离信号与噪声。

### 5.3 颜色条件与低频融合

在 `H/2`、`H/4` 两个尺度分别构造条件，通道目标 c 分别为 32、64：

```text
Q_s = Conv3x3(concat(resize(Cmix, size_s), resize(Z-X, size_s)), 35→c)
Q_s = GELU(Q_s)
F = Conv1x1(concat(projected_decoder_low, original_LL), 2c→c)
[gamma, beta] = Conv1x1(Q_s, c→2c)
F = F * (1 + 0.1*tanh(gamma)) + 0.1*tanh(beta)
L_hat = GatedConvBlock(F)
```

`gamma/beta` 输出卷积可以零初始化，使条件调制从中性状态开始。不要 detach Cmix、Z-X 或 Q_s；最终输出的损失需要能够反传到第一阶段。

### 5.4 低频引导高频修正

同一尺度上原始高频 H 有 `3c` 通道，低频 L_hat 和颜色条件 Q_s 各 c 通道：

```text
T = Conv1x1(concat(H, L_hat, Q_s), 5c→3c)
T = 两个 GatedConvBlock(T)
delta_H = Conv3x3(T, 3c→3c)
gate_H = sigmoid(Conv1x1(concat(L_hat,Q_s), 2c→3c))
H_hat = H + gate_H * delta_H
```

此处门控控制修正量，不把整片高频直接乘到接近零。`delta_H` 最后一层使用小尺度非零初始化；最终 R 输出卷积也使用小尺度非零初始化，例如权重标准差 `1e-3`、bias=0，方便首轮检查整条链路梯度。

不加入 FFT、Mamba、扩散、预训练语义网络或额外 RGB 独立分支。第一版按上述完整方案实现，避免实现者自行叠加另一套研究路线。

## 6. 模型接口和精度

建议类名 `ColorWaveletNet`，默认接口：

```python
y = model(x)  # Tensor，与 x 同尺寸，FP32、未截断 RGB
aux = model(x, return_aux=True)
# aux 至少包含 {"output": y, "coarse": z, "weights": w}
```

`weights` 为裁回原尺寸的 `[B,4,H,W]` 张量。必要时通过明确的 debug 接口暴露两个尺度的条件供自检，不默认保存中间特征到磁盘。

模型的构造参数需能由 checkpoint config 完整重建：架构版本、通道数、各级深度、LUT 数量/分辨率。模型只接收输入图像；GT 仅用于外部训练损失。

卷积允许 AMP；归一化统计、LUT 插值、最终图像加法和颜色空间损失保持数值稳定。CPU 应可完整运行，不依赖 CUDA 扩展。输入很小或宽高为奇数时，仍返回正确尺寸。

## 7. 损失：第一版固定定义

所有图像平均、通道平均、像素平均使用 mean，避免损失随分辨率变化。以下权重是本候选的起始配置，不是文献证明的最优值，应写入配置并允许命令行覆盖。

```text
L_rgb = mean(abs(Y - GT))

P(I) = area 下采样到 (max(1,H//8), max(1,W//8))
L_coarse = mean(abs(P(Z) - P(GT)))

Lab_y = sRGB_to_CIELab_D65(clamp(Y,0,1))
Lab_t = sRGB_to_CIELab_D65(GT)
L_ab = mean(abs((Lab_y[:,1:3] - Lab_t[:,1:3]) / 128))

L_lut = 三个 LUT 网格轴上相邻残差表值差的平方均值之和

L_total = L_rgb + 0.2*L_coarse + 0.05*L_ab + 1e-4*L_lut
```

- 前三项分别约束最终重建、前端整体外观和色度。L_lut 是很轻的平滑正则，抑制查表值突变；只计算残差 LUT，不把空间权重图强行平滑成常数。
- L_rgb 和 L_coarse 在未截断输出上计算；只在 Lab 转换前截断 Y，避免色域外值破坏颜色转换。主 L1 仍为色域外输出提供梯度。
- Lab 转换使用标准 sRGB 解码、D65 XYZ 和分段 Lab 公式，不能直接把 encoded RGB 当作线性 RGB。用原生 torch 可微实现即可，所有幂函数的分支在零值和负值处都要数值安全，不能依赖 torch.where 隐藏 NaN 梯度。
- 不直接把 CIEDE2000 当训练损失；它仍是最终评价指标。
- 每个日志条目分开记录这四项以及总损失。先明确各项尺度，不自行追加 LPIPS、SSIM、对抗或频域损失。

## 8. 训练、保存、恢复的契约

复用旧训练流程的成熟部分，在新目录实现入口，不对旧 runner 打补丁。建议首次完整候选配置如下；此表是后续执行计划，不是授权立即运行。

| 项目 | 第一版设置 |
| --- | --- |
| 数据 | 现有 train/val 划分，整图，按原文件名配对 |
| 像素域 | encoded RGB `[0,1]`，保持现有协议 |
| 尺寸 | 读取原始训练图尺寸；现有整图协议为 448×224（宽×高） |
| 增强 | 沿用旧流程，无随机裁剪、resize、颜色增强或 GT 均值匹配 |
| 优化器 | AdamW，weight_decay=1e-4，默认 betas=(0.9,0.999) |
| Batch | 8，保留旧采样器最后不足 8 的批次行为 |
| 学习率 | 2e-4，余弦衰减至 2e-6 |
| 总步数 | 建议 30000；必须可由参数指定 |
| seed | 100 |
| 精度 | CUDA AMP；CPU FP32 |
| 梯度裁剪 | 1.0 |
| 验证 | 每 500 步，完整 200 对；训练结束时也验证 |
| 保存 | 每 1000 步 + last + best，日志每 50 步 |
| workers | 0 |
| 选权重 | 提供 `l1` 与 `psnr` 两种模式；建议首轮 `psnr`，配置中明确记录 |

注意：与 A 做同预算比较时，训练步数和选权重规则必须与对应 A 结果一致。不能将旧 10k/min-L1 结果冒充新 30k/max-PSNR 的严格对照；实现阶段不为此扫描已有 runs。

训练时的验证口径沿用旧代码：逐图未截断 L1；逐图 `clamp(Y,0,1)` 与 GT 计算 RGB float PSNR，再做图像等权平均，MSE 下限 1e-12。训练验证和保存为 PNG 后的四指标不是同一个数值口径。

必须支持：

1. 输出目录非空时拒绝隐式覆盖，恢复训练需显式 `--resume`。
2. checkpoint 保存 model、optimizer、scaler、step、完整 config、RNG、best score、训练文件名和数据划分记录。
3. 学习率由原目标总步数和绝对 step 决定。短程 `--stop-after` 不改变余弦总周期。
4. 恢复时严格核对架构、损失权重、数据划分/文件名、batch、seed、学习率计划、目标步数、选权重规则；不要静默重置 optimizer/scaler。
5. AMP 跳过更新需记录，不能把“无报错”当作确实完成了参数更新。出现非有限值应报告并停止，不自动减小模型、batch 或修改损失来掩盖。
6. 记录每项损失、学习率、验证结果、实际参数量。CUDA 显存与速度只在用户后来运行时测量，不在报告中猜测。

建议生成 `config.json`、`loss.jsonl`、`status.json`、`last.pt`、`best_val.pt`、`step_XXXXXX.pt`。全部写入新项目下用户指定的 run 目录。保存 config 时包含模型版本，禁止误加载旧 A 的权重冒充新模型恢复。

## 9. 推理与四指标入口

推理只使用 checkpoint 和输入图像。输出 PNG 沿用旧格式：

```python
np.rint(output.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
```

不得使用 GT 均值校正、直方图匹配、推理时拟合参数或 test 图像挑选。

准备一个手动调用的 `eval_val.py`：从新 checkpoint 对 val 推理，再调用现有四指标脚本。接口封装参照旧 `run_ablation_queue.py` 的 `evaluate`，不要复制或启动它的 A/B/C/D 队列。

- 建立本次评价的 `manifest.csv`，字段为 `sample_id,video,lowlight_relpath,gt_relpath,output_filename`，仅选择既定 val 行。
- 预测保存到新实验目录的 `enhanced` 子目录。
- 四指标脚本支持 `--experiment-root`、`--method-label`、`--dataset-root`。实施时可阅读该脚本确认依赖与接口，但本轮不运行它。
- 读取 `metrics_per_sample.csv` 中 `output_psnr`、`output_ssim`、`output_lpips_alex`、`output_ciede2000`，核对 val 的 sample_id 完整且数值有限，输出图像等权均值与简短汇总。
- LPIPS 的网络/输入归一化、SSIM 和 CIEDE2000 均沿用已有评价脚本，不自行更换库口径。
- 如果评价依赖尚未就绪，明确报告，不下载权重或将缺失指标填零。
- 本轮只写入口，后续由用户启动。不要默认跟在训练后自动评估，更不要追加 test 评估。

## 10. 建议文件组织

```text
endo_color_wavelet/
  GROK_IMPLEMENTATION_SPEC.md    # 本文，保留为规格
  model.py                       # ColorWaveletNet、上下文/引导/重建模块
  lut.py                         # 残差 LUT 查询和混合
  wavelet.py                     # Haar DWT/IWT
  losses.py                      # 联合损失、稳定的 Lab 转换
  run.py                         # train / infer，含恢复功能
  eval_val.py                    # 手动验证集四指标入口
  verify.py                      # 小型合成输入自检
  requirements.txt               # 确有必要的依赖
  README.md                      # 使用方式及命令
  IMPLEMENTATION_REPORT.md       # 实际实现、检查结果、偏离规格之处
  checks/                        # 轻量自检结果
  runs/                          # 用户后续启动的实验
```

无需搭建通用训练框架、插件系统、任务调度器或大量配置文件。文件可按实现需要小幅合并，但要容易定位模型、LUT、小波、损失和训练入口。

优先使用现有 torch/numpy/Pillow。正式四指标评价继续使用既有评价环境；不要为了复现论文仓库安装 Mamba、自定义 LUT CUDA 算子或升级现有 PyTorch。

## 11. 必要验收：先小型 CPU 合成测试

检查要证明关键计算有效，不只验证文件存在或 forward 能运行。控制 CPU 线程数和输入大小，避免耗时长的“大而全”测试。汇总写入 `checks/cpu_checks.json`，只记真实执行结果。

| 检查 | 要求 |
| --- | --- |
| 尺寸与有限值 | 正常小图、奇数宽高和 1×1 输入的 Y/Z 尺寸正确，无 NaN/Inf |
| DWT/IWT | FP32 随机特征往返 max error < 1e-5；常数图高频接近零；四个单点输入检查子带符号和排列 |
| LUT 轴顺序 | 用解析残差表实现已知通道置换，如输出 `[G,B,R]`，检查采样结果，避免只测 identity 掩盖轴顺序错误 |
| 空间权重 | 4 个权重逐像素非负、和为 1；在构造的不同 LUT 下，改变局部权重仅按公式改变相应结果 |
| 关键梯度 | 用不同于输入的合成 GT 跑少量（例如 3–5）优化步骤，LUT、路由/上下文、低频、高频、引导和最终输出层存在有限非零梯度，并确实发生参数更新 |
| 最终损失的反传 | 单独从 Y 的重建损失反传，确认第一阶段获得梯度，而不只是靠 coarse 辅助损失更新 |
| 引导有实际作用 | 临时改变颜色条件或 L_hat 后，对应解码/高频结果发生变化，排除计算了条件却没有接到输出 |
| Lab 数值 | 黑、白、中性灰的色度接近零；固定标准色的 Lab 值与一组预先写明来源/标准的参考值一致；转换和损失梯度有限 |
| 序列化与恢复 | checkpoint 严格加载后推理一致；微型合成流程中连续训练与保存后恢复的下一步学习率、样本顺序及 CPU 参数结果一致（允许合理浮点误差） |
| 入口 | train/infer/eval_val 的 help 可运行；测试不读取真实数据、不启动训练队列 |

临时干预测试要在独立实例或恢复参数后进行，不污染待训练模型。不要将近似 identity 初始化错误地要求为逐位 identity；残差 LUT 本来就有微小不同的初始化。

CUDA AMP、自定义输入 `B=8,3,224,448` 的显存与优化步检查另提供可运行命令，例如 `verify.py --device cuda --full-shape`，本轮不要执行。在报告中列为“待用户启动”，不能写成已经通过。

## 12. README 与最终交付要求

README 提供已与实际 argparse 对齐的 PowerShell 命令，至少包括：

1. CPU 自检。
2. 可选 CUDA AMP/整图自检。
3. 建议配置的 30000 步训练，但不执行。
4. 使用同一完整配置从 last.pt 恢复。
5. best_val.pt 的单图/文件夹推理。
6. val 200 对四指标评价。

PowerShell 一律正确引用含空格和中文的绝对路径。使用如 `$cwPython`、`$cwRoot` 的变量名；不要覆盖 HOME/CODEX_HOME。示例运行目录位于新项目内。不要创建会自动执行的后台训练脚本或计划任务。

`IMPLEMENTATION_REPORT.md` 用简短条目交代：

- 创建了哪些文件、实际模型参数量、架构/损失与本文是否一致。
- 已执行的命令和自检结果；未执行的 GPU/真实数据检查明确列出。
- 如遇环境依赖或必须偏离规格，说明具体原因和影响，不能自行换成另一套方法。
- 供用户直接复制的训练及恢复命令。
- 本轮没有启动正式训练、没有修改旧项目、没有读取实验结果。

完成后停止，等用户后续命令。用户会再请 Codex 核查；无需主动调用 Codex/Grok/其他代理，也不要循环检查进度。

## 13. 思路来源与边界

- [SVDLUT，ICCV 2025](https://arxiv.org/html/2508.16121v1)：借鉴颜色映射与空间上下文结合的思路。原文使用 LUT/双边网格分解及 SVD；本文采用 4 个残差 3D LUT 与空间权重，是本候选的简化实现，不声称复现其完整模块。[官方代码](https://github.com/WontaeaeKim/SVDLUT)。
- [CAGE，2026 年预印本](https://arxiv.org/html/2608.10512v1)：借鉴在主干恢复前处理颜色偏差的安排。本方案没有实现 AdaLAB/AdaCCT，不以 CAGE 的名称标注本模块，也不直接移用论文中的性能结论。
- [Wave-Mamba，ACM MM 2024](https://arxiv.org/html/2408.01276v1)：借鉴小波多尺度和低频引导高频恢复。本文使用卷积条件模块，没有复现其 Mamba 或具体高频匹配算子。[官方代码](https://github.com/AlexZou14/Wave-Mamba)。
- [FlowLUT，2025 年预印本](https://arxiv.org/abs/2509.23608)：已有颜色 LUT 与后续恢复结合的相关工作。本文不使用 flow matching；组合思路本身不作为已确立的新颖性声明。

这些论文用于说明设计依据，性能只在用户后续实验后判断。不要照搬论文的自然图像结论来证明本数据集的机制。
