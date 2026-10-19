# B_vgg0005：B_joint_subband_residual + 0.005 VGG 感知损失（2026-10-19）

在 B_joint_subband_residual 基础上加入 0.005 × VGG19 relu3_3 MSE 感知损失，其余结构与训练设置不变。

## 本分支附带权重

`checkpoint/seed100_step037500_best_psnr.pt`：seed100，第 37,500 步（训练中 test PSNR 最高点），数据为 train1700/test300 划分。

| PSNR | SSIM | LPIPS-Alex | CIEDE2000 |
|---|---|---|---|
| 26.047 | 0.8767 | 0.2214 | 4.663 |

逐图结果见 `test_result/metrics.csv`，汇总见 `test_result/summary.json`，该次训练的完整配置见 `checkpoint/seed100_run_config.json`。

对比原版 B（无 VGG）：seed102 第 42,000 步 26.068 / 0.8786 / 0.2264 / 4.679；seed101 第 15,000 步 26.032 / 0.8784 / 0.2247 / 4.674。

测试：

```bash
python test.py --checkpoint checkpoint/seed100_step037500_best_psnr.pt --data-root /path/to/dataset --output results
```

---

## 原 B 包说明


可独立上传、解压后训练的正式包。模型为 V3 的 MSWF-B 版本，参数 940,427；原 V3 为 1,063,115。

两级小波各用独立的四子带联合残差块：四子带分别归一化后拼接，PW(4C→C/2)→DW3→GELU→PW(C/2→4C)，拆为四路修正，加回原始子带。输出 PW 零初始化。替换原低频专用仿射与高频独立处理。

11 处 MSWF、两级 DWT/IDWT、外层空间路径与 FFN、编码解码和跳连保留。两处 LFGSI、V3 theta、原串联顺序保留。每个模块、尺度、方向需要独立的权重均独立；没有与早期 V4 共用子带连接。

## 默认训练

随机初始化，从头训练；不加载旧 V3 或其他变体权重。

- 默认数据 `M:/picture data/cholec80_t/训练`，1700 对 train、300 对 test。
- 100,000 步、batch 8、128×128、seed 100、workers 4。
- AdamW，学习率 2e-4→2e-6，余弦覆盖完整 100,000 步；weight decay 1e-4，梯度裁剪 1。
- 保留 V3 的 D4、成对饱和度增强 p=0.25、范围 0.9–1.1。
- 没有输入亮度增强；没有采用 brightness_cosine50k 包的 50,000 步余弦设置。
- 损失为 GT-Mean L1(sigma=0.1) + 0.1 complex FFT L1 + 0.005 × VGG19 features[:16]（relu3_3）特征 MSE（ImageNet 归一化，VGG 冻结，权重随包在 `weights/vgg19_features16.pth`，由 torchvision vgg19 导出）。无 SSIM/HF 训练损失；FFT 权重为 0 时跳过 FFT。
- FP32，AMP/TF32 关闭，不强制严格 deterministic。

默认每 500 步在 test 计算全图 RGB uint8 PSNR/SSIM，test 不反向传播；按你已选的 train/test 方案监测和选权。另行运行 test.py 才计算 LPIPS-Alex 和 CIEDE2000，训练进程不会额外创建感知网络。

在本包目录运行：

```bash
pip install -r requirements.txt
python train.py
```

默认输出为本包 `outputs/seed100`。第二个随机种子：

```bash
python train.py --config config_seed101.json
```

服务器指定数据路径；两版分别进入各自目录执行，可使用不同输出目录或 GPU：

```bash
bash train_server.sh --data-root /path/to/dataset --output outputs/seed100
```

默认后台启动并打印 PID 和日志路径。前台用 `--foreground`，指定 Python 用 `--python /path/to/python`。也可直接运行 `python -u train.py --data-root /path/to/dataset --output outputs/seed100`。

Windows：

```powershell
.\train_local.ps1
# 前台运行：
.\train_local.ps1 -Foreground
```

默认使用本机 `M:\Anaconda_envs\envs\retinexformer\python.exe`，可用 `-PythonPath` 改；可用 `-DataRoot`、`-OutputPath`、`-Config` 指定路径。

## 与表中的旧 V3 比较

默认 1700/300 与旧 V3@33k 的 1500/200/300 数据协议不同。默认配置下的新结果不能直接当作相对旧表的纯结构提升。

如需复用旧 V3 的对照结果，两版都选择附带的 `config_original_split.json`：使用原 train_test 的 train/val、V3 原浮点评估口径，其他训练配方相同。

```bash
python train.py --config config_original_split.json --data-root /path/to/original_train_test --output outputs/original_split_seed100
```

该配置的 val 不参与训练，最终 test.py 仍在 test 上按 RGB uint8 PNG 算四指标。此为显式的实验配置选项，不是旧权重兼容分支。

## 保存和续训

保存 `step_XXXXXX.pt`、`last.pt`、`best_psnr.pt`、`best_ssim.pt`；`selection.json` 记录选点，`train.jsonl` 记录训练和评估，`status.json` 记录运行状态，`code/` 保存本次源码和配置快照。

```bash
python train.py --resume outputs/seed100/last.pt
```

服务器续训重复原来的 `--config`、`--data-root`、`--output` 参数。只接受本版生成的断点；模型标识、数据清单、训练配置、源码均核对。A/B、原 V3、旧 R3 断点不互相混用。模型推理使用本版配置和原生 `strict=True`，没有旧参数名映射或旧随机状态格式适配。

## 测试和推理

```bash
python test.py --checkpoint outputs/seed100/best_ssim.pt --output outputs/test_best_ssim
# 服务器增加 --data-root /path/to/dataset
python test.py --checkpoint outputs/seed100/last.pt --input /path/to/image_or_folder --output outputs/inference
```

输出增强 PNG、逐图 `metrics.csv`、四指标 `summary.json` 和推理元信息。每行报告四指标应来自同一个 checkpoint，不混用各指标最好点。已有输出目录不覆盖。

本包的训练/测试入口独立，不依赖上级工作区、原型目录或另一版本。交付时未启动正式训练；构建和必要的小样本检查不代表质量改善。
