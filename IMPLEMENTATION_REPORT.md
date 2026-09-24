# Prior-A 实现报告

日期：2026-09-24。Grok 交付模型和 CPU 自检；Codex 复核时补做 CUDA AMP 合成检查。没有启动真实数据训练。

## 交付文件

- `model.py`：`PriorA`。主干模块顺序与冻结 A 一致，之后才加两个适配器。
- `priors.py`：六通道亮度、固定 CIConv-W、两个适配器。
- `run.py`、`eval_val.py`、`verify.py`、`README.md`、`checks/cpu_checks.json`。
- 规格 `GROK_PRIOR_A_SPEC.md` 未改。未修改旧项目、其他候选、数据或环境。

可训练参数量：**208027**。相对冻结 A 的 206067，增加 **1960**。架构版本 `prior_a_v1`。没有 Fourier 参数。

## 配置

`width=24`，`output_mode=direct`，`spectral_mode=off`，`fusion_mode=gated`。亮度顺序 mean、rec709、vmax、lightness、ycgco、l2norm_scaled。`eps_L=1e-6`，L2 除以 `sqrt(3)`，`gamma_limit=0.5`。结构为 `ciconv_w_rgb_float_v1`，`scale=0.9`（标准差 `2**0.9`），`k=3`，15×15 零填充核。适配器宽度 8。亮度只乘 `encoder2` 第一条完整残差；结构偏置只加在 Spatial Fusion 的 sigmoid 之前。

## 与参考的差别

- Multinex 的 L2 亮度没有除以 `sqrt(3)`。本版加了这个尺度。
- SG-LLIE 离线脚本用 `cv2.imread` 的 BGR，并导出截断 PNG。本版按 RGB 浮点计算，不截断、不转 PNG，因此不是官方先验文件的逐像素复现。核按离散高斯和一阶导数构造，和为 1、导数绝对值和为 1。署名见 `priors.py`。

## 已执行

```text
M:\Anaconda_envs\envs\retinexformer\python.exe -u verify.py --threads 2
```

`checks/cpu_checks.json` 全部通过：

| 检查 | 结果 |
|---|---|
| 亮度公式最大误差 | 2.98e-8 |
| 参数量 | 208027，无 FFT 参数 |
| 共有初始化最大误差 | 0 |
| encoder2 / Spatial Fusion / decoder0 / 非零 head 输出 | 最大误差都是 0 |
| 非零亮度调制、非零结构偏置 | 都会改变约定位置 |
| 适配器末层与前层 | 合成优化后都会更新 |
| 32×48、33×49、严格恢复 | 通过，恢复误差 0 |

## Codex 复核补充

- 复跑 `verify.py --threads 2`：全部通过。原脚本把测试 checkpoint 写进新建临时目录，当前 Windows 沙箱会拒绝在该目录继续创建文件，导致 `tempfile.mkstemp` 重试。已改为在项目现有 `checks` 目录中暂存，并在检查结束后删除测试 checkpoint。
- `inspect_split` 实际检查 train 1500、val 200；manifest SHA256 为 `d969938d27c82e72bd5dce875979e4c2a6c1094c25ae1d7d1267b66e87a68797`，训练文件顺序与原 A 的排序相同。
- CUDA AMP 合成训练：batch 8、224×448、4 步。去除第一次暖机后，A 与 Prior-A 平均分别为 0.091 s/步和 0.101 s/步；峰值已分配显存分别约 3512.7 MiB 和 3604.7 MiB。两者的前向、反向与更新均完成。此时间只计合成计算，不含数据读取、验证或 checkpoint 保存。
- `build_model` 现在检查 checkpoint 中完整的固定模型配置；验证入口要求 checkpoint 记录匹配的 manifest SHA256，并记录实际评价脚本路径与哈希。
- `run.py train` 现在检查 v1 固定的 10000 步训练配方及 CUDA AMP；`--stop-after` 仍可用于分段执行，余弦周期保持 10000 步。

## 真实训练与测试

2026-09-24 按默认 10000 步配方训练，目录 `runs/prior_a_10k_seed100`。日志写到第 **9288** 步后进程退出，没有形成 `complete`。`best_val.pt` 是验证 L1 最低的第 **8000** 步，验证 L1 **0.04898**，验证 float PSNR **25.089**。第 8500、9000 步都更差。

300 对测试（与 A/D 同一 manifest、量化 RGB PNG）：

| 模型 | PSNR | SSIM | LPIPS | CIEDE |
|---|---:|---:|---:|---:|
| Prior-A，第 8000 步 | 25.435 | 0.865 | 0.200 | 5.086 |
| A，1 万步，第 8000 步 | 25.425 | 0.865 | 0.201 | 5.093 |

四项都略好于 A，PSNR 只高 0.010 dB。权重、输出图不进入 Git。

## 偏离

无方法替换。`return_debug` 只用于验收，默认推理不返回它。
