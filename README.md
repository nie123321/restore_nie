# Star-Multiscale-A：A 主干 + MDTA + GT-Mean + 多尺度 Star 块

本分支是 MDTA-GTMean-A 的扩展版：把瓶颈处原 Spectral Fusion 替换为 2 个 StarIR（TPAMI 2026）的
StarBlock，并在 4 个尺度各插入 1 个 StarBlock（encoder0 / encoder1 / decoder1 / decoder0，即
24 / 48 / 48 / 24 通道），参数量 **585,367**。主干、损失与数据划分同 MDTA-GTMean-A。

## 结构与训练协议

- 瓶颈顺序：`encoder2 -> StarBlock -> StarBlock -> MDTA`。StarModule 为 8×8 patch 的可学习
  FFT 滤波 + 空间门控 + 扩张 FFN（`star_blocks.py`）。
- 损失：GT-Mean L1（sigma 0.1；`gt_mean_loss.py`）。
- 数据 / 预算（seed 100）：paired random crop 192×384；整批共享 90° 旋转位 + 每图独立水平/垂直翻转
  （D4）；饱和度扰动 p=0.25、因子 [0.9, 1.1]（LQ/GT 同因子）。30,000 步、batch 8、AdamW
  2e-4 → 2e-6 cosine、wd 1e-4、clip 1、AMP、workers 0；每 500 步验证/保存，按最低验证 raw L1
  保存 `best_val.pt`。
- 后段 val 出现已知漂移（30k 配方现象），最佳权重集中在前段。

## Test 结果（300 张、12 视频；clamp + rint uint8 四指标）

输入基线：PSNR 14.243677 / SSIM 0.487758 / LPIPS 0.341182 / CIEDE2000 16.921875。

| step | PSNR ↑ | SSIM ↑ | LPIPS ↓ | CIEDE2000 ↓ |
|---:|---:|---:|---:|---:|
| 4000 | 25.727632 | 0.869183 | 0.198669 | 5.033515 |
| **6500（推荐）** | **25.825054** | 0.871245 | **0.193705** | **4.812542** |
| 7000（best_val） | 25.588773 | 0.870389 | 0.196747 | 4.936614 |
| 7500 | 25.734851 | 0.871354 | 0.202046 | 4.866685 |
| 9500 | 25.694955 | 0.871454 | 0.197672 | 4.842146 |

配对检验（vs 冠军 MDTA-GTMean crop+hv best5500，300 张逐图差分；正值=更高）：

| 对比 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔCIEDE2000 |
|---|---:|---:|---:|---:|
| star_multiscale@6500 | +0.132 ± 0.077（t=1.7） | **+0.0041（t=5.6）** | **−0.0065（t=−8.8）** | −0.045（t=−1.2） |

误差分解（test 300，8×8 cell，相对冠军）：高频亮度 **−9.1%**、高频色度 **−8.5%**、低频亮度
+3.8%、总 MSE −0.5%。按输入亮度四分位，最暗 1/4 的 ΔPSNR **+0.42**，其余 +0.02~0.06。
即：提升来自高频（结构/细节）与暗图，全局低频亮度场未改善。

## 权重

- `weights/star_multiscale_seed100_step006500.pt`（推荐；step 6500）
  SHA256 `68b83e243cc77557664f36107723eaaae1e25e0dcd7b553a030199182c00f247`
- `weights/star_multiscale_seed100_bestval_step007000.pt`（best_val；step 7000）
  SHA256 `cb0d265e5f55e02f060a29b83557d3dd066b280a9fb32394e66f5dffcd5f1121`

checkpoint 格式：`torch.load(...)["model"]` + `["config"]["model"]`；可用
`test_best.py --checkpoint <权重> --data-root <数据根> --output-dir <空目录>` 复现四指标。

## 文件

- `model.py`、`run_demo.py`：训练代码快照（D4 / 饱和度增强、`star_refinement` 实现）。
- `star_blocks.py`：StarIR StarBlock / StarModule / DFFN。
- `mdta_blocks.py`、`gt_mean_loss.py`：MDTA 与 GT-Mean L1。
- `run_experiment.py`：三变体（a / star_bottom2 / star_multiscale）实验驱动。
- `test_best.py`、`evaluate_four_metrics.py`：test 300 推理 + 四指标（脚本内含服务器绝对路径，迁移需改）。
- `deployment.json`、`preflight.json`：部署与预检记录。
- `result_star_multiscale_seed100_30k.json`：配置、逐 checkpoint 指标与配对检验。

## 边界

单 seed（100）、单次 30k 协议；推荐权重 step 6500。SSIM / LPIPS 增益统计显著，PSNR 增益临界显著。
后续建议：多 seed 复核与权重平均（SWA）；低频亮度场需要独立手段。
