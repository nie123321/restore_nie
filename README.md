# MDTA-GTMean-A：A 主干 + MDTA 瓶颈 + GT-Mean L1

本分支是 A baseline（width 24、`spectral_mode=off`、`output_mode=direct`）在瓶颈替换为
Restormer MDTA 注意力、并使用 GT-Mean L1 训练损失的版本，参数量 **245,719**。
训练协议与 55 epoch 标准一致：整图 448×224、无裁剪无增强、batch 8、seed 100、
AdamW、梯度裁剪 1、CUDA AMP、cosine 2e-4 → 2e-6、55 epoch / 10,340 步；
每 500 步验证，按最低验证 raw L1 保存 `best_val.pt`。

## 结构与损失

- 瓶颈：`MDTAResidual`（Zamir et al., CVPR 2022 / Restormer 的通道注意力 MDTA + 残差，
  q/k 归一化、温度系数、depthwise conv；无额外 GDFN），`mdta_blocks.py`。
- 损失：GT-Mean L1（Liao et al., ICCV 2025；`gt_mean_loss.py`），与官方实现一致：

```
W = clip(Bhattacharyya distance(mean_gray(pred), mean_gray(GT)), 0, 1)   # detached
loss = W * L1(pred, GT) + (1 - W) * L1(clamp(gain * pred, 0, 1), GT)
gain = mean_gray(GT) / mean_gray(pred)                                    # differentiable
```

GT 的灰度均值只参与训练损失，推理不需要 GT。sigma=0.1；灰度均值 ≤1e-6 时回退原始 L1。

## 结果

- `best_val.pt` 在 5,500 步：验证 raw L1 0.049144，clamp 后 FP32 PSNR 25.0835。
- 训练于 10,340 步完整结束；峰值显存 3,686.9 MiB。
- checkpoint SHA256 与完整配置见 `result_mdta_gtmean_best5500.json`。

Test（300 张、12 个视频，clamp + round PNG 四指标；输入基线 PSNR 14.243677 / SSIM 0.487758 /
LPIPS 0.341095 / CIEDE2000 16.921876）：

| 版本 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | CIEDE2000 ↓ |
|---|---:|---:|---:|---:|
| **MDTA-GTMean-A（本分支，best5500）** | **25.643097** | 0.866133 | 0.198554 | **4.850180** |
| 原 U3（7.15M 参数，历史） | 25.546591 | 0.866792 | 0.197788 | 5.055436 |
| A＋MDTA（无 GT-Mean 损失） | 25.503275 | 0.865625 | 0.202940 | 5.030903 |
| Retinexformer 重训 | 25.456256 | 0.868613 | 0.203084 | 5.147427 |
| A baseline（width24，55ep） | 25.448755 | 0.864963 | 0.200453 | 5.086370 |

配对检验（300 张逐图差分，正值表示本版本更高）：

| 对比 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔCIEDE2000 |
|---|---:|---:|---:|---:|
| vs A baseline | +0.194 ± 0.100（t=1.93） | +0.0012（n.s.） | −0.0019（t=−4.9） | −0.236（t=−4.7） |
| vs A＋MDTA | +0.140 ± 0.087（t=1.61） | +0.0005（n.s.） | −0.0044（t=−12.4） | −0.181（t=−4.1） |

结论：PSNR 与 CIEDE2000 为目前全部版本最好；LPIPS、CIEDE2000 的提升统计显著，
PSNR 临界显著。单 seed 结果，PSNR 的 +0.19 仍建议用多 seed 复核。

## 文件

- `model.py`、`run_demo.py`：训练时使用的代码快照（与 run 目录 `code/` 一致）。
- `mdta_blocks.py`：MDTA 注意力残差。
- `gt_mean_loss.py`：GT-Mean L1。
- `star_blocks.py`、`restormer_blocks.py`、`fremlp_blocks.py`：`model.py` 的可选依赖。
- `test_best.py`、`evaluate_four_metrics.py`：test 300 推理 + 四指标评测（脚本内为本地绝对路径，
  迁移时需修改）。
- `result_mdta_gtmean_best5500.json`：配置与结果摘要。
- 未上传 `runs/`：权重、日志、逐张指标与 PNG 均保留在本地工作区。

## 运行

```powershell
$py = 'M:\Anaconda_envs\envs\retinexformer\python.exe'
& $py -u -X utf8 run_demo.py train --run-dir runs/a_mdta_gtmean_l1_sigma01_whole_b8_e55_seed100_20260928 `
    --epochs 55 --batch-size 8 --seed 100 --lr 2e-4 --lr-schedule cosine --min-lr 2e-6 `
    --weight-decay 1e-4 --grad-clip 1.0 --val-every 500 --save-every 1000 --best-metric l1 `
    --width 24 --spectral-mode off --output-mode direct --bottleneck-attention mdta `
    --loss-mode gt-mean-l1 --gt-mean-sigma 0.1 --amp --device cuda --workers 0
```

测试：`test_best.py --checkpoint runs/.../best_val.pt --output-dir runs/.../test_best5500_20260928`。

边界：本分支仅记录一次 seed=100、55 epoch 标准协议的训练与 test；不代表其它 sigma、
其它模块组合或多种子平均结果。
