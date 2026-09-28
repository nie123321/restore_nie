# Width32-A：A 主干加宽版（width 32）

本分支是原始 A（门控卷积共享 U 型主干，`output_mode=direct`）在宽度 32 下的代码与测试记录。
结构与训练协议和 55 epoch 标准一致：整图 448×224、无裁剪无增强、batch 8、seed 100、
AdamW、L1、梯度裁剪 1、CUDA AMP、cosine 2e-4 → 2e-6、55 epoch / 10,340 步；
每 500 步验证，按最低验证 raw L1 保存 `best_val.pt`。

## 结果

参数量 360,515（width 24 的 A baseline 为 206,067）。

- `best_val.pt` 在 2,000 步：验证 raw L1 0.049336，clamp 后 float PSNR 25.0508。
- 训练于 10,340 步完整结束；峰值显存 4,668.8 MiB。
- checkpoint SHA256 与完整配置见 `result_width32_best2000.json`。

Test（300 张、12 个视频，clamp + round PNG 四指标；输入基线 PSNR 14.243677 / SSIM 0.487758 /
LPIPS 0.341095 / CIEDE2000 16.921876）：

| 版本 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | CIEDE2000 ↓ |
|---|---:|---:|---:|---:|
| Width32-A best2000 | 25.262555 | 0.863539 | 0.208566 | 5.333527 |
| A baseline width24（同协议 55 epoch） | 25.4488 | 0.8650 | 0.2005 | 5.0864 |

结论：单纯加宽（参数 +75%）在 test 四项上均未超过 width24 baseline，这条路线未取得收益。

## 文件

- `model.py`、`run_demo.py`：当前主线代码（含 `--flip-hv`、star 融合、mdta 瓶颈等实验开关；
  本分支只使用 `--width 32 --spectral-mode off --output-mode direct`）。
- `star_blocks.py`、`mdta_blocks.py`：`model.py` 的可选依赖。
- `test_best.py`：test 300 推理 + 四指标驱动；脚本内的评测脚本为本地绝对路径，迁移时需修改。
- `evaluate_four_metrics.py`：四指标评测脚本快照（PSNR / SSIM / LPIPS-Alex / CIEDE2000）。
- `requirements.txt`：最小依赖。
- 未上传 `runs/`：权重、日志、逐张指标与 PNG 均保留在本地工作区。

## 运行

```powershell
$py = 'M:\Anaconda_envs\envs\retinexformer\python.exe'
& $py -u -X utf8 run_demo.py train --run-dir runs/a_width32_whole_b8_e55_seed100_20260927 `
    --epochs 55 --batch-size 8 --seed 100 --lr 2e-4 --lr-schedule cosine --min-lr 2e-6 `
    --weight-decay 1e-4 --grad-clip 1.0 --val-every 500 --save-every 1000 `
    --width 32 --spectral-mode off --output-mode direct --amp --device cuda --workers 0
```

测试（需先准备 300 张 test 清单与评测脚本）：

```powershell
& $py -u -X utf8 test_best.py --checkpoint runs/.../best_val.pt `
    --output-dir runs/.../test_best2000_20260927 --method-label A-width32-best2000-test
```

边界：本分支仅记录一次 width=32、55 epoch 标准协议的训练与 test，结果不代表其它宽度、
其它数据增强或其它训练预算。
