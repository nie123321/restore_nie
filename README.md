# Brightness-Multiscale-A：四图亮度先验 + 五站点交叉注意力

本分支在 A（width 24、`spectral_mode=off`、`output_mode=direct`）+ 底部 MDTA + GT-Mean L1
的基础上，加入四张固定亮度图作为先验，并在 **五个站点**接入"主特征 Q / 共享先验 K,V"的
通道交叉注意力（+2×FFN，残差缩放初始 0.1）：

- 编码器两处（全分辨率、半分辨率）；
- 底部原 `bottom-cross4` 站点（Spatial Fusion 之后、MDTA 之前）；
- 解码器两处（半分辨率、全分辨率）。

先验来自 low RGB 的四张固定 `[0,1]` 变换（不做逐图归一化）：
`power(γ0.4,ε0.02)`、`gaussian5+log20`、`sine^0.2`、`log20`；编码器 4→16→32→96，
共享先验金字塔（新增 full 4/16/24 与 half 16/48 投影）。固定图、Q/K 归一化与注意力分数、
softmax 均在 FP32；主干/投影/FFN 走 AMP。参考实现：SG-LLIE / RetinexFormer 的 Cross_attention
（`BRIGHTNESS_CROSS_SOURCE.md`、`BRIGHTNESS_MULTISCALE_SOURCE.md`）。

参数量 **388,473**（比基础版 +142,754；比底部单站点版 +55,534）。

## 训练协议

与 55 epoch 标准一致：整图验证、训练用 paired random crop `192×384` + 水平/垂直翻转各 0.5；
batch 8、seed 100、AdamW、weight decay 1e-4、梯度裁剪 1、CUDA AMP；
cosine 2e-4 → 2e-6、55 epoch / 10,340 步；每 500 步验证与保存；
按最低验证 raw L1 选择 `best_val.pt`；test 用 300 张 clamp+round PNG 四指标。

## 结果

**待补**：本分支在上传时训练仍在进行（test 尚未执行）。完成后会在本文件与
`result_brightness_multiscale.json` 中更新 val best 与 test 四指标。

## 文件

- `model.py`、`run_demo.py`、`test_best.py`：训练/测试快照（与 run 目录 `code/` 一致）。
- `brightness_cross_blocks.py`、`brightness_multiscale_blocks.py`：底部与多尺度亮度交叉注意力。
- `BRIGHTNESS_CROSS_SOURCE.md`、`BRIGHTNESS_MULTISCALE_SOURCE.md`：来源与边界说明。
- `mdta_blocks.py`、`gt_mean_loss.py`：MDTA 与 GT-Mean L1。
- `train_brightness_multiscale.py`（本次启动脚本）、`train_brightness_cross.py`、`train_gt_mean.py`、`train_mdta.py`。
- `fremlp_blocks.py`、`illumination_blocks.py`、`lfpv_blocks.py`、`restormer_blocks.py`、`star_blocks.py`、`window_attention_blocks.py`：`model.py` 的可选依赖。
- `evaluate_four_metrics.py`：四指标评测脚本快照。
- `result_brightness_multiscale.json`：配置与状态摘要。
- 未上传 `runs/`：权重、日志与逐张指标保留在本地。

## 运行

```powershell
$py = 'M:\Anaconda_envs\envs\retinexformer\python.exe'
& $py -u -X utf8 run_demo.py train --run-dir runs/a_mdta_brightness_cross4_multiscale_gtmean_crop192x384_hv_b8_e55_seed100_20260929 `
    --epochs 55 --batch-size 8 --seed 100 --lr 2e-4 --lr-schedule cosine --min-lr 2e-6 `
    --weight-decay 1e-4 --grad-clip 1.0 --best-metric l1 --val-every 500 --save-every 500 `
    --width 24 --spectral-mode off --output-mode direct --bottleneck-attention mdta `
    --brightness-prior-attention multiscale-cross4 --loss-mode gt-mean-l1 --gt-mean-sigma 0.1 `
    --crop-size 192 384 --flip-hv --device cuda --amp --log-every 50
```

边界：单 seed、固定 55 epoch 预算；val 与 test 口径不同（val 为整图 float L1/PSNR，test 为
clamp+round PNG 四指标）。本目录不代表多种子平均结果。
