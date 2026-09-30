# StarIR / Cholec80：停止租卡前保存的 41,000 步 checkpoint

本次训练由用户要求停止，目标 100,000 步尚未完成。当前保存边界为 41,000 次 optimizer 更新。

- net_g_41000.pth：官方格式 params 模型权重，509,072 参数。
- 41000.state：同一步的 optimizer / scheduler 状态，可继续训练。
- config.yml：实际运行配置；code/：配套 StarIR 网络和单卡训练器。
- stop_report.json：停止记录和已完成的验证记录；train.log：原训练日志。

训练设置：单卡 batch 16，seed 42，128×128 成对随机裁剪；
官方 D4，饱和度独立概率 1/3、因子 0.8–1.2，LQ / GT 共用参数；
gamma 分支的 gamma=1。颜色增强仅用于训练，验证使用原图。

损失 L1 + 0.1 FFTLoss；FP32；AdamW lr 1e-3、weight_decay 1e-3、betas (0.9,0.9)；
梯度范数裁剪 0.01。余弦 T_max=100000、eta_min=1e-7，无 warmup。
每 1000 步保存和验证。源代码版本：fc5f3681f16e600a5777dcc284e2b7040fdb0caa。

41,000 步 checkpoint 已保存，但该步的验证被停止请求打断。
日志中最后完成的是 40,000 步验证；验证 PSNR 最高的已完成记录在 28,000 步。
本目录保留的是用户要求的当前 41,000 步权重，没有对它重新跑测试集。

CPU strict=True 加载通过，文件 SHA-256 与服务器导出一致。
恢复训练时，将模型和状态放回 code/experiments/Enhancement_StarIR_cholec80_single_b16_100k_seed42/
下的 models/ 和 training_states/，设置数据路径后运行配套 train_single_gpu.py。
训练器可自动读取最近的状态；这是官方恢复机制，不承诺恢复后的逐位确定性。
