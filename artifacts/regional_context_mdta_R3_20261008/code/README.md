# R3_bottom_interleaved：底部融合→MDTA→融合→MDTA→融合

以 V2_spatial_first 为基础。11 个自有主线小波融合块，零原始完整 StarBlock、零 StarModule、零 DFFN，保留逐点门控 FFN。底部：**fusion -> MDTA -> fusion -> MDTA -> fusion**。MDTA 数量：2。

三个版本训练协议完全一致：FP32（AMP/TF32 关闭），24/48/96 通道，128×128 裁剪，batch8，seed100，workers4，100000 步，AdamW，cosine 2e-4→2e-6；D4＋配对饱和度增强 p=0.25；GT-Mean L1（σ=0.1）＋0.1 complex FFT L1。按验证集 raw RGB L1 选 best_val.pt。普通 RGB L1 不加入训练 loss。

本目录可独立复制到服务器。激活安装了 PyTorch/CUDA 的环境后：

```bash
python -B verification/smoke.py --device cuda --batch-size 8
python -B training/train.py --print-command
python -B training/train.py --data-root /root/autodl-tmp/datasets/cholec80/train_test --run-dir outputs/seed100
```

续训和测试：

```bash
python -B training/train.py --data-root /root/autodl-tmp/datasets/cholec80/train_test --run-dir outputs/seed100 --resume outputs/seed100/last.pt
python -B evaluation/test_best.py --checkpoint outputs/seed100/best_val.pt --output-dir outputs/test_best
```

Windows 本地固定使用 `M:\Anaconda_envs\envs\retinexformer\python.exe`。三个版本分别有唯一模型标识，只接受各自完整训练权重；从头训练，不把旧 V2 权重作为本版正式续训。两个 MDTA 版本之间也不可混用续训权重。

模块改动见 [architecture_zh.md](docs/architecture_zh.md)，实现见 models/candidate_blocks.py，底部执行顺序见 models/core/model.py。verification/reports 保存实现检查结果，非正式训练指标。正式训练由用户在服务器启动，参数量暂不核算。
