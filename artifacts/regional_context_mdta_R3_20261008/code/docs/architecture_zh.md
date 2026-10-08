# 底部融合→MDTA→融合→MDTA→融合

共同基准是 V2 的空间优先联合融合：

`S = X + Rs(X)`；`T = X + gamma1 * (W(S) - X)`；`Y = T + gamma2 * FFN(Norm(T))`。

W 是两级正交 Haar 小波分解、子带处理与逆变换。Rs 是原有局部空间门控。gamma1/gamma2 是逐通道可学习系数，初始为 0.1；不存在独立 frequency-only 缩放。FFN 保留原有逐点扩展、相乘和投影。

网络五个阶段融合深度为 2/2/3/2/2，共 11 个独立融合块；U 型下采样、跳连、解码器和 RGB 输出头沿用 V2。源码保留历史 core 兼容类，但实例化网络没有原始 StarBlock、StarModule 或 DFFN。

## 本版唯一改动

底部融合→MDTA→融合→MDTA→融合。

底部执行顺序：`fusion -> MDTA -> fusion -> MDTA -> fusion`。完整可核对的模块调用顺序：

```text
encoder0
star_encoder0
encoder1
star_encoder1
encoder2
mdta
spectral.block.0
mdta_second
spectral.block.1
decoder1
star_decoder1
decoder0
star_decoder0
```

LL₂ 的处理：LL2 only: concat channel-normalized LL2 with raw per-channel spatial mean and sqrt(var+1e-6); PW(3C,2C) -> DW5(2C) -> GELU -> PW(2C,2C); gain=exp(tanh(log_gain)); LL2_new=gain*LL2+bias。

两个尺度的高频仍是三个独立的有符号 `H_i + DW3_i(H_i)` 更新；较细尺度的 LL₁ 直接使用粗尺度逆变换结果，不新增第二个低频预测头。

MDTA：沿用原有子带 MDTA 的内部实现，保留第一个模块的基准初始化，独立初始化第二个模块；R2/R3 参数初值一致。LL 生成的注意力混合通道，再作用于 LL 与三个高频子带，逆小波变换回主线。该 MDTA 不是空间区域注意力。

所有基准 V2 参数在构造后被保留；只在最后初始化新增层。R2/R3 的参数初值完全相同，仅执行顺序不同。
