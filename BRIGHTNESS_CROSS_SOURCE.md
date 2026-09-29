# Four-map brightness cross-attention, 2026-09-29

Intervention: A encoder -> original GatedConvBlock + Spatial Fusion -> new
brightness cross-attention + light FFN -> unchanged MDTA -> original decoder.
One guidance site, one-stage output, all parameters train from scratch.

Prior input is augmented low RGB only. Y=0.299R+0.587G+0.114B.
1. ((Y+0.02)^0.4 - 0.02^0.4)/(1.02^0.4 - 0.02^0.4).
2. log(1+20*Gaussian_sigma5(Y))/log(21); separable support +/-15, reflect padding.
3. sin(pi*Y/2)^0.2.
4. log(1+20Y)/log(21).

Fixed [0,1]; no per-image min/max. Maps run FP32, without gradients. The
Gaussian uses replicate padding only when a dimension is too small for reflect.
These are brightness descriptions, not GT illumination or correction targets.
The first two transforms are our adaptations; the nonlinear and log transforms
follow BIP-CENet (https://github.com/DeepUG-AI/BIP-CENet/blob/main/net/Mynet.py),
with its per-image min/max deliberately omitted.

Shared prior encoder: stride2 3x3 Conv4->16, GELU, stride2 3x3 Conv16->32,
GELU, 1x1 Conv32->96. It matches the quarter-resolution main feature.
Cross-attention uses main Q and prior K,V; four heads; each matrix is 24x24.
1x1 + depthwise3x3 projections and learned temperature follow SG-LLIE:
https://github.com/minyan8/imagine/blob/main/basicsr/models/archs/RetinexFormer_arch.py
This is channel cross-attention. We adopt only cross-attention + FFN, not the
full SGTB/HSGFE; no extra CSA, DRDB or SAM. FFN uses expansion2, GELU and DWConv.
Both residual branches have learned per-channel scales initialized to 0.1.
Q/K normalization, attention scores and softmax run FP32; other new layers use AMP.

The added module is initialized after all shared A+MDTA parameters to retain
their seed100 values. Base 245719; added 87220; total 332939 trainable parameters.
Training: batch8, seed100, 55 epochs/10340 steps, paired crop192x384,
independent H/V flips p=0.5, no D4; AdamW lr2e-4 cosine to2e-6, weight_decay1e-4,
clip1, AMP, workers0, existing GT-Mean L1 sigma0.1. GT is used only by the loss.
Save every500 steps and final10340; retain all periodic weights. Existing raw
val L1 best selection and evaluator are preserved; wrapper evaluates this best
once automatically. Periodic weights remain available for later comparison.
