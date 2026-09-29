# Four-map brightness guidance at five sites, 2026-09-29

Build on the bottom-only experiment without changing its maps, projections,
attention, FFN, residual scale, or common seeded initialization. All weights
train from scratch. Original MDTA remains after the bottom guidance block.

Guidance sites, after each existing local block:
- Encoder0 full resolution, 24 channels, one attention head.
- Encoder1 half resolution, 48 channels, two heads.
- Bottom after Spatial Fusion and before MDTA, 96 channels, four heads.
- Decoder1 half resolution, 48 channels, two heads.
- Decoder0 full resolution, 24 channels, one head, before the RGB head.

Four maps are computed once from augmented low RGB; formulas and provenance
remain in BRIGHTNESS_CROSS_SOURCE.md. The existing bottom encoder generates
16-channel half features and 96-channel quarter features. New 4->16->24 full
projection and 16->48 half projection provide matching priors. Encoder and
decoder reuse the same full/half priors, with independent attention/FFN weights.
Each site uses main Q, prior K/V, channel cross-attention, FFN expansion2 and
learned residual scales initialized to0.1. This is a five-site adaptation of
SG-LLIE cross-attention, not a reproduction of its whole HSGFE or spatial attention.

The five-site module preserves parameter names and inference of old bottom-only
checkpoints. New full/half components are initialized after A+MDTA and the
bottom branch. The numerical source of the four maps is unchanged; no GT
is fed into the prior encoder. All five sites receive gradients only through
the same final-output GT-Mean loss.

Parameters: base245719; bottom-only332939; multi-scale388473.
Added55534 vs bottom-only, or142754 vs base. Five attention matrices each
operate on24 channels per head; no HWxHW attention matrix is introduced.

Same recipe: 55epochs/10340steps, batch8, seed100, crop192x384, H/V flips each0.5,
no D4; AdamW lr2e-4 cosine to2e-6, weight_decay1e-4, clip1, AMP, workers0,
GT-Mean L1 sigma0.1. Save and validate every500 steps; retain all periodic
weights and final10340. Wrapper evaluates raw-val-L1 best once automatically.

Comparison tests the whole additional four-site package, including its new
attention/FFN capacity. Attribution specifically to prior information would
require an additional control; this experiment only expands the accepted scheme.
