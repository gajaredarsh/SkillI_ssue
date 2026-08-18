# KLA PS01 — Joint Denoising and 2× Super-Resolution

**Team:** `<TEAM_NAME>`

Restores grayscale `.npy` images degraded by combined multiplicative
speckle, additive Gaussian noise, and 2× downsampling.

| Metric | Bicubic baseline | Ours |
|---|---|---|
| PSNR (dB) | 22.99 | **29.20** |
| SSIM | 0.559 | **0.804** |
| LPIPS ↓ | 0.425 | *see report* |

Measured on 200 held-out training pairs (indices 0–199), never used for
fitting.

---

## Quick start

```bash
git clone <REPO_URL>
cd <REPO_NAME>
pip install -r requirements.txt
python run.py <input-dir> <output-dir>
```

Example:

```bash
python run.py /data/Test_NoisyLR /data/restored
```

No internet access, API keys, extra downloads, or manual configuration are
required at run time. The model weights are included in `models/`.

---

## Repository layout

```
.
├── run.py                       # inference entry point (the benchmarked script)
├── requirements.txt             # pip freeze from the training environment
├── README.md
├── models/
│   ├── __init__.py
│   ├── network_scunet.py        # SCUNet architecture (cszn/SCUNet)
│   └── scunet_finetuned_multidegradation.pth   # final weights, 17.95M params
├── training/
│   └── train_scunet.ipynb       # reproduces training from scratch
└── restored_test_outputs/       # model outputs on the provided test set (400 .npy)
```

---

## Method

**Backbone.** SCUNet (Swin-Conv-UNet), pretrained by Zhang et al. Chosen
because its training-data synthesis explicitly models **speckle as
multiplicative noise** — the dominant degradation in this task — unlike
denoisers that assume additive Gaussian noise only. Its swin-transformer
branch provides non-local modelling, the learned analogue of the
self-similarity prior that BM3D exploits.

**Fine-tuning.** The pretrained denoiser was fine-tuned end-to-end on the
provided paired data with a combined objective:

```
L = 1.0·L1 + 0.3·(1 − SSIM) + 0.35·Sobel + 0.15·Laplacian + 0.2·contrast
```

The Sobel and Laplacian terms target edge and fine-texture fidelity; the
contrast term penalises the dulling that a pure L1 objective produces. A
plain L1 objective converges to the posterior mean, which is provably
smooth wherever the degraded input does not determine high-frequency phase
— visible as washed-out texture. These terms counteract that directly.

**Inference pipeline.** Bicubic upsample ×2 → reflect-pad to a multiple of
64 → SCUNet → crop → average channels → clip to [0, 1].

---

## Design notes

**No input normalisation.** The model is fine-tuned on the raw native value
scale of the `.npy` files. The degraded inputs are *not* bounded to [0, 1]
— multiplicative speckle pushes bright pixels above 1, and the additive
component pushes dark pixels slightly below 0 — while the ground truth is
per-image max-normalised to [0, 1]. An earlier version applied a per-image
1st/99th-percentile contrast stretch; removing it improved PSNR by roughly
4 dB, because the stretch placed inputs outside the training distribution.

**Channel handling.** SCUNet's colour model expects 3 input channels, so
the single grayscale plane is replicated on input and the 3 output channels
are **averaged** on output. ITU-R luminance weights would be inappropriate
here — the channels are copies of one plane, not true RGB.

**Resolution.** The model is fully convolutional, so any input resolution
is accepted at a fixed 2× scale. The provided test set is 400 images at
128×128 → 256×256. A 256×256 input yields 512×512 unchanged.

---

## Command-line reference

```
python run.py <input-dir> <output-dir> [options]
```

| Argument       | Default                                              | Description                                       |
|----------------|------------------------------------------------------|---------------------------------------------------|
| `input-dir`    | *(required)*                                         | Directory of input `.npy` files                   |
| `output-dir`   | *(required)*                                         | Output directory, created if missing              |
| `--weights`    | `models/scunet_finetuned_multidegradation.pth`       | Path to the checkpoint                            |
| `--scale`      | `2`                                                  | Input→output upsampling factor                    |
| `--no_clip`    | off                                                  | Skip clipping the output to [0, 1]                |

### Output guarantees

- one `.npy` per input, **same filename**
- shape `(H·2, W·2)`, dtype `float32`
- values within `[0, 1]`, no NaN or Inf
- a file that fails is logged to stderr and skipped, so one bad input
  cannot abort the batch

---

## Reproducing training

```bash
jupyter notebook training/train_scunet.ipynb
```

Expects the dataset at `train/NoisyLR`, `train/GT`, and `NoisyLR` (test).
The notebook downloads the SCUNet pretrained weights, fine-tunes for 20
epochs, and writes `scunet_finetuned_multidegradation.pth`.

Hardware used: single NVIDIA T4. Runtime: approximately 1 hour.

---

## Environment

```bash
pip install -r requirements.txt
```

Requires a CUDA-enabled PyTorch build. `run.py` falls back to CPU
automatically, but CPU inference is roughly 60× slower.
