# Final submission checklist

## Repository layout

```
<TEAM_NAME>/                      # repo root, public on GitHub
├── run.py                        # THE benchmarked file — positional CLI
├── requirements.txt              # real `pip freeze` from the training venv
├── README.md
├── models/
│   ├── __init__.py               # empty file — REQUIRED for the import to work
│   ├── network_scunet.py         # copied from SCUNet/models/network_scunet.py
│   └── scunet_finetuned_multidegradation.pth      # 71.9 MB
├── training/
│   └── train_scunet.ipynb        # the v4 notebook, renamed
└── restored_test_outputs/        # 400 .npy files from the test set
```

## Before pushing

- [ ] `models/__init__.py` exists and is empty. Without it
      `from models.network_scunet import SCUNet` fails.
- [ ] `pip freeze > requirements.txt` run INSIDE the training venv.
      The spec asks for the complete freeze, not a hand-written list.
- [ ] Weights are 71.9 MB — under GitHub's 100 MB hard cap but over the
      50 MB warning. Plain `git push` works; use Git LFS if it complains:
          git lfs install
          git lfs track "*.pth"
          git add .gitattributes
- [ ] Repository visibility set to PUBLIC.
- [ ] Notebook renamed to training/train_scunet.ipynb and its first cell
      states the expected dataset paths.

## Fresh-machine test (do this — an unrunnable script cannot be scored)

```bash
cd /tmp && rm -rf verify
git clone <REPO_URL> verify && cd verify
python -m venv .v && source .v/bin/activate
pip install -r requirements.txt
python run.py /path/to/Test_NoisyLR/NoisyLR /tmp/out
```

Then validate the outputs:

```python
import numpy as np, glob
f = sorted(glob.glob('/tmp/out/*.npy'))
a = np.stack([np.load(p) for p in f])
assert len(f) == 400,                       f"expected 400 files, got {len(f)}"
assert a.shape[1:] == (256, 256),           f"wrong shape {a.shape[1:]}"
assert a.dtype == np.float32,               f"wrong dtype {a.dtype}"
assert np.isfinite(a).all(),                "NaN or Inf present"
assert a.min() >= 0.0 and a.max() <= 1.0,   f"out of range [{a.min()}, {a.max()}]"
src = sorted(glob.glob('/path/to/Test_NoisyLR/NoisyLR/*.npy'))
assert [p.split('/')[-1] for p in f] == [p.split('/')[-1] for p in src], "filenames differ"
print("ALL CHECKS PASS")
```

## Deck (separate upload)

- PDF, named `TeamName_KLA_PS01.pdf`
- 8–9 slides maximum, instruction slide removed

Suggested slide order:
1. Problem + measured degradation (speckle + additive, values outside [0,1])
2. Results table vs bicubic: 22.99 -> 29.20 dB, 0.559 -> 0.804 SSIM
3. Visual comparisons — input / bicubic / restored / GT, with one zoomed crop
4. Method: why SCUNet (its pretraining models speckle as multiplicative)
5. Loss design: L1 + SSIM + Sobel + Laplacian + contrast, and why
6. Error decomposition: perfect denoise + bicubic = 31.67 dB, so denoising
   is the bottleneck, not upsampling
7. Ablations, including the negative results
8. Efficiency: parameters, ms/image on GPU
9. Limitations and future work
