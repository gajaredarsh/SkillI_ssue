# SCUNet Fine-tuned Image Restoration

Fine-tuned [SCUNet](https://github.com/cszn/SCUNet) (Swin-Conv-UNet) blind-denoising model,
adapted from the pretrained `scunet_color_real_psnr` checkpoint to restore images degraded
by a combination of Gaussian noise, speckle noise, and downscale/upscale blur.

## Repository contents

| Path | Description |
|---|---|
| `README.md` | This file. |
| `evaluate.py` | Standalone evaluation/inference script (see below). |
| `training/train_scunet.ipynb` | Notebook reproducing the full training process from scratch. |
| `weights/scunet_finetuned.pth` | Final trained model checkpoint. |
| `outputs/` | Restored images produced by `evaluate.py` on the test set. |
| `models/network_scunet.py` | Vendored SCUNet architecture (see "Architecture source" below). |
| `requirements.txt` | Python package requirements. |

## Setup

```bash
git clone <this-repo-url>
cd <this-repo>
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

Requires Python 3.10+ and a CUDA-capable GPU for reasonable inference speed (CPU also
works, just slower).

## Running inference (evaluation script)

```bash
python evaluate.py --input_dir /path/to/test/NoisyLR --output_dir /path/to/output_dir
```

This loads `weights/scunet_finetuned.pth`, runs restoration on every `.npy` file in
`--input_dir`, and writes the restored `.npy` image (same filename) to `--output_dir`.
No manual edits are required -- both paths are supplied entirely via command-line
arguments.

**Optional flags:**

| Flag | Default | Purpose |
|---|---|---|
| `--weights` | `weights/scunet_finetuned.pth` | Path to the trained checkpoint. |
| `--upsample_scale` | `1` | Set to the LR:GT resolution ratio used during training (e.g. `2`) if your test noisy images are lower resolution than the ground truth the model was trained against. Leave at `1` if noisy/GT were the same resolution. |
| `--device` | `cuda` if available, else `cpu` | Inference device. |

## Reproducing training from scratch

Open `training/train_scunet.ipynb` (Jupyter locally, or upload to Kaggle -- the
notebook's markdown includes Kaggle-specific setup notes) and run all cells top to
bottom. It expects a dataset with this layout:

```
train/NoisyLR/   noisy training images (.npy)
train/GT/        matching ground-truth images (.npy), same filenames
NoisyLR/         test set: noisy images only (.npy)
```

Update the dataset paths in the notebook's `CFG` class if your data lives elsewhere.
Training fine-tunes the pretrained SCUNet checkpoint (auto-downloaded from KAIR's
GitHub release on first run) using a combined L1 + SSIM + Sobel/Laplacian-sharpening +
contrast loss, and saves the best checkpoint (by validation PSNR) to disk.

## Architecture source

`evaluate.py` and the training notebook both need the `SCUNet` class definition.
**`models/network_scunet.py` is vendored directly in this repo** (from the official
[cszn/SCUNet](https://github.com/cszn/SCUNet)), so evaluation runs fully offline --
no network access needed on the benchmarking machine. `evaluate.py` only falls back
to cloning the repo at runtime if this file is ever removed.

## Model weights

`weights/scunet_finetuned.pth` must be present for `evaluate.py` to run. If the file
is large, use Git LFS (`git lfs track "weights/*.pth"`) or host it externally and
document the download command here.

## Notes

- Pretrained base weights (`scunet_color_real_psnr.pth`) are downloaded automatically
  during training from `https://github.com/cszn/KAIR/releases/download/v1.0/` --
  training requires internet access on first run.
- `requirements.txt` lists the packages actually imported by this repo's code. Before
  final submission, consider regenerating it via `pip freeze > requirements.txt`
  inside your actual training environment for exact reproducibility.
