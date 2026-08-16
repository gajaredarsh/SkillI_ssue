#!/usr/bin/env python3
"""
Standalone evaluation / inference script for the fine-tuned SCUNet restoration model.

Loads the SCUNet architecture and trained weights, runs inference on every .npy image
found under --input_dir, and writes the restored .npy image to --output_dir under the
same filename. Runs with no manual edits required.

Usage:
    python evaluate.py --input_dir /path/to/test/NoisyLR --output_dir /path/to/restored_outputs

Optional flags:
    --weights          Path to the trained checkpoint (.pth). Default: weights/scunet_finetuned.pth
    --upsample_scale   Set to the LR:GT scale factor used during training if your noisy test
                        images are lower resolution than the ground truth (e.g. 2). Default: 1
                        (no upsampling -- noisy and GT were the same resolution during training).
    --device           'cuda' or 'cpu'. Default: cuda if available, else cpu.
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_FILE = os.path.join(REPO_ROOT, "models", "network_scunet.py")
FALLBACK_CLONE_DIR = os.path.join(REPO_ROOT, "_scunet_src")


def _ensure_scunet_importable():
    """Prefer a vendored local copy of models/network_scunet.py (fully offline-safe,
    zero network dependency). If it isn't present in this repo, fall back to cloning
    the official cszn/SCUNet repository at runtime (requires internet access on the
    machine running this script)."""
    if os.path.isfile(LOCAL_MODEL_FILE):
        sys.path.insert(0, REPO_ROOT)
        return

    if not os.path.isdir(FALLBACK_CLONE_DIR):
        print(
            "models/network_scunet.py not found in this repo -- cloning cszn/SCUNet "
            "as a fallback (requires internet access)...",
            flush=True,
        )
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/cszn/SCUNet.git", FALLBACK_CLONE_DIR],
            check=True,
        )
    sys.path.insert(0, FALLBACK_CLONE_DIR)


_ensure_scunet_importable()
from models.network_scunet import SCUNet  # noqa: E402  (import must follow path setup above)


def load_npy_image(path, lo_pct=1.0, hi_pct=99.0):
    """Normalizes a .npy image to float32 HWC in [0, 1] via a per-image percentile
    contrast stretch. Must match the normalization used during training (see
    training/train_scunet.ipynb, section 4) so the model sees the same input
    distribution it was fine-tuned on."""
    arr = np.load(path)
    arr = np.asarray(arr)

    # CHW -> HWC if the first dim looks like a small channel count
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[-1]:
        arr = arr.transpose(1, 2, 0)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, lo_pct), np.percentile(arr, hi_pct)
    if hi - lo < 1e-8:
        lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    arr = (arr - lo) / (hi - lo)
    return np.clip(arr, 0.0, 1.0)


def pad_to_multiple(img, multiple=64):
    """SCUNet has 3 downsampling stages with window-based attention -- input H/W must
    be a multiple of 64. Reflect-pads up to the next multiple, returns original size
    so the output can be cropped back."""
    h, w = img.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    return padded, h, w


@torch.no_grad()
def restore_image(model, img, device, upsample_scale=1):
    """img: HWC float32 in [0,1]. Optionally bicubic-upsamples by `upsample_scale`
    (use this if your noisy test images are lower resolution than the GT the model
    was trained against), pads to a multiple of 64, runs SCUNet, crops back."""
    if upsample_scale > 1:
        h, w = img.shape[:2]
        t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
        t = F.interpolate(t, size=(h * upsample_scale, w * upsample_scale), mode="bicubic", align_corners=False)
        img = t.squeeze(0).clamp(0, 1).numpy().transpose(1, 2, 0)

    padded, out_h, out_w = pad_to_multiple(img)
    inp = torch.from_numpy(padded.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    out = model(inp).clamp(0, 1).squeeze(0).cpu().numpy().transpose(1, 2, 0)
    return out[:out_h, :out_w, :]


def main():
    parser = argparse.ArgumentParser(description="Run SCUNet restoration on a directory of .npy test images.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of noisy .npy test images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write restored .npy images.")
    parser.add_argument(
        "--weights",
        type=str,
        default=os.path.join(REPO_ROOT, "weights", "scunet_finetuned.pth"),
        help="Path to the trained model weights (.pth).",
    )
    parser.add_argument(
        "--upsample_scale",
        type=int,
        default=1,
        help="LR:GT scale factor used during training, if noisy test images are lower "
        "resolution than GT (e.g. 2). Default 1 = no upsampling.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.isfile(args.weights):
        print(f"ERROR: weights file not found at {args.weights}", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device)
    print(f"Loading SCUNet and weights from {args.weights} onto {device} ...")
    model = SCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
    state_dict = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    test_files = sorted(glob.glob(os.path.join(args.input_dir, "*.npy")))
    if not test_files:
        print(f"No .npy files found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Running inference on {len(test_files)} image(s) from {args.input_dir} ...")
    for i, path in enumerate(test_files, 1):
        img = load_npy_image(path)
        restored = restore_image(model, img, device, upsample_scale=args.upsample_scale)
        out_path = os.path.join(args.output_dir, os.path.basename(path))
        np.save(out_path, restored.astype(np.float32))
        if i % 10 == 0 or i == len(test_files):
            print(f"  [{i}/{len(test_files)}] saved {out_path}")

    print(f"Done. Restored {len(test_files)} image(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
