#!/usr/bin/env python3
"""
run.py -- SCUNet restoration inference entry point.

Reads every .npy file from an input directory, restores it with a
fine-tuned SCUNet model, and writes a grayscale .npy file with the same
filename to an output directory. Designed to run fully offline on an
NVIDIA GPU (falls back to CPU automatically if none is available).

Usage:
    python run.py --input_dir /path/to/NoisyLR --output_dir /path/to/results
    python run.py --input_dir /path/to/NoisyLR --output_dir /path/to/results \
                   --weights weights/scunet_finetuned_multidegradation.pth --scale 2

All arguments have defaults matching the expected submission layout, so
`python run.py` with no arguments also works if you keep the default
input/ and output/ folder names next to this script.
"""

import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# --- locate the bundled SCUNet architecture code -----------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    from models.network_scunet import SCUNet
except ImportError as e:
    raise ImportError(
        "Could not import SCUNet from models/network_scunet.py. "
        "Make sure models/network_scunet.py (from the cszn/SCUNet repo) "
        "is present next to run.py. See README.md."
    ) from e


# --- model config (must match training) ---------------------------------
SCUNET_IN_NC = 3
SCUNET_CONFIG = [4, 4, 4, 4, 4, 4, 4]
SCUNET_DIM = 64
PAD_MULTIPLE = 64  # SCUNet has 3 downsampling stages + window size 8


def load_npy_image(path):
    """Load a .npy file and normalize it to an HWC float32 array in [0, 1].

    Handles CHW->HWC, drops an alpha channel, replicates grayscale to 3
    channels (SCUNet's color model expects 3 input channels), and rescales
    arbitrary-range data to [0, 1] using a robust 1st/99th percentile
    stretch (falls back to true min/max, then to zeros for a constant
    image) so images stored in any native range/dtype are handled safely.
    """
    arr = np.load(path)
    arr = np.asarray(arr)

    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[-1]:
        arr = arr.transpose(1, 2, 0)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    arr = arr.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    if hi - lo < 1e-8:
        lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def upsample_to(img, target_h, target_w):
    """Bicubic-upsample an HWC float32 [0,1] image to (target_h, target_w)."""
    if img.shape[:2] == (target_h, target_w):
        return img
    t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
    t = F.interpolate(t, size=(target_h, target_w), mode="bicubic", align_corners=False)
    out = t.squeeze(0).numpy().transpose(1, 2, 0)
    return np.clip(out, 0.0, 1.0)


def pad_to_multiple(img, multiple=PAD_MULTIPLE):
    h, w = img.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    return padded, h, w


@torch.no_grad()
def restore_image(model, device, noisy_img, scale):
    """noisy_img: HWC float32 in [0,1] at native LR resolution.
    Upsamples by `scale`, pads to a multiple of 64, runs SCUNet, crops back.
    Returns an HWC float32 array clipped to [0,1].
    """
    h, w = noisy_img.shape[:2]
    target_h, target_w = h * scale, w * scale
    upsampled = upsample_to(noisy_img, target_h, target_w)

    padded, out_h, out_w = pad_to_multiple(upsampled)
    inp = torch.from_numpy(padded.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    out = model(inp).squeeze(0).cpu().numpy().transpose(1, 2, 0)
    out = out[:out_h, :out_w, :]
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(out, 0.0, 1.0)


def to_grayscale(img_hwc):
    """HWC float32 [0,1] (any channel count) -> (H, W) float32 [0,1] grayscale."""
    if img_hwc.shape[-1] == 1:
        gray = img_hwc[:, :, 0]
    else:
        # standard luminance weighting; falls back to plain mean for non-RGB
        if img_hwc.shape[-1] == 3:
            weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
            gray = (img_hwc * weights).sum(axis=-1)
        else:
            gray = img_hwc.mean(axis=-1)
    gray = np.nan_to_num(gray, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(gray, 0.0, 1.0).astype(np.float32)


def build_model(weights_path, device):
    model = SCUNet(in_nc=SCUNET_IN_NC, config=SCUNET_CONFIG, dim=SCUNET_DIM)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="SCUNet restoration inference")
    parser.add_argument("--input_dir", default=os.path.join(SCRIPT_DIR, "input"),
                         help="Directory containing input .npy files")
    parser.add_argument("--output_dir", default=os.path.join(SCRIPT_DIR, "output"),
                         help="Directory to write restored .npy files")
    parser.add_argument("--weights", default=os.path.join(
        SCRIPT_DIR, "weights", "scunet_finetuned_multidegradation.pth"),
        help="Path to the fine-tuned SCUNet checkpoint (.pth)")
    parser.add_argument("--scale", type=int, default=2,
                         help="Upsampling factor from input resolution to target "
                              "output resolution (2x used during fine-tuning)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type != "cuda":
        print("WARNING: no CUDA GPU detected -- running on CPU will be slow.")

    if not os.path.isfile(args.weights):
        raise FileNotFoundError(
            f"Model weights not found at {args.weights}. "
            "Place your fine-tuned checkpoint there or pass --weights."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    # Additional viewable-image folder, kept next to run.py.
    output_png_dir = os.path.join(SCRIPT_DIR, "output_png")
    os.makedirs(output_png_dir, exist_ok=True)

    input_files = sorted(glob.glob(os.path.join(args.input_dir, "*.npy")))
    if not input_files:
        raise FileNotFoundError(f"No .npy files found in {args.input_dir}")

    print(f"Loading model from {args.weights} ...")
    model = build_model(args.weights, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded ({n_params / 1e6:.2f}M params).")

    print(f"Running inference on {len(input_files)} file(s), scale={args.scale}x ...")
    for i, path in enumerate(input_files, 1):
        fname = os.path.basename(path)
        try:
            noisy = load_npy_image(path)
            restored = restore_image(model, device, noisy, args.scale)
            gray = to_grayscale(restored)

            assert np.isfinite(gray).all(), f"Non-finite values in output for {fname}"
            assert gray.min() >= 0.0 and gray.max() <= 1.0, f"Output out of [0,1] for {fname}"

            # Keep the existing .npy output unchanged.
            np.save(os.path.join(args.output_dir, fname), gray)

            # Additionally save the same restored grayscale image as PNG.
            png_name = os.path.splitext(fname)[0] + ".png"
            png_path = os.path.join(output_png_dir, png_name)
            Image.fromarray(
                (gray * 255.0).round().astype(np.uint8),
                mode="L"
            ).save(png_path)

            print(f"[{i}/{len(input_files)}] {fname} -> {gray.shape} OK")
        except Exception as exc:
            print(f"[{i}/{len(input_files)}] {fname} FAILED: {exc}", file=sys.stderr)

    print(f"Done. Restored files written to {args.output_dir}")


if __name__ == "__main__":
    main()
