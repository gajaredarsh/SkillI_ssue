#!/usr/bin/env python3
"""
run.py -- SCUNet restoration inference entry point.

Reads every .npy file from an input directory, restores it with the
fine-tuned SCUNet model, and writes a 2D grayscale .npy file with the same
filename to an output directory. Runs fully offline on an NVIDIA GPU and
falls back to CPU automatically.

Pipeline (matches the training notebook exactly):
    load raw .npy  ->  replicate to 3 channels  ->  bicubic upsample x2
    ->  reflect-pad to a multiple of 64  ->  SCUNet  ->  crop back
    ->  average channels  ->  clip to the ground-truth range  ->  save

IMPORTANT -- no normalisation is applied. The model was fine-tuned on the
raw native value scale of the .npy files, so any percentile stretch or
[0,1] rescale here would put the input outside the training distribution
and cost several dB.

Usage:
    python run.py <input-dir> <output-dir>
    python run.py /data/Test_NoisyLR /data/results
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    from models.network_scunet import SCUNet
except ImportError as exc:                                   # pragma: no cover
    raise ImportError(
        "Could not import SCUNet from models/network_scunet.py. Make sure "
        "models/network_scunet.py is present next to run.py. See README.md."
    ) from exc


# --- model config: must match the fine-tuning notebook exactly ----------
SCUNET_IN_NC = 3
SCUNET_CONFIG = [4, 4, 4, 4, 4, 4, 4]
SCUNET_DIM = 64
PAD_MULTIPLE = 64          # 3 downsampling stages + window size 8

# Ground truth is per-image max-normalised to [0, 1]; clipping the output to
# that range can only reduce error. Pass --no_clip to disable.
GT_MIN, GT_MAX = 0.0, 1.0


def load_npy_image(path):
    """Load a .npy file as an HWC float32 array on its NATIVE value scale.

    Handles CHW -> HWC, drops an alpha channel, and replicates a single
    channel to 3 (SCUNet's colour model expects 3 input channels).
    Deliberately performs NO rescaling -- see the module docstring.
    """
    arr = np.asarray(np.load(path))

    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[-1]:
        arr = arr.transpose(1, 2, 0)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    arr = arr.astype(np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def upsample_to(img, target_h, target_w):
    """Bicubic-upsample an HWC float32 image. No clamp -- the raw scale is
    not bounded to [0, 1]."""
    if img.shape[:2] == (target_h, target_w):
        return img
    t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
    t = F.interpolate(t, size=(target_h, target_w), mode="bicubic",
                      align_corners=False)
    return t.squeeze(0).numpy().transpose(1, 2, 0)


def pad_to_multiple(img, multiple=PAD_MULTIPLE):
    h, w = img.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    return padded, h, w


@torch.no_grad()
def restore_image(model, device, noisy_img, scale):
    """noisy_img: HWC float32 at native LR resolution and native value scale.

    Returns an HWC float32 array at scale x the input resolution.
    """
    h, w = noisy_img.shape[:2]
    upsampled = upsample_to(noisy_img, h * scale, w * scale)

    padded, out_h, out_w = pad_to_multiple(upsampled)
    inp = torch.from_numpy(padded.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    out = model(inp).squeeze(0).cpu().numpy().transpose(1, 2, 0)
    out = out[:out_h, :out_w, :]
    return np.nan_to_num(out, nan=0.0, posinf=GT_MAX, neginf=GT_MIN)


def to_grayscale(img_hwc, clip=True):
    """HWC float32 -> (H, W) float32.

    The 3 input channels are a replicated grayscale image, not true RGB, so
    the channels are averaged rather than combined with luminance weights.
    """
    gray = img_hwc[:, :, 0] if img_hwc.shape[-1] == 1 else img_hwc.mean(axis=-1)
    gray = np.nan_to_num(gray, nan=0.0, posinf=GT_MAX, neginf=GT_MIN)
    if clip:
        gray = np.clip(gray, GT_MIN, GT_MAX)
    return gray.astype(np.float32)


def build_model(weights_path, device):
    model = SCUNet(in_nc=SCUNET_IN_NC, config=SCUNET_CONFIG, dim=SCUNET_DIM)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(
        description="SCUNet restoration inference",
        usage="python run.py <input-dir> <output-dir> [--weights PATH] [--scale N]")
    # Positional form required by the submission spec:
    #     python run.py <input-dir> <output-dir>
    # The optional flags below are accepted as an alternative but are never
    # required -- running with two bare paths is fully sufficient.
    parser.add_argument("input_dir", nargs="?", default=None,
                        help="Directory containing input .npy files")
    parser.add_argument("output_dir", nargs="?", default=None,
                        help="Directory for restored .npy files (created if missing)")
    parser.add_argument("--input_dir", dest="input_dir_flag", default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--output_dir", dest="output_dir_flag", default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--weights", default=os.path.join(
        SCRIPT_DIR, "models", "scunet_finetuned_multidegradation.pth"),
        help="Path to the fine-tuned checkpoint")
    parser.add_argument("--scale", type=int, default=2,
                        help="Upsampling factor from input to output resolution")
    parser.add_argument("--no_clip", action="store_true",
                        help="Do not clip the output to the [0,1] ground-truth range")
    args = parser.parse_args()

    input_dir = args.input_dir or args.input_dir_flag or os.path.join(SCRIPT_DIR, "input")
    output_dir = args.output_dir or args.output_dir_flag or os.path.join(SCRIPT_DIR, "output")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type != "cuda":
        print("WARNING: no CUDA GPU detected -- CPU inference will be slow.")

    if not os.path.isfile(args.weights):
        raise FileNotFoundError(
            f"Model weights not found at {args.weights}. Place the checkpoint "
            "there or pass --weights.")

    os.makedirs(output_dir, exist_ok=True)
    input_files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    if not input_files:
        raise FileNotFoundError(f"No .npy files found in {input_dir}")

    print(f"Loading model from {args.weights} ...")
    model = build_model(args.weights, device)
    print(f"Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params).")
    print(f"Running inference on {len(input_files)} file(s) at {args.scale}x ...")

    start, ok = time.time(), 0
    for i, path in enumerate(input_files, 1):
        fname = os.path.basename(path)
        try:
            noisy = load_npy_image(path)
            restored = restore_image(model, device, noisy, args.scale)
            gray = to_grayscale(restored, clip=not args.no_clip)

            assert gray.ndim == 2, f"expected 2D output, got {gray.shape}"
            assert np.isfinite(gray).all(), "non-finite values in output"

            np.save(os.path.join(output_dir, fname), gray)
            ok += 1
            if i % 50 == 0 or i == len(input_files):
                print(f"  [{i}/{len(input_files)}] {fname} -> {gray.shape}")
        except Exception as exc:                             # noqa: BLE001
            print(f"  [{i}/{len(input_files)}] {fname} FAILED: {exc}", file=sys.stderr)

    elapsed = time.time() - start
    print(f"Done. {ok}/{len(input_files)} restored in {elapsed:.1f}s "
          f"({1000*elapsed/max(len(input_files),1):.1f} ms/image) -> {output_dir}")


if __name__ == "__main__":
    main()
