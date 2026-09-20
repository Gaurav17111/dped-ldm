"""Metrics: PSNR, SSIM (Y-channel) and optional LPIPS on val crop pairs.

Usage:
  python src/metrics.py --gt data/processed/iphone/crops/val/dslr --pred runs/infer
  python src/metrics.py --gt ... --pred ... --lpips   (adds LPIPS, downloads VGG weights)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import list_images, load_rgb


# --------------------------------------------------------------------------- numpy metrics
def to_y(img: np.ndarray) -> np.ndarray:
    """RGB float [0,1] -> BT.601 Y channel."""
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 100.0 if mse == 0 else float(10.0 * np.log10(1.0 / mse))


def _ssim_window(w: np.ndarray, sigma: float = 1.5):
    size = int(3 * sigma) * 2 + 1
    ax = np.arange(size) - size // 2
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return np.outer(g, g), size


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Single-channel SSIM (skimage-style, gaussian window, no downsampling)."""
    w, size = _ssim_window(np.zeros((11, 11)))
    c1, c2 = (0.01 * 1.0) ** 2, (0.03 * 1.0) ** 2
    mu_a = _filter(a, w)
    mu_b = _filter(b, w)
    aa, bb, ab = a * a, b * b, a * b
    s_a = _filter(aa, w) - mu_a ** 2
    s_b = _filter(bb, w) - mu_b ** 2
    s_ab = _filter(ab, w) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * s_ab + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (s_a + s_b + c2)
    return float(np.mean(num / den))


def _filter(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return _gauss2d(x.astype(np.float64), w)


def _gauss2d(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    from scipy.ndimage import convolve
    return convolve(x, w, mode="reflect")


# --------------------------------------------------------------------------- lpips (optional)
@torch.no_grad()
def lpips_scores(pairs: list[tuple[np.ndarray, np.ndarray]], device) -> np.ndarray:
    import lpips as lpips_lib
    fn = lpips_lib.LPIPS(net="vgg").to(device).eval()
    out = []
    for a, b in pairs:
        ta = torch.from_numpy(a).permute(2, 0, 1)[None].to(device) * 2 - 1
        tb = torch.from_numpy(b).permute(2, 0, 1)[None].to(device) * 2 - 1
        out.append(float(fn(ta, tb)))
    return np.asarray(out)


# --------------------------------------------------------------------------- eval driver
def evaluate(gt_dir: Path, pred_dir: Path, use_lpips: bool, max_images: int) -> dict:
    gt_files = {p.name: p for p in list_images(gt_dir)}
    pred_files = list_images(pred_dir)
    if max_images:
        pred_files = pred_files[:max_images]
    pairs, used = [], []
    for p in pred_files:
        # pred names may be prefixed (e.g. enhanced_xxx.png); match by stem substring
        g = gt_files.get(p.name)
        if g is None:
            cands = [k for k in gt_files if p.stem in k or Path(k).stem in p.stem]
            g = gt_files[cands[0]] if cands else None
        if g is None:
            continue
        gt = np.asarray(load_rgb(g), dtype=np.float32) / 255.0
        pr = np.asarray(load_rgb(p).resize(gt.shape[1::-1], Image.LANCZOS),
                        dtype=np.float32) / 255.0
        pairs.append((gt, pr))
        used.append(p.name)

    if not pairs:
        raise SystemExit(f"[metrics] no comparable pairs between {gt_dir} and {pred_dir}")

    psnrs = [psnr(p, g) for g, p in pairs]
    ssims = [ssim(to_y(g), to_y(p)) for g, p in pairs]
    res = {"n": len(pairs), "psnr_mean": float(np.mean(psnrs)),
           "psnr_std": float(np.std(psnrs)), "ssim_mean": float(np.mean(ssims)),
           "ssim_std": float(np.std(ssims))}
    if use_lpips:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        res["lpips_mean"] = float(np.mean(lpips_scores(pairs, device)))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", required=True, help="ground-truth (DSLR) images dir")
    ap.add_argument("--pred", required=True, help="enhanced outputs dir")
    ap.add_argument("--lpips", action="store_true")
    ap.add_argument("--max-images", type=int, default=200)
    args = ap.parse_args()

    res = evaluate(Path(args.gt), Path(args.pred), args.lpips, args.max_images)
    print("[metrics] results:")
    for k, v in res.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
