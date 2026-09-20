"""DPED data preparation: download (Kaggle/official), verify, dedupe, extract 512px crops, split.

Outputs (default --out data/processed/iphone):
  crops/train/phone/*.png   crops/train/dslr/*.png
  crops/val/phone/*.png     crops/val/dslr/*.png
  test/phone/*.jpg          test/dslr/*.jpg      (full-size DPED test images)
  manifest.json             (split info + provenance)

Usage:
  python src/prepare_data.py --verify-only          # step 0: inspect structure, no writes
  python src/prepare_data.py --source kaggle        # default: kagglehub, fallback official
  python src/prepare_data.py --source official      # direct DPED patch release
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

from PIL import Image

from common import IMAGE_SIZE, file_sha1, list_images, load_rgb

# Expected official DPED layout:
#   dped/<phone>/training_data/{phone,dslr}_/...jpg   (pre-extracted patch pairs)
#   dped/<phone>/test_data/full_size_test_images/*.jpg
KAGGLE_DATASET = "philiphofmann/dped-dataset"
PATCH_DIR_NAMES = ("phone_", "dslr_")
TEST_IMAGE_DIRNAME = "full_size_test_images"


# --------------------------------------------------------------------------- download
def download_dped(source: str, workdir: Path) -> Path:
    """Return a directory that contains dped/<phone>/... (downloading if needed)."""
    if source in ("kaggle", "auto"):
        try:
            import kagglehub
            print(f"[data] downloading Kaggle dataset '{KAGGLE_DATASET}' via kagglehub ...")
            path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
            root = find_dped_root(path)
            if root is not None:
                print(f"[data] kagglehub OK -> {root}")
                return root
            print("[data] kagglehub download finished but no dped/ dir found inside")
        except Exception as e:  # noqa: BLE001 - report and fall through to official source
            print(f"[data] kagglehub failed: {e!r}")
        if source == "kaggle":
            print("[data] falling back to official DPED patches ...")

    url = "https://data.vision.ee.ethz.ch/cvl/DPED.zip"
    zip_path = workdir / "DPED.zip"
    workdir.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        print(f"[data] downloading official DPED patches: {url}")
        shutil.copyfileobj(_stream(url), open(zip_path, "wb"))
    extract_dir = workdir / "official"
    if not (extract_dir / "dped").exists():
        print("[data] extracting ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
    root = find_dped_root(extract_dir)
    if root is None:
        raise RuntimeError("Official DPED download did not contain dped/ dir")
    return root


def _stream(url: str):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=60)


def find_dped_root(base: Path) -> Path | None:
    """Locate the dir that directly contains sony/ iphone/ blackberry/ subdirs."""
    candidates = [base, base / "dped"]
    for p in base.rglob("dped"):
        candidates.append(p)
    for c in candidates:
        if c.is_dir() and all((c / s).is_dir() for s in ("sony", "iphone", "blackberry")):
            return c
    return None


# --------------------------------------------------------------------------- structure inspect
def inspect_dped(root: Path) -> dict:
    """Probe the DPED tree for the iphone subset and report what we found."""
    info: dict = {"root": str(root), "iphone": {}}
    ip = root / "iphone"
    if not ip.exists():
        print(f"[data] WARNING: no iphone/ subdir under {root}")
        return info
    for child in sorted(ip.iterdir()):
        if child.is_dir():
            n = len(list_images(child))
            info["iphone"][child.name] = {"images": n}
            print(f"[data] iphone/{child.name}: {n} images")
    return info


def locate_patch_dirs(root: Path) -> tuple[Path, Path] | None:
    ip = root / "iphone"
    for parent in (ip / "training_data", ip):
        if not parent.exists():
            continue
        phone = parent / PATCH_DIR_NAMES[0]
        dslr = parent / PATCH_DIR_NAMES[1]
        if phone.is_dir() and dslr.is_dir():
            return phone, dslr
    return None


def locate_test_dir(root: Path) -> Path | None:
    ip = root / "iphone"
    for cand in (ip / "test_data" / TEST_IMAGE_DIRNAME, ip / "test_data"):
        if cand.is_dir() and len(list_images(cand)) > 0:
            return cand
    return None

# --------------------------------------------------------------------------- crop extraction
def load_pair(phone: Path, dslr: Path):
    """Load a patch pair, enforcing exact shape equality (misaligned -> None)."""
    p, d = load_rgb(phone), load_rgb(dslr)
    if p.size != d.size:
        return None
    return p, d


def crop_windows(w: int, h: int, size: int, stride: int):
    ys = range(0, max(h - size, 0) + 1, stride) or [0]
    xs = range(0, max(w - size, 0) + 1, stride) or [0]
    for y in ys:
        for x in xs:
            yield x, y


def extract_crops(patch_dir_phone: Path, patch_dir_dslr: Path, out_root: Path,
                  per_image: int, size: int, stride: int,
                  min_sharpness: float, dedupe: bool, seed: int,
                  max_crops: int = 0) -> tuple[list[tuple[str, str]], list[str]]:
    """Cut fixed-size aligned crops from DPED patch pairs; returns kept + skipped names."""
    import random as _random
    import numpy as np

    rng = _random.Random(seed)
    dslr_map = {p.name: p for p in list_images(patch_dir_dslr)}
    out_phone = out_root / "phone"
    out_dslr = out_root / "dslr"
    out_phone.mkdir(parents=True, exist_ok=True)
    out_dslr.mkdir(parents=True, exist_ok=True)

    kept: list[tuple[str, str]] = []
    skipped: list[str] = []
    seen_hashes: set[str] = set()
    n_resized = 0

    for phone_path in sorted(list_images(patch_dir_phone)):
        dslr_path = dslr_map.get(phone_path.name)
        if dslr_path is None:
            skipped.append(phone_path.name)
            continue
        try:
            pair = load_pair(phone_path, dslr_path)
        except Exception:
            skipped.append(phone_path.name)
            continue
        if pair is None:
            skipped.append(phone_path.name)
            continue
        p_img, d_img = pair
        w, h = p_img.size
        if w < size or h < size:
            # Official DPED patches are 100x100 -> upsample to target size (LANCZOS).
            # Tone/color enhancement signal survives upsampling; fine detail is limited.
            p_img = p_img.resize((size, size), Image.LANCZOS)
            d_img = d_img.resize((size, size), Image.LANCZOS)
            wins = [(0, 0)]
            n_resized += 1
        else:
            wins = list(crop_windows(w, h, size, stride))
        wins = list(crop_windows(w, h, size, stride))
        if not wins:
            wins = [(0, 0)]
        if max_crops and len(kept) >= max_crops:
            print(f"[data] reached --max-crops={max_crops}, stopping early")
            break
        rng.shuffle(wins)
        n_saved = 0
        for (x, y) in wins:
            if n_saved >= per_image:
                break
            p_c = p_img.crop((x, y, x + size, y + size))
            d_c = d_img.crop((x, y, x + size, y + size))
            if min_sharpness > 0:
                gray = np.asarray(p_c.convert("L"), dtype=np.float32)
                sharp = float(np.abs(np.diff(gray, axis=1)).mean())
                if sharp < min_sharpness:
                    continue
            if dedupe:
                hsh = file_sha1(phone_path)  # patch-level identity via source file + coords
                hsh = f"{hsh}_{x}_{y}"
                if hsh in seen_hashes:
                    continue
                seen_hashes.add(hsh)
            name = f"{phone_path.stem}_{x}_{y}.png"
            p_c.save(out_phone / name)
            d_c.save(out_dslr / name)
            kept.append((name, phone_path.name))
            n_saved += 1
    if n_resized:
        print(f"[data] NOTE: {n_resized} sources were smaller than {size}px -> "
              f"upsampled (LANCZOS). Detail quality limited by source patch size.")
    return kept, skipped


# --------------------------------------------------------------------------- split + manifest
def split_and_manifest(crops_root: Path, out_root: Path, val_frac: float, seed: int,
                       test_dir: Path | None) -> dict:
    """Group crops by source patch, split at group level, write val/ and manifest.json."""
    import random as _random

    rng = _random.Random(seed)
    train_dir = out_root / "crops" / "train"
    val_dir = out_root / "crops" / "val"
    for d in (train_dir / "phone", train_dir / "dslr", val_dir / "phone", val_dir / "dslr"):
        d.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list[str]] = {}
    for p in list_images(train_dir / "phone"):
        stem = p.stem.rsplit("_", 2)[0]
        groups.setdefault(stem, []).append(p.name)
    stems = sorted(groups)
    rng.shuffle(stems)
    n_val = max(1, int(round(len(stems) * val_frac)))
    val_stems = set(stems[:n_val])
    moved = 0
    for stem in stems:
        if stem not in val_stems:
            continue
        for name in groups[stem]:
            for sub in ("phone", "dslr"):
                (train_dir / sub / name).rename(val_dir / sub / name)
            moved += 1

    manifest = {
        "dataset": "DPED iphone",
        "crop_size": IMAGE_SIZE,
        "train_crops": len(list_images(train_dir / "phone")),
        "val_crops": len(list_images(val_dir / "phone")),
        "val_groups": n_val,
        "moved": moved,
    }
    if test_dir is not None:
        t_out = out_root / "test"
        (t_out / "phone").mkdir(parents=True, exist_ok=True)
        (t_out / "dslr").mkdir(parents=True, exist_ok=True)
        d_map = {p.name: p for p in list_images(test_dir)}
        n_test = 0
        for p in list_images(test_dir):
            d = d_map.get(p.name)
            if d is None:
                continue
            shutil.copy(p, t_out / "phone" / p.name)
            shutil.copy(d, t_out / "dslr" / p.name)
            n_test += 1
        manifest["test_images"] = n_test
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["kaggle", "official", "auto"], default="auto")
    ap.add_argument("--out", default="data/processed/iphone")
    ap.add_argument("--workdir", default="data/raw")
    ap.add_argument("--per-image", type=int, default=4, help="max crops kept per patch pair")
    ap.add_argument("--max-crops", type=int, default=20000,
                    help="total crop cap (0 = unlimited); keeps Kaggle disk budget safe")
    ap.add_argument("--size", type=int, default=IMAGE_SIZE)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--min-sharpness", type=float, default=1.0,
                    help="mean |dx| on grayscale phone crop; 0 disables")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    root = download_dped(args.source, workdir)
    info = inspect_dped(root)
    if args.verify_only:
        print(json.dumps(info, indent=2))
        return

    located = locate_patch_dirs(root)
    if located is None:
        print("[data] could not locate training patch dirs; run --verify-only and inspect")
        sys.exit(2)
    phone_dir, dslr_dir = located
    test_dir = locate_test_dir(root)
    out_root = Path(args.out)
    tmp = out_root / "crops" / "train"
    if tmp.exists():
        shutil.rmtree(tmp)
    print(f"[data] extracting {args.size}px crops (per_image={args.per_image}, stride={args.stride}) ...")
    kept, skipped = extract_crops(phone_dir, dslr_dir, tmp, args.per_image, args.size,
                                  args.stride, args.min_sharpness, not args.no_dedupe,
                                  args.seed, args.max_crops)
    print(f"[data] kept {len(kept)} crops, skipped {len(skipped)} patch pairs")
    manifest = split_and_manifest(tmp.parent, out_root, args.val_frac, args.seed, test_dir)
    print("[data] manifest:", json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
