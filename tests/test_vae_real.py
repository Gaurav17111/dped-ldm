"""Real-model smoke test (network needed, ~350MB download on first run).
Validates the exact latent path used on Kaggle with the actual SD 2.1-base VAE.
Run: .venv-smoke/Scripts/python.exe tests/test_vae_real.py
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

TMP = Path(__file__).parent.parent / "data" / "smoke_vae_tmp"


def main() -> None:
    from common import (PairedCropDataset, build_latent_cache, decode_latents,
                        encode_latents, latent_cache_paths, load_vae)

    device = torch.device("cpu")
    vae = load_vae("fp32", device)
    print("VAE loaded:", type(vae).__name__)

    # fake paired crops
    if TMP.exists():
        shutil.rmtree(TMP)
    phone, dslr = TMP / "crops" / "train" / "phone", TMP / "crops" / "train" / "dslr"
    phone.mkdir(parents=True)
    dslr.mkdir(parents=True)
    rng = np.random.default_rng(3)
    for i in range(2):
        arr = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
        Image.fromarray(arr).save(phone / f"p{i}.png")
        Image.fromarray(arr).save(dslr / f"p{i}.png")

    ds = PairedCropDataset(phone, dslr)
    batch = torch.utils.data.StackDataset if False else None  # noqa: F841
    loader_batch = torch.utils.data.DataLoader(ds, batch_size=2)
    b = next(iter(loader_batch))

    lat = encode_latents(vae, b["phone"])
    assert lat.shape == (2, 4, 64, 64), f"latent shape {lat.shape}"
    assert -50 < float(lat.min()) and float(lat.max()) < 50, "latent scale sane"
    print("encode OK:", tuple(lat.shape), "range",
          f"[{float(lat.min()):.2f}, {float(lat.max()):.2f}]")

    rec = decode_latents(vae, lat)
    assert rec.shape == (2, 3, 512, 512), f"decode shape {rec.shape}"
    assert float(rec.min()) >= -1.5 and float(rec.max()) <= 1.5, "decode range sane"
    print("decode OK:", tuple(rec.shape))

    # full cache path with the real VAE
    cache = TMP / "latent_cache"
    build_latent_cache(vae, ds, cache, "train", batch_size=2, device=device)
    p, d = latent_cache_paths(cache, "train")
    assert p.exists() and d.exists()
    from common import LatentPairDataset
    lds = LatentPairDataset(cache, "train")
    assert lds[0]["phone"].shape == (4, 64, 64)
    assert lds.phone.dtype == torch.float16
    print("latent cache with real VAE OK")

    shutil.rmtree(TMP)
    print("REAL-VAE SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
