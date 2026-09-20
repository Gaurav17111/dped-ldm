"""Network-free smoke tests: crops, split, EMA, grids, metrics.
Run: .venv-smoke/Scripts/python.exe tests/test_pipeline.py
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

ROOT = Path(__file__).parent.parent
TMP = ROOT / "data" / "smoke_tmp"


def test_make_grid():
    from common import make_grid
    a = np.random.rand(64, 64, 3)
    g = make_grid([a, a], ["x", "y"])
    assert g.size[0] > 128 and g.size[1] >= 64


def test_ema():
    from common import EMA
    lin = torch.nn.Linear(8, 8)
    ema = EMA(lin, 0.9, torch.device("cpu"))
    assert set(ema.shadow) == {"weight", "bias"}
    orig = lin.weight.clone()
    with torch.no_grad():
        lin.weight.add_(1.0)               # training weights = orig + 1
    ema.update(lin)                        # shadow ~= orig + 0.1 (fp16)
    backup = ema.copy_into(lin)            # EMA in, training weights saved
    assert torch.allclose(lin.weight, ema.shadow["weight"].float(), atol=1e-3)
    with torch.no_grad():
        lin.weight.add_(5.0)               # drift further
    ema.restore(lin, backup)               # training weights back
    assert torch.allclose(lin.weight, orig + 1.0)
    # state_dict roundtrip
    sd = ema.state_dict()
    ema2 = EMA(lin, 0.9, torch.device("cpu"))
    ema2.load_state_dict(sd, torch.device("cpu"))
    assert torch.allclose(ema2.shadow["weight"].float(), ema.shadow["weight"].float())


def test_paired_dataset_and_latent_cache():
    from common import PairedCropDataset, build_latent_cache, latent_cache_paths

    if TMP.exists():
        shutil.rmtree(TMP)
    phone, dslr = TMP / "crops" / "train" / "phone", TMP / "crops" / "train" / "dslr"
    phone.mkdir(parents=True); dslr.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(4):
        arr = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(phone / f"p{i}.png")
        Image.fromarray(arr).save(dslr / f"p{i}.png")

    ds = PairedCropDataset(phone, dslr)
    assert len(ds) == 4
    s = ds[0]
    assert s["phone"].shape == (3, 512, 512)  # resized to IMAGE_SIZE
    assert float(s["phone"].min()) >= -1.0 and float(s["phone"].max()) <= 1.0

    # latent cache path check without a real VAE: monkeypatch encode
    import common
    class FakeVAE:
        dtype = torch.float32
        class config:  # noqa: N801
            scaling_factor = 0.18215
        def encode(self, x):
            class LD:
                def sample(self):
                    return torch.nn.functional.avg_pool2d(x, 8)
            return type("E", (), {"latent_dist": LD()})()
    cache_dir = TMP / "cache"
    build_latent_cache(FakeVAE(), ds, cache_dir, "train", batch_size=2, device="cpu")
    p, d = latent_cache_paths(cache_dir, "train")
    assert p.exists() and d.exists()
    lds = common.LatentPairDataset(cache_dir, "train")
    assert lds.phone.dtype == torch.float16
    assert lds[0]["phone"].shape == (3, 64, 64)  # channels x H/8 x W/8 (fake VAE)


def test_ssim_psnr():
    from metrics import psnr, ssim, to_y
    rng = np.random.default_rng(1)
    a = rng.random((64, 64))
    b = np.clip(a + rng.normal(0, 0.05, a.shape), 0, 1)
    assert 10 < psnr(a, b) < 40
    assert ssim(a, b) > 0.5
    assert ssim(a, a) > 0.99
    y = to_y(rng.random((8, 8, 3)))
    assert y.shape == (8, 8)


def test_prepare_data_split():
    import prepare_data
    if TMP.exists():
        shutil.rmtree(TMP)
    train = TMP / "out" / "crops" / "train"
    (train / "phone").mkdir(parents=True)
    (train / "dslr").mkdir(parents=True)
    for i in range(10):
        img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
        img.save(train / "phone" / f"g{i}_0_0.png")
        img.save(train / "dslr" / f"g{i}_0_0.png")
    manifest = prepare_data.split_and_manifest(train.parent, TMP / "out", 0.3, 0,
                                               test_dir=None)
    assert manifest["train_crops"] + manifest["val_crops"] == 10
    assert manifest["val_crops"] >= 1
    assert (TMP / "out" / "crops" / "val" / "phone").exists()


def test_extract_crops_small_patches():
    import prepare_data
    if TMP.exists():
        shutil.rmtree(TMP)
    src = TMP / "src"
    (src / "phone_").mkdir(parents=True)
    (src / "dslr_").mkdir(parents=True)
    rng = np.random.default_rng(2)
    for i in range(3):
        arr = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(arr).save(src / "phone_" / f"s{i}.jpg")
        Image.fromarray(arr).save(src / "dslr_" / f"s{i}.jpg")
    kept, skipped = prepare_data.extract_crops(
        src / "phone_", src / "dslr_", TMP / "out2", per_image=2, size=512, stride=64,
        min_sharpness=0.0, dedupe=False, seed=0, max_crops=100)
    assert len(kept) == 3 and not skipped
    assert Image.open(TMP / "out2" / "phone" / kept[0][0]).size == (512, 512)


def cleanup():
    if TMP.exists():
        shutil.rmtree(TMP)


if __name__ == "__main__":
    fns = [test_make_grid, test_ema, test_paired_dataset_and_latent_cache,
           test_ssim_psnr, test_prepare_data_split, test_extract_crops_small_patches]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    cleanup()
    print("ALL SMOKE TESTS PASSED")
