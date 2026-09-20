"""Shared utilities: model loading, latent caching, paired datasets, image grids."""
from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

# --------------------------------------------------------------------------- constants
SD_MODEL_ID = "stabilityai/stable-diffusion-2-1-base"
IMAGE_SIZE = 512  # SD 2.1-base native resolution
VAE_DOWNSAMPLE = 8
LATENT_CHANNELS = 4

DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


# --------------------------------------------------------------------------- seeding
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def auto_dtype(precision: str) -> torch.dtype:
    """Resolve requested precision; T4 GPUs have no bf16, so fall back to fp16."""
    if precision == "bf16" and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        print("[common] bf16 not supported on this GPU -> falling back to fp16")
        return torch.float16
    return DTYPE_MAP[precision]


# --------------------------------------------------------------------------- io helpers
def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


def file_sha1(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_rgb(path: str | Path) -> Image.Image:
    img = Image.open(path)
    return img.convert("RGB")


def save_png(img: Image.Image, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


# --------------------------------------------------------------------------- vae latents
def load_vae(precision: str = "fp16", device: str | torch.device = "cuda") -> "torch.nn.Module":
    from diffusers import AutoencoderKL

    dtype = auto_dtype(precision)
    device = torch.device(device)
    vae = AutoencoderKL.from_pretrained(SD_MODEL_ID, subfolder="vae", torch_dtype=dtype)
    return vae.to(device).eval()


@torch.no_grad()
def encode_latents(vae, batch: torch.Tensor) -> torch.Tensor:
    """Images in [-1, 1], (B,3,H,W) -> latents (B,4,H/8,W/8). Matches SD training scaling."""
    batch = batch.to(vae.dtype)
    posterior = vae.encode(batch).latent_dist
    return posterior.sample() * vae.config.scaling_factor


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    latents = latents.to(vae.dtype) / vae.config.scaling_factor
    return vae.decode(latents).sample


# --------------------------------------------------------------------------- grid visuals
def _to_pil(img):
    """Accept PIL Image or (H,W,3) float[0..1] / uint8 array."""
    import numpy as _np
    if isinstance(img, Image.Image):
        return img
    arr = _np.asarray(img)
    if arr.dtype != _np.uint8:
        arr = arr.astype(_np.float32)
        arr = _np.clip(arr * 255.0, 0, 255).astype(_np.uint8)
    return Image.fromarray(arr)


def make_grid(images: list, labels: list[str] | None = None, pad: int = 4) -> Image.Image:
    images = [_to_pil(im) for im in images]
    n = len(images)
    w, h = images[0].size
    grid = Image.new("RGB", (n * w + (n + 1) * pad, h + 2 * pad + (18 if labels else 0)), (24, 24, 24))
    draw = ImageDraw.Draw(grid)
    x = pad
    for i, img in enumerate(images):
        grid.paste(img, (x, pad))
        if labels:
            draw.text((x + 4, h + pad + 2), labels[i], fill=(240, 240, 240))
        x += w + pad
    return grid


# --------------------------------------------------------------------------- datasets
@dataclass
class PairedCrop:
    phone_path: str
    dslr_path: str
    crop_w: int
    crop_h: int
    flip: bool = False

    def key(self) -> str:
        return f"{Path(self.phone_path).stem}_{self.crop_w}_{self.crop_h}_{int(self.flip)}"


class PairedCropDataset(torch.utils.data.Dataset):
    """Loads pre-cut aligned crop pairs stored as PNG files under two parallel folders."""

    def __init__(self, phone_dir: str | Path, dslr_dir: str | Path):
        self.phone_dir = Path(phone_dir)
        self.dslr_dir = Path(dslr_dir)
        self.items = self._scan()

    def _scan(self) -> list[PairedCrop]:
        dslr_map = {p.name: p for p in list_images(self.dslr_dir)}
        items = []
        for p in list_images(self.phone_dir):
            d = dslr_map.get(p.name)
            if d is not None:
                items.append(PairedCrop(str(p), str(d), 0, 0))
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        from torchvision.transforms import functional as TF

        it = self.items[idx]
        phone = load_rgb(it.phone_path)
        dslr = load_rgb(it.dslr_path)
        size = (IMAGE_SIZE, IMAGE_SIZE)
        phone = phone.resize(size, Image.LANCZOS) if phone.size != size else phone
        dslr = dslr.resize(size, Image.LANCZOS) if dslr.size != size else dslr
        if it.flip:
            phone = phone.transpose(Image.FLIP_LEFT_RIGHT)
            dslr = dslr.transpose(Image.FLIP_LEFT_RIGHT)
        phone_t = TF.to_tensor(phone) * 2.0 - 1.0
        dslr_t = TF.to_tensor(dslr) * 2.0 - 1.0
        return {"phone": phone_t, "dslr": dslr_t, "key": it.key()}


def latent_cache_paths(cache_dir: str | Path, split: str) -> tuple[Path, Path]:
    cache_dir = Path(cache_dir)
    return cache_dir / f"{split}_phone.pt", cache_dir / f"{split}_dslr.pt"


@torch.no_grad()
def build_latent_cache(vae, dataset: PairedCropDataset, cache_dir: str | Path, split: str,
                       batch_size: int = 8, device: str | torch.device = "cuda") -> None:
    """Pre-encode all crop pairs into two stacked latent tensors (fast epoch restarts)."""
    phone_p, dslr_p = latent_cache_paths(cache_dir, split)
    if phone_p.exists() and dslr_p.exists():
        print(f"[common] latent cache for split '{split}' already exists -> skip")
        return
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                         num_workers=2, pin_memory=True)
    ph, ds = [], []
    for batch in loader:
        ph.append(encode_latents(vae, batch["phone"].to(device)).half().cpu())
        ds.append(encode_latents(vae, batch["dslr"].to(device)).half().cpu())
    torch.save(torch.cat(ph), phone_p)
    torch.save(torch.cat(ds), dslr_p)
    print(f"[common] cached {len(dataset)} latent pairs (fp16) for split '{split}' -> {cache_dir}")


class LatentPairDataset(torch.utils.data.Dataset):
    """Serves pre-cached VAE latents; used for ControlNet / UNet training."""

    def __init__(self, cache_dir: str | Path, split: str):
        phone_p, dslr_p = latent_cache_paths(cache_dir, split)
        self.phone = torch.load(phone_p, weights_only=True)
        self.dslr = torch.load(dslr_p, weights_only=True)
        assert self.phone.shape[0] == self.dslr.shape[0], "latent cache mismatch"

    def __len__(self) -> int:
        return self.phone.shape[0]

    def __getitem__(self, idx: int):
        return {"phone": self.phone[idx], "dslr": self.dslr[idx]}


# --------------------------------------------------------------------------- EMA
class EMA:
    """Exponential moving average of trainable params (kept in fp16 on device)."""

    def __init__(self, model: torch.nn.Module, decay: float, device):
        self.decay = decay
        self.shadow = {n: p.detach().clone().half().to(device)
                       for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach().half(), alpha=1.0 - d)

    def state_dict(self) -> dict:
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: dict, device) -> None:
        self.shadow = {k: v.to(device).half() for k, v in sd.items()}

    @torch.no_grad()
    def copy_into(self, model: torch.nn.Module) -> dict:
        """Swap EMA weights into the model; returns backup dict to restore later."""
        backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n].to(p.dtype))
        return backup

    @torch.no_grad()
    def restore(self, model: torch.nn.Module, backup: dict) -> None:
        for n, p in model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n])


# --------------------------------------------------------------------------- checkpointing
def save_checkpoint(path: str | Path, **objects) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(objects, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)
