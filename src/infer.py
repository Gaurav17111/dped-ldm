"""Inference: enhance DPED images with a trained Stage A / Stage B checkpoint.

Modes:
  crops  - val crops -> side-by-side grids (phone | enhanced | dslr)
  full   - full-size DPED test images -> tiled inference with feathered blending

Usage:
  python src/infer.py --ckpt runs/stage_a/controlnet_step_0015000.pt --mode crops
  python src/infer.py --ckpt runs/stage_b/finetune_step_0006000.pt --mode full --ema
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import (IMAGE_SIZE, PairedCropDataset, decode_latents, encode_latents,
                    load_checkpoint, load_rgb, make_grid, save_png, seed_everything)
from train_controlnet import empty_prompt_embeds

# --------------------------------------------------------------------------- model loading
def load_pipeline(ckpt_path: str, dtype, device, use_ema: bool):
    """Returns (unet, controlnet, text_encoder) from a Stage A or Stage B checkpoint."""
    from diffusers import UNet2DConditionModel
    from train_controlnet import load_models

    unet, controlnet, text_encoder, _ = load_models(dtype, device)
    state = load_checkpoint(ckpt_path)
    if "unet" in state:  # Stage B checkpoint bundles both
        unet.load_state_dict(state["unet"])
        controlnet.load_state_dict(state["controlnet"])
        if use_ema and state.get("ema"):
            _apply_ema(unet, state["ema"])
        print(f"[infer] loaded Stage B checkpoint {Path(ckpt_path).name}"
              f"{' (EMA weights)' if use_ema and state.get('ema') else ''}")
    else:
        controlnet.load_state_dict(state["controlnet"])
        print(f"[infer] loaded Stage A checkpoint {Path(ckpt_path).name}")
    return unet.to(device).eval(), controlnet.to(device).eval(), text_encoder.to(device).eval()


def _apply_ema(unet, ema_sd: dict) -> None:
    with torch.no_grad():
        for n, p in unet.named_parameters():
            if n in ema_sd:
                p.data.copy_(ema_sd[n].to(p.dtype))


def export_diffusers(controlnet, unet, out_dir: str) -> None:
    """Save as diffusers-format folders (portable, reloadable without our .pt)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    controlnet.save_pretrained(str(Path(out_dir) / "controlnet"))
    unet.save_pretrained(str(Path(out_dir) / "unet"))
    print(f"[infer] exported diffusers pipeline parts -> {out_dir}")


# --------------------------------------------------------------------------- core sampling
@torch.no_grad()
def enhance_latents(unet, controlnet, emb, phone_lat: torch.Tensor,
                    sched, strength: float = 0.85) -> torch.Tensor:
    """SDEdit-style: noise the phone latents to t0 = strength*T, then denoise to data."""
    device = phone_lat.device
    b = phone_lat.shape[0]
    t0 = max(1, int(strength * sched.config.num_train_timesteps))
    sched.set_timesteps(t0, device=device)
    noise = torch.randn_like(phone_lat)
    lat = sched.add_noise(phone_lat, noise, torch.full((b,), sched.timesteps[0].item(),
                                                       device=device, dtype=torch.long))
    for t in sched.timesteps:
        tb = torch.full((b,), int(t), device=device, dtype=torch.long)
        down, mid = controlnet(lat, tb, encoder_hidden_states=emb,
                               controlnet_cond=phone_lat, return_dict=True)
        eps = unet(lat, tb, encoder_hidden_states=emb,
                   down_block_additional_residuals=down,
                   mid_block_additional_residual=mid).sample
        lat = sched.step(eps, t, lat).prev_sample
    return lat


@torch.no_grad()
def enhance_tiles(unet, controlnet, text_encoder, vae, phone_img: Image.Image,
                  dtype, device, tile: int = 512, stride: int = 384,
                  steps: int = 30, strength: float = 0.85, batch_tiles: int = 4) -> Image.Image:
    """Full-res tiled enhancement with cosine-feather blending in pixel space."""
    from diffusers import DDPMScheduler

    sched = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base",
                                          subfolder="scheduler")
    W, H = phone_img.size
    img = np.asarray(phone_img, dtype=np.float32) / 255.0
    out = np.zeros_like(img)
    weight = np.zeros((H, W, 1), dtype=np.float32)

    ys = sorted(set(list(range(0, max(H - tile, 0) + 1, stride)) + [max(H - tile, 0)]))
    xs = sorted(set(list(range(0, max(W - tile, 0) + 1, stride)) + [max(W - tile, 0)]))
    tiles = [(x, y) for y in ys for x in xs]

    def feather(size: int) -> np.ndarray:
        r = np.hanning(size)
        return np.outer(r, r).astype(np.float32)

    emb_cache: dict = {}
    for i in range(0, len(tiles), batch_tiles):
        chunk = tiles[i:i + batch_tiles]
        batch = np.stack([img[y:y + tile, x:x + tile] for (x, y) in chunk])
        t_img = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device, dtype=dtype)
        t_img = t_img * 2.0 - 1.0
        lat = encode_latents(vae, t_img)
        key = lat.shape[0]
        if key not in emb_cache:
            emb_cache[key] = empty_prompt_embeds(text_encoder, key, device, dtype)
        emb = emb_cache[key]
        lat_out = enhance_latents(unet, controlnet, emb, lat, sched, strength)
        rec = decode_latents(vae, lat_out.float()).clamp(-1, 1)
        rec = ((rec + 1) / 2).float().cpu().numpy().transpose(0, 2, 3, 1)
        for j, (x, y) in enumerate(chunk):
            f = feather(tile)[..., None]
            out[y:y + tile, x:x + tile] += rec[j] * f
            weight[y:y + tile, x:x + tile] += f
        print(f"[infer] tile batch {i // batch_tiles + 1}/{(len(tiles) + batch_tiles - 1) // batch_tiles}")
    out = out / np.maximum(weight, 1e-6)
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


# --------------------------------------------------------------------------- modes
@torch.no_grad()
def run_crops(args, unet, controlnet, text_encoder, vae, device, dtype) -> None:
    from diffusers import DDPMScheduler

    ds = PairedCropDataset(Path(args.data) / "crops" / "val" / "phone",
                           Path(args.data) / "crops" / "val" / "dslr")
    n = min(args.num, len(ds))
    sched = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base",
                                          subfolder="scheduler")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        sample = ds[i]
        phone = sample["phone"].unsqueeze(0).to(device, dtype=dtype)
        dslr = sample["dslr"].unsqueeze(0).to(device, dtype=dtype)
        lat = encode_latents(vae, phone)
        emb = empty_prompt_embeds(text_encoder, 1, device, dtype)
        lat_out = enhance_latents(unet, controlnet, emb, lat, sched,
                                  strength=args.strength)
        rec = decode_latents(vae, lat_out.float()).clamp(-1, 1)[0]
        rec = ((rec + 1) / 2).float().cpu().permute(1, 2, 0).numpy()
        phone_np = ((phone[0] + 1) / 2).float().cpu().permute(1, 2, 0).numpy()
        dslr_np = ((dslr[0] + 1) / 2).float().cpu().permute(1, 2, 0).numpy()
        grid = make_grid([phone_np, rec, dslr_np], ["phone", "enhanced", "dslr"])
        save_png(grid, out_dir / f"{Path(sample['key']).stem}_grid.png")
        print(f"[infer] [{i + 1}/{n}] {sample['key']}")


@torch.no_grad()
def run_full(args, unet, controlnet, text_encoder, vae, device, dtype) -> None:
    test_phone = Path(args.data) / "test" / "phone"
    test_dslr = Path(args.data) / "test" / "dslr"
    files = sorted(p for p in test_phone.iterdir() if p.suffix.lower() in (".jpg", ".png"))[:args.num]
    if not files:
        raise SystemExit(f"[infer] no test images under {test_phone} (run prepare_data first)")
    out_dir = Path(args.out) / "full"
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in files:
        phone_img = load_rgb(p)
        enhanced = enhance_tiles(unet, controlnet, text_encoder, vae, phone_img,
                                 dtype, device, steps=args.steps, strength=args.strength)
        save_png(enhanced, out_dir / f"enhanced_{p.stem}.png")
        d_path = test_dslr / p.name
        if d_path.exists():
            dslr_img = load_rgb(d_path).resize(enhanced.size, Image.LANCZOS)
            side = make_grid([phone_img.resize((768, int(768 * phone_img.height / phone_img.width)),
                                               Image.LANCZOS),
                              enhanced.resize((768, int(768 * enhanced.height / enhanced.width)),
                                              Image.LANCZOS),
                              dslr_img.resize((768, int(768 * dslr_img.height / dslr_img.width)),
                                              Image.LANCZOS)],
                             ["phone", "enhanced", "dslr"])
            save_png(side, out_dir / f"side_{p.stem}.png")
        print(f"[infer] full-res done: {p.name}")


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/processed/iphone")
    ap.add_argument("--out", default="runs/infer")
    ap.add_argument("--mode", choices=["crops", "full"], default="crops")
    ap.add_argument("--ema", action="store_true", help="use EMA weights (Stage B checkpoints)")
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--strength", type=float, default=0.85)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--stride", type=int, default=384)
    ap.add_argument("--batch-tiles", type=int, default=4)
    ap.add_argument("--export", default="", help="optional: export diffusers format to this dir")
    ap.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from common import auto_dtype, load_vae
    dtype = auto_dtype(args.precision)
    unet, controlnet, text_encoder = load_pipeline(args.ckpt, dtype, device, args.ema)
    vae = load_vae(args.precision, device)
    if args.export:
        export_diffusers(controlnet, unet, args.export)
    if args.mode == "crops":
        run_crops(args, unet, controlnet, text_encoder, vae, device, dtype)
    else:
        run_full(args, unet, controlnet, text_encoder, vae, device, dtype)


if __name__ == "__main__":
    main()
