"""Stage A: train a ControlNet adapter on frozen SD 2.1-base for DPED iPhone -> DSLR enhancement.

Conditioning = phone-image VAE latents (concatenated to UNet input channels).
Text conditioning = fixed empty prompt (unconditional).

Usage:
  python src/train_controlnet.py --data data/processed/iphone --steps 15000
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from common import (LATENT_CHANNELS, SD_MODEL_ID, LatentPairDataset, auto_dtype,
                    load_checkpoint, make_grid, save_checkpoint, seed_everything)

_tokenizer = None


def get_tokenizer() -> CLIPTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = CLIPTokenizer.from_pretrained(SD_MODEL_ID, subfolder="tokenizer")
    return _tokenizer


@torch.no_grad()
def empty_prompt_embeds(text_encoder, batch: int, device, dtype) -> torch.Tensor:
    """Fixed empty-prompt CLIP embeddings (B,77,1024) - our unconditional conditioning."""
    tok = get_tokenizer()
    tokens = tok([""], padding="max_length", max_length=tok.model_max_length,
                 return_tensors="pt")
    ids = tokens.input_ids.to(device).expand(batch, -1)
    emb = text_encoder(ids)[0]
    return emb.to(dtype)


# --------------------------------------------------------------------------- model assembly
def load_models(precision: str, device: torch.device):
    dtype = auto_dtype(precision)
    unet = UNet2DConditionModel.from_pretrained(SD_MODEL_ID, subfolder="unet",
                                                torch_dtype=dtype)
    controlnet = ControlNetModel.from_unet(unet)
    text_encoder = CLIPTextModel.from_pretrained(SD_MODEL_ID, subfolder="text_encoder",
                                                 torch_dtype=dtype)
    return unet.to(device), controlnet.to(device), text_encoder.to(device), dtype


def make_trainable_controlnet(controlnet: ControlNetModel) -> ControlNetModel:
    """Fresh ControlNet: zero-conv output layers start as identity-ish no-op."""
    for p in controlnet.parameters():
        p.requires_grad_(True)
    controlnet.controlnet_down_blocks.apply(_zero_last_conv)
    controlnet.controlnet_mid_block.apply(_zero_last_conv)
    return controlnet


def _zero_last_conv(module: torch.nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.zeros_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)


# --------------------------------------------------------------------------- train loop
def train(args) -> None:
    from common import build_latent_cache, load_vae

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[train-cnet] WARNING: no CUDA device, running CPU smoke-test mode")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- data (latent cache built once, then reused across sessions)
    cache_dir = Path(args.data) / "latent_cache"
    try:
        train_ds = LatentPairDataset(cache_dir, "train")
        val_ds = LatentPairDataset(cache_dir, "val")
        print(f"[train-cnet] latent cache loaded: train={len(train_ds)} val={len(val_ds)}")
    except (FileNotFoundError, AssertionError):
        print("[train-cnet] latent cache missing -> building from crops (one-time cost)")
        vae = load_vae(args.precision, device)
        crop_train = Path(args.data) / "crops" / "train"
        crop_val = Path(args.data) / "crops" / "val"
        from common import PairedCropDataset
        build_latent_cache(vae, PairedCropDataset(crop_train / "phone", crop_train / "dslr"),
                           cache_dir, "train", args.cache_batch, device)
        build_latent_cache(vae, PairedCropDataset(crop_val / "phone", crop_val / "dslr"),
                           cache_dir, "val", args.cache_batch, device)
        del vae
        torch.cuda.empty_cache()
        train_ds = LatentPairDataset(cache_dir, "train")
        val_ds = LatentPairDataset(cache_dir, "val")

    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        pin_memory=(device.type == "cuda"), drop_last=True)

    # --- models
    unet, controlnet, text_encoder, dtype = load_models(args.precision, device)
    controlnet = make_trainable_controlnet(controlnet)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()
    unet.eval()

    vae = load_vae(args.precision, device)
    noise_sched = DDPMScheduler.from_pretrained(SD_MODEL_ID, subfolder="scheduler")

    # --- optimizer / scaler / resume
    params = [p for p in controlnet.parameters() if p.requires_grad]
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=args.weight_decay)
        print("[train-cnet] optimizer: AdamW8bit")
    except Exception:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        print("[train-cnet] optimizer: AdamW (bitsandbytes unavailable)")
    scaler = _make_scaler(device, dtype == torch.float16)

    start_step, rng_state = 0, None
    ckpts = sorted(out_dir.glob("controlnet_step_*.pt"))
    if args.resume and ckpts:
        last = ckpts[-1]
        print(f"[train-cnet] resuming from {last.name}")
        state = load_checkpoint(last)
        controlnet.load_state_dict(state["controlnet"])
        opt.load_state_dict(state["opt"])
        scaler.load_state_dict(state["scaler"])
        start_step = state["step"]
        rng_state = state.get("rng")

    if rng_state is not None:
        torch.set_rng_state(rng_state)

    # --- fixed validation batch for visual grids (>=5 comparison PNGs)
    val_batch = next(iter(torch.utils.data.DataLoader(
        val_ds, batch_size=min(args.vis_samples, len(val_ds)), shuffle=False)))

    def lr_at(step: int) -> float:
        warm = min(1.0, step / max(1, args.warmup))
        prog = step / max(1, args.steps)
        return args.lr * warm * (0.5 * (1 + math.cos(math.pi * prog)))

    # --- training loop
    running = 0.0
    data_iter = iter(loader)
    for step in range(start_step, args.steps):
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        phone = batch["phone"].to(device, dtype=dtype, non_blocking=True)
        dslr = batch["dslr"].to(device, dtype=dtype, non_blocking=True)
        b = phone.shape[0]

        noise = torch.randn_like(dslr)
        t = torch.randint(0, noise_sched.config.num_train_timesteps, (b,),
                          device=device, dtype=torch.long)
        noisy = noise_sched.add_noise(dslr, noise, t)

        emb = empty_prompt_embeds(text_encoder, b, device, dtype)
        down, mid = controlnet(noisy, t, encoder_hidden_states=emb,
                               controlnet_cond=phone, return_dict=True)
        noise_pred = unet(noisy, t, encoder_hidden_states=emb,
                          down_block_additional_residuals=down,
                          mid_block_additional_residual=mid).sample
        loss = F.mse_loss(noise_pred.float(), noise.float())

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        scaler.step(opt)
        scaler.update()
        running += loss.item()

        if (step + 1) % args.log_every == 0:
            print(f"[train-cnet] step {step + 1}/{args.steps} loss={running / args.log_every:.4f} lr={lr:.2e}")
            running = 0.0

        if (step + 1) % args.ckpt_every == 0 or (step + 1) == args.steps:
            save_checkpoint(out_dir / f"controlnet_step_{step + 1:08d}.pt",
                            controlnet=controlnet.state_dict(), opt=opt.state_dict(),
                            scaler=scaler.state_dict(), step=step + 1,
                            rng=torch.get_rng_state())
            _keep_last_n(out_dir, "controlnet_step_*.pt", args.keep_ckpts)

        if (step + 1) % args.vis_every == 0 or (step + 1) == args.steps:
            _visualize(unet, controlnet, vae, text_encoder, val_batch, out_dir / "visuals",
                       step + 1, dtype, device, args)

    print("[train-cnet] done.")


def _make_scaler(device: torch.device, enabled: bool):
    """GradScaler across torch versions (new torch.amp API with legacy fallback)."""
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _keep_last_n(out_dir: Path, pattern: str, n: int) -> None:
    ckpts = sorted(out_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    for old in ckpts[:-n]:
        old.unlink()


# --------------------------------------------------------------------------- visualization
@torch.no_grad()
def _visualize(unet, controlnet, vae, text_encoder, val_batch, out_dir: Path, step: int,
               dtype, device, args) -> None:
    from common import decode_latents

    was_training = controlnet.training
    controlnet.eval()
    unet.eval()
    phone = val_batch["phone"].to(device, dtype=dtype)
    dslr = val_batch["dslr"].to(device, dtype=dtype)
    b = phone.shape[0]
    emb = empty_prompt_embeds(text_encoder, b, device, dtype)

    sched = DDPMScheduler.from_pretrained(SD_MODEL_ID, subfolder="scheduler")
    sched.set_timesteps(args.vis_steps, device=device)
    lat = torch.randn(b, LATENT_CHANNELS, 64, 64, device=device, dtype=dtype)
    for t in sched.timesteps:
        tb = torch.full((b,), int(t), device=device, dtype=torch.long)
        down, mid = controlnet(lat, tb, encoder_hidden_states=emb,
                               controlnet_cond=phone, return_dict=True)
        eps = unet(lat, tb, encoder_hidden_states=emb,
                   down_block_additional_residuals=down,
                   mid_block_additional_residual=mid).sample
        lat = sched.step(eps, t, lat).prev_sample

    # latents -> pixels via VAE (phone/dslr val batches are latents too!)
    phone_img = decode_latents(vae, phone.float())
    dslr_img = decode_latents(vae, dslr.float())
    imgs = [phone_img, decode_latents(vae, lat.float()), dslr_img]
    imgs = [(im.clamp(-1, 1) + 1) / 2 for im in imgs]
    labels = ["INPUT (phone)", "OUTPUT (enhanced)", "REAL (DSLR)"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(b):
        grid = make_grid([im[i].cpu().permute(1, 2, 0).numpy() for im in imgs], labels)
        grid.save(out_dir / f"step_{step:06d}_compare_{i}.png")
    print(f"[visualize] {b} comparison PNGs saved -> {out_dir} (step {step})")
    if was_training:
        controlnet.train()


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/processed/iphone")
    ap.add_argument("--out", default="runs/stage_a")
    ap.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--cache-batch", type=int, default=8)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--vis-every", type=int, default=2000)
    ap.add_argument("--vis-steps", type=int, default=30)
    ap.add_argument("--vis-samples", type=int, default=5,
                    help="how many INPUT|OUTPUT|REAL comparison PNGs per visualize call")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--keep-ckpts", type=int, default=3)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
