"""Stage B: full fine-tune of SD 2.1-base UNet for DPED iPhone -> DSLR enhancement.

Initializes ControlNet from a Stage A checkpoint (frozen by default; --train-controlnet
to unfreeze it as well). Adds EMA, gradient accumulation and gradient checkpointing.

Usage:
  python src/finetune_unet.py --init-controlnet runs/stage_a/controlnet_step_0015000.pt --steps 6000
"""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler

from common import (LATENT_CHANNELS, SD_MODEL_ID, EMA, LatentPairDataset, auto_dtype,
                    load_checkpoint, save_checkpoint, seed_everything)
from train_controlnet import (_keep_last_n, _make_scaler, _visualize, empty_prompt_embeds,
                              load_models)


def build_controlnet_from_ckpt(ckpt_path: str | Path, dtype, device):
    """Create a ControlNet with SD2.1-base UNet config and load Stage A weights."""
    from diffusers import ControlNetModel, UNet2DConditionModel

    unet_ref = UNet2DConditionModel.from_pretrained(SD_MODEL_ID, subfolder="unet",
                                                    torch_dtype=dtype)
    controlnet = ControlNetModel.from_unet(unet_ref)
    del unet_ref
    state = load_checkpoint(ckpt_path)
    sd = state["controlnet"] if "controlnet" in state else state
    controlnet.load_state_dict(sd, strict=True)
    return controlnet.to(device)


# --------------------------------------------------------------------------- train loop
def train(args) -> None:
    from common import build_latent_cache, load_vae
    from common import PairedCropDataset

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[train-unet] WARNING: no CUDA device, running CPU smoke-test mode")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- data
    cache_dir = Path(args.data) / "latent_cache"
    try:
        train_ds = LatentPairDataset(cache_dir, "train")
        val_ds = LatentPairDataset(cache_dir, "val")
        print(f"[train-unet] latent cache loaded: train={len(train_ds)} val={len(val_ds)}")
    except (FileNotFoundError, AssertionError):
        print("[train-unet] latent cache missing -> building from crops")
        vae = load_vae(args.precision, device)
        crop_train = Path(args.data) / "crops" / "train"
        crop_val = Path(args.data) / "crops" / "val"
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
    if not args.init_controlnet:
        raise SystemExit("[train-unet] --init-controlnet is required (path to Stage A checkpoint)")
    controlnet = build_controlnet_from_ckpt(args.init_controlnet, dtype, device)

    unet.requires_grad_(True)
    unet.enable_gradient_checkpointing()
    if args.train_controlnet:
        controlnet.requires_grad_(True)
        trainable_modules = [unet, controlnet]
    else:
        controlnet.requires_grad_(False)
        trainable_modules = [unet]
    unet.train()
    controlnet.train()

    vae = load_vae(args.precision, device)
    noise_sched = DDPMScheduler.from_pretrained(SD_MODEL_ID, subfolder="scheduler")

    # --- optimizer over all trainable params
    groups = []
    for m in trainable_modules:
        groups += [p for p in m.parameters() if p.requires_grad]
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(groups, lr=args.lr_start, weight_decay=args.weight_decay)
        print("[train-unet] optimizer: AdamW8bit")
    except Exception:
        opt = torch.optim.AdamW(groups, lr=args.lr_start, weight_decay=args.weight_decay)
        print("[train-unet] optimizer: AdamW")
    scaler = _make_scaler(device, dtype == torch.float16)

    ema = EMA(unet, args.ema_decay, device)
    if args.train_controlnet:
        ema_c = EMA(controlnet, args.ema_decay, device)
    else:
        ema_c = None

    start_step = 0
    if args.resume:
        ckpts = sorted(out_dir.glob("finetune_step_*.pt"))
        if ckpts:
            last = ckpts[-1]
            print(f"[train-unet] resuming from {last.name}")
            state = load_checkpoint(last)
            unet.load_state_dict(state["unet"])
            controlnet.load_state_dict(state["controlnet"])
            opt.load_state_dict(state["opt"])
            scaler.load_state_dict(state["scaler"])
            ema.load_state_dict(state["ema"], device)
            if ema_c is not None and state.get("ema_controlnet"):
                ema_c.load_state_dict(state["ema_controlnet"], device)
            start_step = state["step"]
            if state.get("rng") is not None:
                torch.set_rng_state(state["rng"])

    # --- fixed validation batch (>=5 comparison PNGs)
    val_batch = next(iter(torch.utils.data.DataLoader(
        val_ds, batch_size=min(args.vis_samples, len(val_ds)), shuffle=False)))

    def lr_at(step: int) -> float:
        warm = min(1.0, step / max(1, args.warmup))
        prog = step / max(1, args.steps)
        cos = 0.5 * (1 + math.cos(math.pi * prog))
        lr = args.lr_end + (args.lr_start - args.lr_end) * cos
        return lr * warm

    # --- training loop with grad accumulation
    running = 0.0
    micro = 0
    data_iter = iter(loader)
    opt.zero_grad(set_to_none=True)
    for step in range(start_step, args.steps):
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr

        accum_loss = 0.0
        for _ in range(args.accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            phone = batch["phone"].to(device, dtype=dtype, non_blocking=True)
            dslr = batch["dslr"].to(device, dtype=dtype, non_blocking=True)
            b = phone.shape[0]
            emb = empty_prompt_embeds(text_encoder, b, device, dtype)

            noise = torch.randn_like(dslr)
            t = torch.randint(0, noise_sched.config.num_train_timesteps, (b,),
                              device=device, dtype=torch.long)
            noisy = noise_sched.add_noise(dslr, noise, t)

            with torch.autocast(device_type=device.type, dtype=dtype,
                                enabled=(dtype != torch.float32)):
                down, mid = controlnet(noisy, t, encoder_hidden_states=emb,
                                       controlnet_cond=phone, return_dict=True)
                noise_pred = unet(noisy, t, encoder_hidden_states=emb,
                                  down_block_additional_residuals=down,
                                  mid_block_additional_residual=mid).sample
            loss = F.mse_loss(noise_pred.float(), noise.float()) / args.accum
            scaler.scale(loss).backward()
            accum_loss += loss.item()
            micro += 1

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(groups, args.max_grad_norm)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        ema.update(unet)
        if ema_c is not None:
            ema_c.update(controlnet)

        running += accum_loss / args.accum
        if (step + 1) % args.log_every == 0:
            print(f"[train-unet] step {step + 1}/{args.steps} "
                  f"loss={running / args.log_every:.4f} lr={lr:.2e}")
            running = 0.0

        if (step + 1) % args.ckpt_every == 0 or (step + 1) == args.steps:
            save_checkpoint(out_dir / f"finetune_step_{step + 1:08d}.pt",
                            unet=unet.state_dict(), controlnet=controlnet.state_dict(),
                            opt=opt.state_dict(), scaler=scaler.state_dict(),
                            ema=ema.state_dict(),
                            ema_controlnet=(ema_c.state_dict() if ema_c else None),
                            step=step + 1, rng=torch.get_rng_state())
            _keep_last_n(out_dir, "finetune_step_*.pt", args.keep_ckpts)

        if (step + 1) % args.vis_every == 0 or (step + 1) == args.steps:
            # visualize with EMA weights for an honest preview
            backup_u = ema.copy_into(unet)
            backup_c = ema_c.copy_into(controlnet) if ema_c else None
            _visualize(unet, controlnet, vae, text_encoder, val_batch, out_dir / "visuals",
                       step + 1, dtype, device, args)
            ema.restore(unet, backup_u)
            if ema_c is not None:
                ema_c.restore(controlnet, backup_c)

    print("[train-unet] done.")


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/processed/iphone")
    ap.add_argument("--out", default="runs/stage_b")
    ap.add_argument("--init-controlnet", default="", help="Stage A controlnet .pt checkpoint")
    ap.add_argument("--train-controlnet", action="store_true",
                    help="also fine-tune the ControlNet (more VRAM)")
    ap.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2, help="grad accumulation -> effective batch")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr-start", type=float, default=5e-5)
    ap.add_argument("--lr-end", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--cache-batch", type=int, default=8)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--vis-every", type=int, default=1000)
    ap.add_argument("--vis-steps", type=int, default=30)
    ap.add_argument("--vis-samples", type=int, default=5)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--keep-ckpts", type=int, default=3)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
