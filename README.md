# DPED iPhone → DSLR-Quality Enhancement (Latent Diffusion)

SD 2.1-base latent diffusion ko DPED iPhone subset pe adapt karke phone photos ko
DSLR-quality me badalta hai. Do-stage training: **Stage A** = frozen backbone pe
ControlNet adapter, **Stage B** = full UNet fine-tune (EMA ke saath). Free Kaggle
GPU (T4 x2 / P100, 16GB) pe chalne ke liye designed: latent caching, resumable
checkpoints, session chaining.

## Setup

```bash
pip install -r requirements.txt
```

Local (CPU) me sirf data prep + smoke tests chalte hain; training/inference ke liye GPU chahiye.

## Pipeline (run order)

### 0. Data prep
```bash
python src/prepare_data.py --verify-only        # structure inspect (koi download nahi)
python src/prepare_data.py --source auto        # kagglehub, fail hone pe official DPED
```
Output `data/processed/iphone/` me: `crops/{train,val}/{phone,dslr}/*.png`
(512px aligned crops), `test/{phone,dslr}/*.jpg` (full-size DPED test images),
`manifest.json`.

### 1. Stage A — ControlNet
```bash
python src/train_controlnet.py --steps 15000 --batch 16 --resume
```
- Conditioning: phone-image VAE latents (`controlnet_cond`) + fixed empty-prompt embeds
- Checkpoints `runs/stage_a/controlnet_step_XXXXXXXX.pt`, visuals `runs/stage_a/visuals/`
- `--resume` kisi bhi session me wapas utha leta hai

### 2. Stage B — UNet fine-tune
```bash
python src/finetune_unet.py \
  --init-controlnet runs/stage_a/controlnet_step_0015000.pt \
  --steps 6000 --batch 8 --accum 2 --resume
```
- EMA weights checkpoint ke andar save hote hain; visuals EMA weights se render hote hain
- `--train-controlnet` se ControlNet bhi saath me tune hota hai (zyada VRAM)

### 3. Inference
```bash
# val crops -> side-by-side grids
python src/infer.py --ckpt runs/stage_b/finetune_step_0006000.pt --mode crops --ema

# full-res DPED test images (tiled, feathered blending) + diffusers export
python src/infer.py --ckpt runs/stage_b/finetune_step_0006000.pt --mode full \
  --ema --num 29 --export runs/export
```
`--strength` (default 0.85) SDEdit noise level control karta hai — kam = zyada
faithful to phone, zyada = zyada "DSLR-fication".

### 4. Metrics
```bash
python src/metrics.py --gt data/processed/iphone/crops/val/dslr --pred runs/infer --lpips
```
PSNR + SSIM (Y-channel) hamesha; LPIPS optional (`pip install lpips`).

## Kaggle notebooks

`kaggle/stage_a_controlnet.ipynb` aur `kaggle/stage_b_finetune.ipynb` —
cell-by-cell setup same as above, with session chaining:

1. **Session 1 (Stage A)**: notebook run karo. End me *Save Version* (Save & Run All).
2. Output ko **New Dataset** banao (`runs/stage_a` checkpoints + `latent_cache`).
3. **Session 2 (Stage B)**: us dataset ko *Add Input* se attach karo, stage_b notebook
   chalao. Resume logic `runs/stage_b` me khud latest checkpoint utha lega.
4. Repeat for longer Stage B training — har 1k steps pe checkpoint milta hai.

Tips:
- Kaggle me `KAGGLE_USERNAME`/`KAGGLE_KEY` secrets Add-ons → Secrets me daalo
  (kagglehub DPED download ke liye) — ya `--source official` use karo.
- Internet ON rakho (pip + SD 2.1-base + tokenizer downloads).
- 12h limit se pehle rukna ho to bhi koi baat nahi: last checkpoint se resume hoga.
- Working dir quota 20GB/output; `latent_cache` (~1-2GB) + checkpoints ke sath fit hota hai.
  Purane checkpoints script khud delete karta hai (`--keep-ckpts`).

## Architecture notes

- **Why ControlNet first?** Backbone frozen rehta hai → memory-safe, fast, aur
  zero-conv init training start me output ko nahi bigaadta.
- **Why fine-tune after?** ControlNet-only limited capacity hai; Stage B UNet ko
  bhi DPED distribution pe adapt karta hai — final quality ke liye.
- **Latent caching**: sab crops ek baar VAE se encode → disk pe `.pt` shards.
  Training epochs me VAE forward cast nahi hota (3-5x speedup).
- **Empty-prompt conditioning**: model sirf image condition se seekhta hai; text
  branch sirf fixed unconditional embeds deta hai (CFG ki zaroorat nahi).

## Data caveats (DPED)

- Pairs SIFT-aligned hain par perfect nahi — slight misregistration edges pe
  dikhta hai. Crop extraction me sharpness filter + margin strategy iska
  mitigation karta hai; diffusion target me ye minor noise ke roop me hi rehta hai.
- Official patch release me patches 100x100 hote hain — prepare_data inko
  automatically 512 tak upsample karta hai (LANCZOS, warning ke saath). Tone/color
  enhancement signal upsample me survive karta hai, fine detail limited rehta hai.
  Agar aapke paas full-res DPED training images hain (ya Kaggle mirror me milein),
  to wahi windowed 512 crops use honge (`--verify-only` me source image sizes
  check kar sakte ho).

## Troubleshooting

- **CUDA OOM (Stage B)**: `--batch 4 --accum 4` karo, ya `--train-controlnet` hatao.
- **bitsandbytes install fail (Kaggle)**: harmless — script AdamW pe fallback karta hai.
- **T4 pe bf16 error**: `--precision fp16` default hai hi; bf16 sirf A100/Volta+ pe.
- **kagglehub 401**: secrets missing hain → `--source official`.
- **prepare_data "could not locate patch dirs"**: `--verify-only` chala ke printed
  tree dekho; Kaggle mirrors kabhi structure shift karte hain — `PATCH_DIR_NAMES`
  ko `src/prepare_data.py` me adjust karo.

## Citation

```
@inproceedings{ignatov2017dslr,
  title={DSLR-Quality Photos on Mobile Devices with Deep Convolutional Networks},
  author={Ignatov, Andrey and Kobyshev, Nikolay and Timofte, Radu and Vanhoey, Kenneth and Van Gool, Luc},
  booktitle={ICCV},
  year={2017}
}
```
