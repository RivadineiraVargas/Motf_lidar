# MOTF — Project State & Handoff (master context document)

**Last updated:** 2026-08-04 · **Branch:** `encoder/validacao-mae` · **Latest commit:** `bbe3f8d`

This is the single source of truth for the project's current state. Written in
English for efficient AI-assisted continuation. For the granular experiment log
see `docs/EXPERIMENTOS_DECODER.md`; for the advisor's plan see `pedido_claudine.md`
(Portuguese); for Fase-1 (older voxel pipeline) results see `docs/RESULTADOS_ADE_FDE.md`.

---

## 1. What this project is

**MOTF (Moving Object Trajectory Forecasting)** — a master's thesis (LCAD/UFES).
Goal: predict future trajectories of movable objects (cars, pedestrians, cyclists…)
from raw multi-beam LiDAR sweeps, using a two-phase Transformer:

- **Phase 1 (encoder):** self-supervised MAE (Masked Autoencoder, He et al. style)
  pre-trained on LiDAR range-view images. No trajectory labels. Learns scene
  representation by reconstructing masked patches.
- **Phase 2 (decoder):** supervised, Wayformer-style. Takes the frozen encoder's
  scene features + object history → predicts future trajectories.

The advisor (Claudine) reframed the project around the MAE approach — "Sapiens" in
her plan means **the MAE method/codebase**, NOT the human-vision Sapiens model. We
use the Sapiens/`mmpretrain` code as an MAE implementation, trained from scratch on
LiDAR. Communication with the user is in **Spanish**.

---

## 2. Data (`waymo_clean/`)

Source: **Waymo Open Motion Dataset - LiDAR (WOMD-LiDAR)**, re-extracted with
persistent `track.id` (fixes an old index-based association bug).

- **25 scenes**, each with **11 LiDAR frames** (t=0..10) = **275 sweeps total**.
- **Hard dataset limit:** LiDAR exists only for the **first ~1.1 s** of each 9-second
  scene (11 of 91 frames). The other 8 s have trajectory labels (`objs_bbox`) but
  **no point cloud**. This is confirmed by the WOMD-LiDAR paper (Chen et al. 2023:
  "LiDAR points for the first 1 second of each of the 9 second windows") AND by our
  own direct count. **Consequence:** past-LiDAR history is capped at ~1 s; we cannot
  test larger input windows even though Claudine's Section 1 asks for it.
- Layout: `range_files/<scene>/<t>.npy` (64×2650×2 = [range, intensity], float32,
  -1 = no return), `objs_bbox/<scene>/<t>/<track.id>.txt` (8 global bbox corners,
  all 91 frames), `poses/<scene>/<t>.txt` (4×4), `bin_files/` (sparse points).
- `waymo_clean_view/` — full-grid bins (64×2650 ordered, -1 for no-return) for the
  C++ viewer, extracted exactly by `utilities/save_grid_bins_exact.py`. USE THIS
  for the viewer, not `waymo_clean` (whose sparse bins break the viewer's reshape).
- Raw tfrecords available in `waymo_raw/lidar/` + `waymo_raw/scenario/` (25 scenes)
  — for re-extraction (env `waymo_env` has TF + waymo_open_dataset).

---

## 3. Current status vs Claudine's plan

Full checklist: `docs/CHECKLIST_CLAUDINE.md`. Summary:

- **Encoder validation (Sec. 3): DONE for 10 & 100 sweeps.**
  - 10 sweeps: `work_dirs/rv_rect_overfit10/epoch_6000.pth`, loss 2.72 → 0.055.
  - 100 sweeps (multi-scene): `work_dirs/rv_rect_overfit100/epoch_3000.pth`,
    loss 2.07 → 0.244. Key finding: generalization peaks ~epoch 1000 then memorizes.
  - **1000 sweeps: BLOCKED by data** (only 275 sweeps on disk).
- **Architecture reduced (Sec. 4): DONE** — 24→6 layers, embed 1024→384 (~15× smaller).
- **Reconstruction visualization (Sec. 5): DONE** — 5-panel viz + trained-vs-untrained.
- **Wayformer study (Sec. 12): DONE** — `docs/ESTUDIO_WAYFORMER.md`.
- **Decoder (Sec. 13-17): DONE (mini scale).** `train_decoder_mini.py`. Predicts the
  real WOMD format: 8 s future (16 waypoints @2Hz) from 1 s of LiDAR + object history.
- **Simulator (Sec. 11): DONE** — C++ viewer with predictions + GIF exporter, incl.
  bbox projection onto range-view (Gabriel's method, calibrated).

---

## 4. THE CENTRAL RESEARCH QUESTION & the negative finding

**Question:** Does the LiDAR scene (via the frozen MAE encoder) improve trajectory
prediction over a purely kinematic baseline (object history + constant velocity)?

**Answer, as of 2026-08-04: NO, with the current frozen-MAE + 20-scene setup.**
Five independent experiments all confirm it (details: `docs/EXPERIMENTOS_DECODER.md`):

| Experiment | Result |
|---|---|
| 1-2. Cross-validation, raw cross-attn (5 folds × 3 seeds, n=15) | baseline 4.65±1.52 vs wayformer 4.97±1.67 ADE8; **baseline wins, t=-3.07** |
| 3. Pooling bridge (16 latents, Perceiver-style) | 5.00±1.67; ties raw wayformer, still loses to baseline (t=-4.17) |
| 4-5. Partial encoder fine-tuning (unfreeze last block) | mixed: better ADE8 at ep20 but validity accuracy collapses 1.00→0.54 |
| 6. Horizon sweep 1s/3s/5s/8s | scene helps at NO horizon; damage GROWS with horizon (+0.07→+0.38). The Fase-1 "3s sweet spot" did NOT reproduce |

**Important context:** the ORIGINAL positive result ("scene helps, 7.19 vs 7.85 on
one scene") was a single non-representative measurement, **reverted** by cross-validation.
This is a documented example of why the CV protocol was necessary.

**Consistency check:** the WOMD-LiDAR paper reports only marginal ADE improvement from
LiDAR even with ~100k scenes + supervised features — our negative result is in line
with the state of the art, not an artifact of a bug.

---

## 5. KEY INSIGHT — why Fase 1 (voxel) worked and this (frozen MAE) doesn't

This is the most important lead for future work. In **Fase 1** (older pipeline,
`docs/RESULTADOS_ADE_FDE.md`) the scene DID help: **+25% ADE at 3s**, a learnable
"gate" opened to 0.20. Direct comparison even showed **voxel BEAT range-view** at 10
scenes (Val ADE 1.303 vs 1.685) — and Claudine's own plan (Sec. 6) acknowledges this.

Both pipelines use MAE. The difference is NOT "voxel vs MAE" — it's THREE things:

| | Fase 1 (worked) | Current (doesn't) |
|---|---|---|
| Representation | voxels (3D grid) | range-view (2D image) |
| Encoder training | **MAE re-pretrained on the train scenes** | frozen generic MAE |
| Encoder→decoder link | **learnable gate** | fixed residual-over-constant-velocity |

We switched to range-view because Claudine's Sec. 5 wants MAE **reconstruction
visualization** (2D range-view is ideal, like the MAE paper) and we adopted Gabriel
Hendrix's working range-view pipeline. The switch was for encoder-validation reasons,
NOT predictive performance.

**Hypothesis for the next experiment (well-grounded, not yet run):** the ingredients
that made the scene useful in Fase 1 are **portable to range-view without abandoning
Claudine's direction**:
1. **Re-pretrain the MAE on the training scenes (per-fold, to avoid leakage)** — the
   biggest lever; domain adaptation vs generic frozen features.
2. **Add a learnable gate** instead of the fixed residual.

This is a research-direction decision that should be made WITH Claudine, with the
report in hand — not pursued unilaterally.

---

## 6. Environment & how to run

- **GPU env:** conda `sapiens_gpu` (torch 2.5.1+cu118). GPU: RTX 4060 Laptop 8GB.
- **Extraction env:** conda `waymo_env` (TF + waymo_open_dataset).
- **Root for training:** `/home/lcad/lidar_sweep_viewer/sapiens/pretrain/`
- Run decoder CV: `conda run -n sapiens_gpu python cross_validate_decoder.py --enc work_dirs/rv_rect_overfit100/epoch_3000.pth --epochs 100 --archs wayformer baseline`
- Run horizon sweep: `conda run -n sapiens_gpu python horizon_sweep.py --enc work_dirs/rv_rect_overfit100/epoch_3000.pth --epochs 100`
- Viewer (from repo root): `./show_point_cloud --input waymo_clean_view` (key `t`=trajectories, `a`/`d`=frames, `space`=play, `b`=bboxes)
- Regenerate viewer predictions (all 25 scenes): `conda run -n sapiens_gpu python export_decoder_mini_global.py --scenes $(ls ../../waymo_clean/range_files/) --sin-gif --out work_dirs/decoder_abc`

---

## 7. Known traps (do not rediscover these)

- **`MAEViT.eval()` returns `None`** in this fork — never chain `.to(dev).eval()`.
- **`mask=False` is ignored** by the backbone forward — to get unmasked features set
  `encoder.mask_ratio = 0.0` (tokens come permuted but that's fine for cross-attn).
- **GPU non-determinism** in attention ops: even with a fixed seed, results vary
  slightly between runs (exp. 4 vs 5 gave 2.79 vs 2.51 for the same config). Never
  trust a single-seed number; use folds+seeds.
- **`max_keep_ckpts`** silently deletes checkpoints — protect milestones before
  extending a run (lost `epoch_1000.pth` of the 100-sweep run this way).
- **ADE is deflated by parked objects** (~2/3 of objects move <1m in 8s). Report
  moving-only metrics alongside.
- **Data contracts (viewer):** the C++ viewer needs full-grid bins (`waymo_clean_view`),
  and `predictions_global.txt` must cover the scene being viewed (regenerate with all
  25 scenes, else `t` shows nothing).
- **`decoder_pos_embed`** was commented out in Gabriel's `mae_neck.py` (decoder had no
  positional info) — fixed early; if starting fresh from his repo, re-check.

---

## 8. Key files

- `sapiens/pretrain/train_decoder_mini.py` — decoder + **`train_decoder()`** (single
  source of truth for the training loop). Models: `MiniBaseline` (no scene),
  `MiniWayformerDecoder` (raw cross-attn), `MiniWayformerPooled` (16-latent bridge).
  Params: `arch`, `hist`, `finetune_encoder_blocks`, `enc_lr`, `n_wp` (horizon).
- `sapiens/pretrain/cross_validate_decoder.py` — 5-fold × seed CV driver, paired
  comparison + simple t-test, extensible via `--archs`.
- `sapiens/pretrain/horizon_sweep.py` — horizon sweep driver.
- `sapiens/pretrain/export_decoder_mini_global.py` — GIF + viewer-txt exporter,
  incl. calibrated bbox→range-view projection.
- `utilities/save_grid_bins_exact.py` — exact full-grid bin extractor (viewer contract).
- `waymo_clean/beam_inclinations.npy` — per-row beam angles (calibrated from data).
- `docs/EXPERIMENTOS_DECODER.md` — granular experiment log (exp 1-6).
- `docs/CHECKLIST_CLAUDINE.md` — 17-step plan status.
- `docs/ESTUDIO_WAYFORMER.md` — decoder design decisions from Wayformer/WOMD/MTR.

---

## 9. Open decisions / next steps (pending user + advisor)

1. **Write the report** for Claudine — the recommended immediate move. Consolidates
   encoder validation + decoder + the 5-experiment negative finding + the voxel/gate
   lead. This is the artifact that should drive the next research-direction decision.
2. **(With advisor's green light) Re-pretrain the MAE on train scenes + add a gate** —
   the well-grounded experiment to test whether the Fase-1 ingredients recover the
   scene benefit in the range-view pipeline.
3. **1000-sweep encoder step + 50k scale** — blocked by data; needs downloading more
   WOMD-LiDAR (pipeline: `utilities/save_range_view.py` + `range_npy_to_png.py`).

**Constraint agreed with the user:** stay at current data scale; do NOT download new
data or start scale steps without the user's explicit go-ahead (they will get
Claudine's green light first).
