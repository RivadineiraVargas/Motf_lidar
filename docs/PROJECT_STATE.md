# MOTF — Project State & Handoff (master context document)

**Last updated:** 2026-08-19 · **Branch:** `encoder/validacao-mae` · **Latest commit:** `4458707`

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
- **Decoder (Sec. 13-17): DONE (mini scale) + fully validated.** `train_decoder_mini.py`.
  Predicts the real WOMD format (16 waypoints @2Hz = 8 s; sweeps also run at 1/3/5 s)
  from 1 s of LiDAR + object history. Validated with 5-fold CV × 8 seeds and an
  architecture-matched control — see §4.
- **Simulator (Sec. 11): DONE** — C++ viewer with predictions + GIF exporter, incl.
  bbox projection onto range-view (Gabriel's method, calibrated).

---

## 4. THE CENTRAL RESEARCH QUESTION — answered, and reframed

**Original question:** does the LiDAR scene (via the MAE encoder) improve trajectory
prediction over a purely kinematic baseline (history + constant velocity)?

**Answer as of 2026-08-19: NO — and the question itself was mis-posed.** Fourteen
experiments (`docs/EXPERIMENTOS_DECODER.md`). Two results carry the conclusion:

### 4.1 Full 5-fold cross-validation (exp. 11) — the effect is split-dependent

One domain-adapted MAE encoder per fold (~12.5 h each), decoder at 3 s, 8 seeds:

| fold | diff way−base | t | seeds | relative |
|---|---|---|---|---|
| 0 | −0.186 ± 0.089 | −5.94 | 8/8 | −20.4% |
| 1 | −0.061 ± 0.074 | −2.35 | 6/8 | −4.9% |
| 2 | +0.086 ± 0.084 | +2.89 | 1/8 | +7.9% |
| 3 | +0.570 ± 0.130 | +12.40 | 0/8 | +40.0% |
| 4 | −0.024 ± 0.115 | −0.59 | 4/8 | −1.3% |

**Between folds: +0.077 ± 0.292, t=0.589, df=4, NOT significant**, CI95
[−0.286, +0.439], 3/5 folds in favour. sd between folds (0.292) is **3×** sd between
seeds (0.098): with 25 scenes the dominant variable is *which scenes land in the split*.

### 4.2 Frozen-gate sweep (exp. 14) — THE CONTROL THAT WAS MISSING

Freezing the scene gate at fixed values gives the dose–response curve, and
`gatefix0.0` (scene fully off) should reproduce the baseline. **It does not** — and
not because of a bug: `MiniBaseline` runs an **independent MLP per object**, while the
gated model keeps **self-attention BETWEEN objects**, 2 layers and an FFN. The
"wayformer vs baseline" comparison that the whole project rested on never isolated the
scene: it measured scene + decoder capacity + agent interaction, together.

Decomposition (8 seeds):

| component | fold 0 | fold 3 |
|---|---|---|
| **architecture** (gatefix0.0 vs baseline, NO scene) | **−0.129** t=−9.19 8/8 | **+0.578** t=+9.15 0/8 |
| **scene** (learned gate vs gate 0, same arch.) | −0.023 t=−1.80 (ns) | +0.121 t=+1.91 (ns) |
| historically reported total | −0.186 | +0.570 |

Architecture accounts for **69%** of the fold-0 effect and **101%** of fold-3. What is
left for the scene is not significant in either. And the curve is **flat**: from 0% to
99% scene the ADE never moves outside noise (the single significant cell, fold 3 @0.5,
runs *against* the scene, is 1 of 10 comparisons — what chance predicts at 5% — and is
non-monotonic). **There is no dose–response relationship.**

### 4.3 What this means

1. **The LiDAR scene contributes nothing** — with no encoder (generic or
   domain-adapted), no bridge (raw cross-attn, pooling, fine-tuning), no horizon
   (1/3/5/8 s), and no **dose** (0–99%).
2. **What depended on the split was never the scene: it was the ARCHITECTURE.** A
   2-layer transformer decoder trained on 20 scenes gains 14% on one split and loses
   39% on another. This resolves the puzzle open since exp. 8, and is consistent with
   `best_ep`=1 on fold 3 (it overfits from the first epoch).
3. **Reframed thesis contribution** — from a bare negative to a methodological claim
   with evidence: *with 25 scenes the decoder's capacity dominates any effect of the
   scene features, and the standard "with-LiDAR vs simple-baseline" comparison is
   confounded with model capacity; controlling for architecture, the scene's
   contribution is indistinguishable from zero across the whole range.* This applies
   to any work making that comparison without an architecture-matched control.

**Consistency check:** the WOMD-LiDAR paper reports only marginal ADE gains from LiDAR
even with ~100k scenes and supervised features — the negative result is in line with
the state of the art, not a bug.

**Secondary result worth reporting:** the *learned* gate converges to **0.0968 ± 0.0135**
across all 5 folds from 40 initialisations at 0.5 — by far the most reproducible
quantity in the project, even though (per 4.2) that amount of scene buys nothing.

---

## 5. The Fase-1 hypothesis WAS tested — and it did not hold

Earlier versions of this document proposed, as the best-grounded next experiment,
porting the two ingredients that made the scene useful in **Fase 1** (voxel pipeline,
+25% ADE at 3 s): (a) re-pretrain the MAE on the training scenes, (b) add a learnable
gate. **Both were run. Neither recovers the benefit.**

| ingredient | experiment | outcome |
|---|---|---|
| Domain-adapted encoder (per fold, no leakage) | 7-8, 11 | looked decisive on fold 0 (−20.4%, p=0.0006, 8/8 seeds) — **evaporated** across 5 folds |
| Learnable gate | 12 | ties the ungated model on fold 0, **loses in 4/5 folds**; does not rescue fold 3 (+40% → +49%) |
| 3 s "sweet spot" | 9, 11 | reproduced on fold 0 only; not a property of the method |

**Why the gate failed, mechanically (exp. 12):** `best_ep`=1 in 6/8 seeds on fold 3,
and at epoch 1 the gate is still ~0.497 (init 0.5). Early stopping freezes the model
*before* the valve closes, so the evaluated checkpoint uses the scene at nearly full
strength precisely where the scene is harmful. The gate learns to close, but too late
for early stopping to benefit. In Fase 1 the decoder was an MLP trained for many more
epochs, which is why the gate helped there.

**Why Fase 1 looked better at all:** its comparison had the same architecture confound
identified in 4.2, plus it was a single split. Nothing in Fase 1 was validated across
folds.

---

## 6. Environment & how to run

- **GPU env:** conda `sapiens_gpu` (torch 2.5.1+cu118). GPU: RTX 4060 Laptop 8GB.
- **Extraction env:** conda `waymo_env` (TF + waymo_open_dataset).
- **Root for training:** `/home/lcad/lidar_sweep_viewer/sapiens/pretrain/`
- **Encoders:** generic 100-sweep = `work_dirs/rv_rect_overfit100/epoch_3000.pth`;
  **domain-adapted, one per fold** = `work_dirs/rv_rect_fold{0..4}/epoch_1000.pth`
  (each pretrained ONLY on that fold's 20 training scenes — using one outside its own
  fold is leakage). Cached features: `work_dirs/cache_fold{0..4}_domain`.
- Decoder CV: `conda run -n sapiens_gpu python cross_validate_decoder.py --enc <encoder> --epochs 100 --archs wayformer baseline`
- Sweep (horizon / architecture): `conda run -n sapiens_gpu python horizon_sweep.py --enc work_dirs/rv_rect_fold3/epoch_1000.pth --folds 3 --seeds 0 1 2 3 4 5 6 7 --horizons 3s --archs wayformer baseline --cache work_dirs/cache_fold3_domain --out work_dirs/horizon_fold3 --epochs 100`
- Frozen-gate sweep: `bash run_gate_sweep.sh` · Direction/magnitude: `python angular_error_analysis.py` · Latency: `python latency_benchmark.py`
- Viewer (from repo root): `./show_point_cloud --input waymo_clean_view` (`t`=trajectories,
  `a`/`d`=frames, `space`=play, `b`=bboxes). **`waymo_clean_view`, NOT `waymo_clean`** —
  the latter's sparse bins break the viewer's `reshape(64,2650)`.
- Regenerate viewer predictions for **all 25 scenes without leakage** — each scene is
  predicted by the model of the fold that held it out (folds partition the 25 into
  disjoint groups), using per fold the seed whose ADE is closest to the 8-seed mean
  (f0 s5, f1 s0, f2 s6, f3 s2, f4 s7):
  ```
  for F in 0 1 2 3 4; do python export_decoder_mini_global.py \
      --enc-cfg configs/sapiens_mae/lidar/config_rangeview_rect_fold${F}.py \
      --enc-ckpt work_dirs/rv_rect_fold${F}/epoch_1000.pth \
      --dec <run_dir>/wayformer_h3s_f${F}s<seed>/decoder_mini.pth \
      --n-wp 6 --sin-gif --scenes <the 5 held-out scenes of fold F> \
      --out work_dirs/sim_cv_completa --txt /tmp/pred_f${F}.txt; done
  cat /tmp/pred_f?.txt > predictions_global.txt
  ```
  The viewer reads `predictions_global.txt` by fixed name from the cwd. Prediction
  `.txt` files are gitignored (regenerable artefacts).

---

## 7. Known traps (do not rediscover these)

- **`MAEViT.eval()` returns `None`** in this fork — never chain `.to(dev).eval()`.
- **`mask=False` is ignored** by the backbone forward — to get unmasked features set
  `encoder.mask_ratio = 0.0`. **The returned tokens are PERMUTED on every call**, so two
  fp32 calls on the same input differ by 69.3% relative error elementwise (0.000% with
  `torch.manual_seed` fixed). It is exactly a permutation of the same token set (max
  1.5e-5 on sorted per-token sums), harmless because cross-attention is
  permutation-invariant over memory — but it **invalidates any elementwise comparison of
  encoder outputs without a fixed seed**. Without that control, autocast fp16 appears to
  break the model (70% error) when its real error is 0.046%.
- **GPU non-determinism** in attention ops: even with a fixed seed, results vary
  slightly between runs (exp. 4 vs 5 gave 2.79 vs 2.51 for the same config).
- **NEVER report a finding from one seed or one fold.** Four retracted claims so far:
  (1) "scene helps" 7.19 vs 7.85 from one scene → reverted by CV; (2) "−20.4% at 3 s,
  p=0.0006, 8/8 seeds" from one fold → evaporated across 5 folds; (3) "fold 0 has fewer
  gross direction errors" from 1 seed → a tie with 8; (4) "better magnitude calibration"
  → artefact of averaging parked objects in. Always state n (seeds AND folds) and the
  population averaged over, *before* the table.
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
  `MiniWayformerDecoder` (raw cross-attn), `MiniWayformerPooled` (16-latent bridge),
  `MiniWayformerGated` (learnable gate) and `gatefix<v>` (gate frozen at v).
  NOTE: `MiniBaseline` is an independent per-object MLP — it is NOT an
  architecture-matched control for the scene; use `gatefix0.0` for that (see §4.2).
  Params: `arch`, `hist`, `finetune_encoder_blocks`, `enc_lr`, `n_wp` (horizon).
- `sapiens/pretrain/cross_validate_decoder.py` — 5-fold × seed CV driver, paired
  comparison + simple t-test, extensible via `--archs`.
- `sapiens/pretrain/horizon_sweep.py` — horizon/arch sweep driver (`--folds`, `--seeds`,
  `--horizons`, `--archs`; appends to CSV and skips already-done cells, so it is
  resumable). `--folds` is MANDATORY with a domain encoder: using it outside its own
  fold is leakage.
- `sapiens/pretrain/angular_error_analysis.py` — splits the error into DIRECTION vs
  MAGNITUDE over moving objects only (exp. 13).
- `sapiens/pretrain/latency_benchmark.py` — per-sweep latency, fp32 vs autocast fp16
  (bridge to stage 2).
- `sapiens/pretrain/run_gate_sweep.sh` — frozen-gate dose–response sweep (exp. 14);
  arch `gatefix<v>` freezes `scene_gate` at v.
- `sapiens/pretrain/export_decoder_mini_global.py` — GIF + viewer-txt exporter,
  incl. calibrated bbox→range-view projection.
- `utilities/save_grid_bins_exact.py` — exact full-grid bin extractor (viewer contract).
- `waymo_clean/beam_inclinations.npy` — per-row beam angles (calibrated from data).
- `docs/EXPERIMENTOS_DECODER.md` — granular experiment log (exp 1-14).
- `docs/CHECKLIST_CLAUDINE.md` — 17-step plan status.
- `docs/ESTUDIO_WAYFORMER.md` — decoder design decisions from Wayformer/WOMD/MTR.

---

## 9. Open decisions / next steps

**Experimental work on the decoder line is CLOSED.** The cheap levers are exhausted;
every angle has been measured with folds + seeds. What remains:

1. **Write the report — the actual deliverable.** It is now a strong artifact: a
   negative result hardened by 5 folds × 8 seeds, a reproducible quantitative result
   (learned gate → 0.0968 ± 0.0135 across 5 splits), an architecture-confound finding
   that reframes the question (4.2), and a methodological lesson with three documented
   instances (§7).
2. **Stage 2 — deploy on the LCAD vehicle.** The declared final goal. Integration point
   is `lidar_sweep_viewer_main.cpp` (astro/velodyne middleware, live messages — NOT a
   file viewer; `show_point_cloud.cpp` is the offline one). Latency is measured and the
   pipeline fits real time **only with mixed precision**: fp32 141.6 ms/sweep (7.1 Hz)
   vs autocast fp16 ~29 ms (~34 Hz, 0.046% error). The encoder is 98% of the compute.
   Open work: recalibrate for the lab's sensor geometry (the model assumes Waymo's
   64-beam layout, see `beam_inclinations.npy`).
3. **Scaling the data (1000 sweeps / 50k)** — the only path that could revive the scene
   hypothesis, and a project in itself. Blocked on downloading more WOMD-LiDAR.

**Explicitly NOT recommended:** re-training the encoder on all 275 sweeps instead of
100. Exp. 11 shows that at this scale the split dominates over encoder size; ~12 h for
a result that would likely fall inside between-fold noise.

**Constraint agreed with the user:** stay at the current data scale; do NOT download
new data or start scale steps without explicit go-ahead.
