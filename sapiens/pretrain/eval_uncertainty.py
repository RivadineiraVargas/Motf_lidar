"""
eval_uncertainty.py — Evalúa la calidad de la incerteza del modelo (Prioridad 3).

El modelo gated_uncert predice media + log-varianza por cada pose. Aquí medimos:
  1. ADE/FDE (usando la media) — ¿se mantuvo la precisión?
  2. Correlación(std predicho, error real) — ¿el modelo SABE cuándo no sabe?
  3. Calibración — ¿el ~68% de los puntos cae dentro de ±1σ? (~95% en ±2σ?)
  4. ¿La incerteza crece con el paso de predicción? (más lejos = más incierto)

Uso: conda activate sapiens_gpu; cd sapiens/pretrain; python eval_uncertainty.py
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mmengine.runner import load_checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mmpretrain.datasets.trajectory_dataset import TrajectoryDataset
from mmpretrain.models.trajectory_pred.trajectory_model_attn import TrajectoryModelWithAttention
from mmpretrain.models.backbones.mae_vit_4d import MAEViT4D  # noqa

DATA_ROOT = '/home/lcad/lidar_sweep_viewer/waymo_clean'
HISTORY_LEN, PRED_LEN, VOXEL_RES = 5, 30, 2.0
SPATIAL_RANGE, NUM_VOXELS = [-10, 10, -10, 10, -2, 4], 300
VAL_SCENES = ['7e2f727866c69ea0', '82f90331a1dfe968']
CKPT = 'work_dirs/clean10_gated_uncert/epoch_100.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model():
    enc = dict(type='MAEViT4D', history_len=HISTORY_LEN, embed_dim=1024,
               num_tokens=NUM_VOXELS, arch='sapiens_0.3b', final_norm=True, mask_ratio=0.75)
    m = TrajectoryModelWithAttention(encoder=enc, history_len=HISTORY_LEN, pred_len=PRED_LEN,
            embed_dim=1024, num_heads=8, hidden_dim=512, scene_dim=64,
            freeze_encoder=True, use_gate=True, gate_init=0.5, predict_uncertainty=True)
    load_checkpoint(m, CKPT, map_location='cpu'); return m.eval().to(DEVICE)


@torch.no_grad()
def main():
    ds = TrajectoryDataset(data_root=DATA_ROOT, sequence_len=HISTORY_LEN+PRED_LEN,
            history_len=HISTORY_LEN, pred_len=PRED_LEN, voxel_res=VOXEL_RES,
            spatial_range=SPATIAL_RANGE, max_jump=5.0, scenes=VAL_SCENES)
    model = load_model()

    ades, fdes = [], []
    all_err_m, all_std_m = [], []          # error y std en metros, por punto (XY)
    z_scores = []                          # |y-mu|/std normalizado, por coord
    err_by_step = np.zeros(PRED_LEN); std_by_step = np.zeros(PRED_LEN); cnt = 0

    for d in ds:
        if d['scene_name'] not in VAL_SCENES:
            continue
        inp = d['inputs'].unsqueeze(0).to(DEVICE)
        h   = d['obj_history_flat'].unsqueeze(0).to(DEVICE)
        mean, std = d['norm_mean'].numpy(), d['norm_std'].numpy()
        mu, log_var = model(inp, h, mode='uncertainty')
        mu = mu.cpu().view(PRED_LEN, 3).numpy()
        sd_norm = np.exp(0.5 * log_var.cpu().view(PRED_LEN, 3).numpy())
        tgt_norm = d['obj_future_flat'].view(PRED_LEN, 3).numpy()

        # desnormalizar a metros
        mu_m   = mu * std + mean
        tgt_m  = tgt_norm * std + mean
        sd_m   = sd_norm * std                      # std escala por norm_std

        dist = np.linalg.norm(mu_m[:, :2] - tgt_m[:, :2], axis=1)   # error XY por paso (m)
        ades.append(dist.mean()); fdes.append(dist[-1])

        # std XY (promedio de las 2 coords) por paso
        sd_xy = sd_m[:, :2].mean(axis=1)
        all_err_m.extend(dist.tolist()); all_std_m.extend(sd_xy.tolist())
        # z-scores por coordenada (para calibración)
        z = np.abs(tgt_norm - mu) / (sd_norm + 1e-9)
        z_scores.extend(z.reshape(-1).tolist())
        err_by_step += dist; std_by_step += sd_xy; cnt += 1

    all_err_m = np.array(all_err_m); all_std_m = np.array(all_std_m)
    z_scores = np.array(z_scores)
    err_by_step /= cnt; std_by_step /= cnt

    corr = np.corrcoef(all_std_m, all_err_m)[0, 1]
    within1 = float((z_scores < 1).mean()) * 100
    within2 = float((z_scores < 2).mean()) * 100

    print(f'\n{"="*60}')
    print(f'  INCERTEZA — modelo gated_uncert (3s, {cnt} tracks val)')
    print(f'{"="*60}')
    print(f'  Precisión (media):   ADE={np.mean(ades):.3f} m   FDE={np.mean(fdes):.3f} m')
    print(f'  Std predicho medio:  {all_std_m.mean():.3f} m')
    print(f'  ── ¿Sabe cuándo no sabe? ──')
    print(f'  Correlación(std, error real):  {corr:+.3f}   (>0 = el modelo SÍ sabe)')
    print(f'  ── Calibración (ideal: 68% / 95%) ──')
    print(f'  Puntos dentro de ±1σ:  {within1:.1f}%   (ideal ~68%)')
    print(f'  Puntos dentro de ±2σ:  {within2:.1f}%   (ideal ~95%)')
    print(f'  ── ¿La incerteza crece con el horizonte? ──')
    print(f'  Std paso 1 (0.1s): {std_by_step[0]:.3f} m  ->  paso {PRED_LEN} (3s): {std_by_step[-1]:.3f} m')
    print(f'{"="*60}')

    # Gráficos
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(all_std_m, all_err_m, s=6, alpha=.3, color='#36c')
    ax[0].set_xlabel('Std predicho (m)'); ax[0].set_ylabel('Error real (m)')
    ax[0].set_title(f'¿Sabe cuándo no sabe?  corr={corr:+.2f}'); ax[0].grid(alpha=.3)
    steps = (np.arange(PRED_LEN) + 1) / 10
    ax[1].plot(steps, err_by_step, 'o-', label='Error real', color='#888')
    ax[1].plot(steps, std_by_step, 's-', label='Std predicho', color='#2a7')
    ax[1].set_xlabel('Horizonte (s)'); ax[1].set_ylabel('metros')
    ax[1].set_title('Incerteza vs error a lo largo del horizonte')
    ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig('incerteza.png', dpi=120)
    print(f'  Gráfico: incerteza.png\n')


if __name__ == '__main__':
    main()
