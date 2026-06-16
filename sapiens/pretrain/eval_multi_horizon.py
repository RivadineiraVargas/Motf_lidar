"""
eval_multi_horizon.py — Curva del beneficio de la escena vs horizonte de predicción.

Para cada pred_len (10/20/30/50 = 1/2/3/5s) evalúa baseline y gated_init (gate_init=0.5,
encoder limpio) en las 2 escenas de val, y reporta Val ADE/FDE + la mejora de la escena.
Genera una tabla y un gráfico PNG (curva del % de mejora vs horizonte).

Uso: conda activate sapiens_gpu; cd sapiens/pretrain; python eval_multi_horizon.py
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
from mmpretrain.models.trajectory_pred.baseline_model import BaselineTrajectoryModel
from mmpretrain.models.trajectory_pred.trajectory_model_attn import TrajectoryModelWithAttention
from mmpretrain.models.backbones.mae_vit_4d import MAEViT4D  # noqa

DATA_ROOT = '/home/lcad/lidar_sweep_viewer/waymo_clean'
HISTORY_LEN, VOXEL_RES = 5, 2.0
SPATIAL_RANGE, NUM_VOXELS = [-10, 10, -10, 10, -2, 4], 300
VAL_SCENES = {'7e2f727866c69ea0', '82f90331a1dfe968'}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# pred_len -> (work_dir baseline, work_dir gated). p30 son los nombres de Fase 1.
HORIZONS = {
    10: ('clean10_baseline_p10', 'clean10_gated_init_p10'),
    20: ('clean10_baseline_p20', 'clean10_gated_init_p20'),
    30: ('clean10_baseline',     'clean10_gated_init'),
    50: ('clean10_baseline_p50', 'clean10_gated_init_p50'),
}


def load_baseline(ckpt, pred_len):
    m = BaselineTrajectoryModel(history_len=HISTORY_LEN, pred_len=pred_len, hidden_dim=512)
    load_checkpoint(m, ckpt, map_location='cpu'); return m.eval().to(DEVICE)


def load_gated(ckpt, pred_len):
    enc = dict(type='MAEViT4D', history_len=HISTORY_LEN, embed_dim=1024,
               num_tokens=NUM_VOXELS, arch='sapiens_0.3b', final_norm=True, mask_ratio=0.75)
    m = TrajectoryModelWithAttention(encoder=enc, history_len=HISTORY_LEN, pred_len=pred_len,
            embed_dim=1024, num_heads=8, hidden_dim=512, scene_dim=64,
            freeze_encoder=True, use_gate=True, gate_init=0.5)
    load_checkpoint(m, ckpt, map_location='cpu'); return m.eval().to(DEVICE)


@torch.no_grad()
def val_metrics(model, dataset, pred_len, is_attn):
    ades, fdes = [], []
    for d in dataset:
        if d['scene_name'] not in VAL_SCENES:
            continue
        inp = d['inputs'].unsqueeze(0).to(DEVICE)
        h   = d['obj_history_flat'].unsqueeze(0).to(DEVICE)
        fut = d['obj_future_flat']
        mean, std = d['norm_mean'], d['norm_std']
        pred = model(inp, h, mode='predict') if is_attn else model(h, mode='predict')
        pred = (pred.cpu().view(pred_len, 3) * std + mean).numpy()
        tgt  = (fut.view(pred_len, 3) * std + mean).numpy()
        dist = np.linalg.norm(pred[:, :2] - tgt[:, :2], axis=1)
        ades.append(dist.mean()); fdes.append(dist[-1])
    return float(np.mean(ades)), float(np.mean(fdes))


def main():
    hs, b_ade, g_ade, imp = [], [], [], []
    print(f'\n{"="*64}')
    print(f'  CURVA MULTI-HORIZONTE — Val ADE (10 escenas, 2 val)')
    print(f'{"="*64}')
    print(f'  {"Horizonte":>10}  {"Baseline":>10}  {"Gated":>10}  {"Mejora":>8}')
    print(f'  {"-"*10}  {"-"*10}  {"-"*10}  {"-"*8}')
    for P in sorted(HORIZONS):
        bdir, gdir = HORIZONS[P]
        bck = f'work_dirs/{bdir}/epoch_100.pth'
        gck = f'work_dirs/{gdir}/epoch_100.pth'
        if not (os.path.exists(bck) and os.path.exists(gck)):
            print(f'  {P/10:>8.0f}s  (falta checkpoint, omitido)')
            continue
        ds = TrajectoryDataset(data_root=DATA_ROOT, sequence_len=HISTORY_LEN+P,
                history_len=HISTORY_LEN, pred_len=P, voxel_res=VOXEL_RES,
                spatial_range=SPATIAL_RANGE, max_jump=5.0, scenes=list(VAL_SCENES))
        ba, _ = val_metrics(load_baseline(bck, P), ds, P, False)
        ga, _ = val_metrics(load_gated(gck, P),   ds, P, True)
        pct = (ba - ga) / ba * 100
        hs.append(P/10); b_ade.append(ba); g_ade.append(ga); imp.append(pct)
        print(f'  {P/10:>8.0f}s  {ba:>8.3f}m  {ga:>8.3f}m  {pct:>6.1f}%')
    print(f'{"="*64}')

    if len(hs) >= 2:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(hs, b_ade, 'o-', label='Baseline (solo histórico)', color='#888')
        ax[0].plot(hs, g_ade, 's-', label='Gated (con escena LiDAR)', color='#2a7')
        ax[0].set_xlabel('Horizonte de predicción (s)'); ax[0].set_ylabel('Val ADE (m)')
        ax[0].set_title('Error vs horizonte'); ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].plot(hs, imp, 'D-', color='#d62')
        ax[1].axhline(0, color='k', lw=.5)
        ax[1].set_xlabel('Horizonte de predicción (s)')
        ax[1].set_ylabel('Mejora de la escena (% ADE)')
        ax[1].set_title('Beneficio de la escena LiDAR vs horizonte'); ax[1].grid(alpha=.3)
        plt.tight_layout(); plt.savefig('curva_multi_horizonte.png', dpi=120)
        print(f'  Gráfico guardado: curva_multi_horizonte.png\n')


if __name__ == '__main__':
    main()
