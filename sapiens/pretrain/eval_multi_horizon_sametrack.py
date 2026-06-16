"""
eval_multi_horizon_sametrack.py — Curva multi-horizonte PAREADA (Opción 2).

Problema de la versión ingenua: a cada horizonte sobreviven tracks distintos
(117 a 1s, 24 a 5s), así que comparar entre horizontes es peras con manzanas.

Fix: evaluar TODOS los horizontes sobre el MISMO conjunto de tracks — los que
sobreviven al horizonte más largo (5s). Comparación pareada: cada auto es su
propio control. Mucho menos ruido aunque sean pocos.

Uso: conda activate sapiens_gpu; cd sapiens/pretrain; python eval_multi_horizon_sametrack.py
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
VAL_SCENES = ['7e2f727866c69ea0', '82f90331a1dfe968']
LONGEST = 50   # horizonte que define el conjunto de tracks (5s)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

HORIZONS = {
    10: ('clean10_baseline_p10', 'clean10_gated_init_p10'),
    20: ('clean10_baseline_p20', 'clean10_gated_init_p20'),
    30: ('clean10_baseline',     'clean10_gated_init'),
    50: ('clean10_baseline_p50', 'clean10_gated_init_p50'),
}


def make_ds(P):
    return TrajectoryDataset(data_root=DATA_ROOT, sequence_len=HISTORY_LEN+P,
        history_len=HISTORY_LEN, pred_len=P, voxel_res=VOXEL_RES,
        spatial_range=SPATIAL_RANGE, max_jump=5.0, scenes=VAL_SCENES)


def load_baseline(ckpt, P):
    m = BaselineTrajectoryModel(history_len=HISTORY_LEN, pred_len=P, hidden_dim=512)
    load_checkpoint(m, ckpt, map_location='cpu'); return m.eval().to(DEVICE)


def load_gated(ckpt, P):
    enc = dict(type='MAEViT4D', history_len=HISTORY_LEN, embed_dim=1024,
               num_tokens=NUM_VOXELS, arch='sapiens_0.3b', final_norm=True, mask_ratio=0.75)
    m = TrajectoryModelWithAttention(encoder=enc, history_len=HISTORY_LEN, pred_len=P,
            embed_dim=1024, num_heads=8, hidden_dim=512, scene_dim=64,
            freeze_encoder=True, use_gate=True, gate_init=0.5)
    load_checkpoint(m, ckpt, map_location='cpu'); return m.eval().to(DEVICE)


@torch.no_grad()
def eval_on(model, dataset, P, survivors, is_attn):
    ades = []
    for d in dataset:
        key = (d['scene_name'], d['object_id'])
        if key not in survivors:
            continue
        inp = d['inputs'].unsqueeze(0).to(DEVICE)
        h   = d['obj_history_flat'].unsqueeze(0).to(DEVICE)
        mean, std = d['norm_mean'], d['norm_std']
        pred = model(inp, h, mode='predict') if is_attn else model(h, mode='predict')
        pred = (pred.cpu().view(P, 3) * std + mean).numpy()
        tgt  = (d['obj_future_flat'].view(P, 3) * std + mean).numpy()
        ades.append(np.linalg.norm(pred[:, :2] - tgt[:, :2], axis=1).mean())
    return float(np.mean(ades)), len(ades)


def main():
    # Conjunto de tracks que sobreviven al horizonte más largo (5s)
    survivors = {(d['scene_name'], d['object_id']) for d in make_ds(LONGEST)}
    print(f'\nTracks que sobreviven a {LONGEST/10:.0f}s: {len(survivors)} (mismo conjunto en TODOS los horizontes)')

    import math
    hs, b_ade, g_ade, imp, gates = [], [], [], [], []
    print(f'\n{"="*72}')
    print(f'  CURVA PAREADA — mismos {len(survivors)} tracks a cada horizonte')
    print(f'{"="*72}')
    print(f'  {"Horizonte":>10}  {"Baseline":>10}  {"Gated":>10}  {"Mejora":>8}  {"gate":>7}  {"N":>4}')
    print(f'  {"-"*10}  {"-"*10}  {"-"*10}  {"-"*8}  {"-"*7}  {"-"*4}')
    for P in sorted(HORIZONS):
        bdir, gdir = HORIZONS[P]
        bck, gck = f'work_dirs/{bdir}/epoch_100.pth', f'work_dirs/{gdir}/epoch_100.pth'
        if not (os.path.exists(bck) and os.path.exists(gck)):
            continue
        ds = make_ds(P)
        ba, n = eval_on(load_baseline(bck, P), ds, P, survivors, False)
        gmodel = load_gated(gck, P)
        ga, _ = eval_on(gmodel, ds, P, survivors, True)
        gate = math.tanh(gmodel.scene_gate.item())
        pct = (ba - ga) / ba * 100
        hs.append(P/10); b_ade.append(ba); g_ade.append(ga); imp.append(pct); gates.append(gate)
        print(f'  {P/10:>8.0f}s  {ba:>8.3f}m  {ga:>8.3f}m  {pct:>6.1f}%  {gate:>+7.3f}  {n:>4}')
    print(f'{"="*72}')

    if len(hs) >= 2:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].plot(hs, b_ade, 'o-', label='Baseline (solo histórico)', color='#888')
        ax[0].plot(hs, g_ade, 's-', label='Gated (con escena LiDAR)', color='#2a7')
        ax[0].set_xlabel('Horizonte (s)'); ax[0].set_ylabel('Val ADE (m)')
        ax[0].set_title(f'Error vs horizonte (mismos {len(survivors)} tracks)')
        ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].plot(hs, imp, 'D-', color='#d62'); ax[1].axhline(0, color='k', lw=.5)
        ax[1].set_xlabel('Horizonte (s)'); ax[1].set_ylabel('Mejora de la escena (% ADE)')
        ax[1].set_title('Beneficio en ADE vs horizonte'); ax[1].grid(alpha=.3)
        ax[2].plot(hs, gates, '^-', color='#36c')
        ax[2].set_xlabel('Horizonte (s)'); ax[2].set_ylabel('tanh(scene_gate) aprendido')
        ax[2].set_title('Peso de la escena que el modelo APRENDE'); ax[2].grid(alpha=.3)
        plt.tight_layout(); plt.savefig('curva_multi_horizonte_pareada.png', dpi=120)
        print(f'  Gráfico: curva_multi_horizonte_pareada.png\n')


if __name__ == '__main__':
    main()
