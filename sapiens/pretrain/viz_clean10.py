"""
viz_clean10.py — Visualización BEV de las predicciones del mejor modelo
(gated_init, gate_init=0.5) sobre una escena de validación de waymo_clean.

Dibuja, en vista de pájaro (Bird's Eye View):
  - nube de puntos LiDAR (gris)
  - por cada objeto: historia (blanco), futuro REAL (verde), futuro PREDICHO (rojo)

Uso:
  conda activate sapiens_gpu
  cd sapiens/pretrain
  python viz_clean10.py [scene_id]     # default: 7e2f727866c69ea0 (val)
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

WAYMO_ROOT = '/home/lcad/lidar_sweep_viewer/waymo_clean'
CKPT = 'work_dirs/clean10_gated_init/epoch_100.pth'
HISTORY_LEN, PRED_LEN, VOXEL_RES = 5, 30, 2.0
SPATIAL_RANGE, NUM_VOXELS = [-10, 10, -10, 10, -2, 4], 300
VIEW = 40   # metros visibles
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SCENE = sys.argv[1] if len(sys.argv) > 1 else '7e2f727866c69ea0'


def load_model():
    enc = dict(type='MAEViT4D', history_len=HISTORY_LEN, embed_dim=1024,
               num_tokens=NUM_VOXELS, arch='sapiens_0.3b', final_norm=True, mask_ratio=0.75)
    m = TrajectoryModelWithAttention(encoder=enc, history_len=HISTORY_LEN, pred_len=PRED_LEN,
            embed_dim=1024, num_heads=8, hidden_dim=512, scene_dim=64,
            freeze_encoder=True, use_gate=True, gate_init=0.5)
    load_checkpoint(m, CKPT, map_location='cpu'); return m.eval().to(DEVICE)


def denorm(flat, mean, std, ref, n):
    """normalizado relativo -> sensor (x,y,z)."""
    arr = flat.reshape(n, 3) * std + mean + ref
    return arr


@torch.no_grad()
def main():
    model = load_model()
    ds = TrajectoryDataset(data_root=WAYMO_ROOT, sequence_len=HISTORY_LEN+PRED_LEN,
            history_len=HISTORY_LEN, pred_len=PRED_LEN, voxel_res=VOXEL_RES,
            spatial_range=SPATIAL_RANGE, max_jump=5.0, scenes=[SCENE])
    print(f'Cena {SCENE}: {len(ds)} objetos')

    # LiDAR de frame 0 (sensor coords)
    bin0 = os.path.join(WAYMO_ROOT, 'bin_files', SCENE, '0.bin')
    pts = np.fromfile(bin0, dtype=np.float32).reshape(-1, 4)[:, :3]
    m = (np.abs(pts[:, 0]) < VIEW) & (np.abs(pts[:, 1]) < VIEW)
    pts = pts[m]

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.scatter(pts[:, 1], pts[:, 0], s=0.5, c='#555', alpha=.4)   # LiDAR

    n_obj = 0
    for d in ds:
        mean = d['norm_mean'].numpy(); std = d['norm_std'].numpy()
        ref  = d['ref_center'].numpy()
        hist = denorm(d['obj_history_flat'].numpy(), mean, std, ref, HISTORY_LEN)
        gt   = denorm(d['obj_future_flat'].numpy(),  mean, std, ref, PRED_LEN)
        pred_flat = model(d['inputs'].unsqueeze(0).to(DEVICE),
                          d['obj_history_flat'].unsqueeze(0).to(DEVICE), mode='predict')
        pred = denorm(pred_flat.cpu().numpy().reshape(-1), mean, std, ref, PRED_LEN)

        # saltar objetos fuera de vista
        if np.abs(hist[-1, 0]) > VIEW or np.abs(hist[-1, 1]) > VIEW:
            continue
        # (plot Y horizontal, X vertical = adelante hacia arriba)
        ax.plot(hist[:, 1], hist[:, 0], '-', color='white', lw=2, zorder=3)
        ax.plot(np.r_[hist[-1, 1], gt[:, 1]],   np.r_[hist[-1, 0], gt[:, 0]],
                '-', color='#1f4', lw=2, zorder=4)
        ax.plot(np.r_[hist[-1, 1], pred[:, 1]], np.r_[hist[-1, 0], pred[:, 0]],
                '--', color='#f33', lw=2, zorder=5)
        ax.plot(hist[-1, 1], hist[-1, 0], 'o', color='yellow', ms=5, zorder=6)
        n_obj += 1

    ax.plot(0, 0, '^', color='cyan', ms=14, zorder=7)   # ego
    ax.set_xlim(-VIEW, VIEW); ax.set_ylim(-VIEW, VIEW)
    ax.set_aspect('equal'); ax.set_facecolor('#111')
    ax.set_xlabel('Y — esquerda (m)'); ax.set_ylabel('X — frente (m)')
    ax.set_title(f'MOTF — predições na cena {SCENE} (val)  ·  {n_obj} objetos  ·  horizonte 3s')
    # leyenda
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], color='white', lw=2, label='Histórico'),
        Line2D([0],[0], color='#1f4', lw=2, label='Futuro real'),
        Line2D([0],[0], color='#f33', lw=2, ls='--', label='Futuro predito (MOTF)'),
        Line2D([0],[0], marker='^', color='cyan', lw=0, label='Ego (veículo)'),
    ], loc='upper right', facecolor='#222', labelcolor='white')
    plt.tight_layout()
    out = f'viz_pred_{SCENE}.png'
    plt.savefig(out, dpi=120, facecolor='#111')
    print(f'Imagem salva: {out}  ({n_obj} objetos desenhados)')


if __name__ == '__main__':
    main()
