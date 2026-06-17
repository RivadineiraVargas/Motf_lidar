"""
export_predictions_clean10.py — exporta predicciones del MEJOR modelo (gated_init,
gate_init=0.5, encoder limpio) en coords GLOBALES para el viewer C++ show_point_cloud.

Formato (predictions_global.txt), una línea por punto:
    <scene> <obj_id> <kind> <t> <x> <y> <z>
  kind: 0=histórico, 1=futuro real, 2=futuro predito (coords globales)

Exporta las 2 escenas de validación de waymo_clean. Luego, en el viewer:
    ./show_point_cloud --input waymo_clean/ ...   y presionar 't' para predicciones.

Uso: conda activate sapiens_gpu; cd sapiens/pretrain; python export_predictions_clean10.py
"""
import os, sys
import numpy as np
import torch
from mmengine.runner import load_checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mmpretrain.datasets.trajectory_dataset import TrajectoryDataset
from mmpretrain.models.trajectory_pred.trajectory_model_attn import TrajectoryModelWithAttention
from mmpretrain.models.backbones.mae_vit_4d import MAEViT4D  # noqa

WAYMO_ROOT = '/home/lcad/lidar_sweep_viewer/waymo_clean'
CKPT = 'work_dirs/clean10_gated_init/epoch_100.pth'
HISTORY_LEN, PRED_LEN, VOXEL_RES = 5, 30, 2.0
SPATIAL_RANGE, NUM_VOXELS = [-10, 10, -10, 10, -2, 4], 300
VAL_SCENES = ['7e2f727866c69ea0', '82f90331a1dfe968']
OUT_TXT = '/home/lcad/lidar_sweep_viewer/predictions_global.txt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_pose(scene, frame):
    path = os.path.join(WAYMO_ROOT, 'poses', scene, f'{frame}.txt')
    with open(path) as f:
        rows = [list(map(float, l.split())) for l in f if len(l.split()) == 4]
    return np.array(rows) if len(rows) == 4 else np.eye(4)


def global_center(scene, frame, obj_id):
    path = os.path.join(WAYMO_ROOT, 'objs_bbox', scene, str(frame), f'{obj_id}.txt')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        verts = [list(map(float, l.split())) for l in f if len(l.split()) == 3]
    return np.mean(verts, axis=0) if len(verts) == 8 else None


def build_model():
    enc = dict(type='MAEViT4D', history_len=HISTORY_LEN, embed_dim=1024,
               num_tokens=NUM_VOXELS, arch='sapiens_0.3b', final_norm=True, mask_ratio=0.75)
    m = TrajectoryModelWithAttention(encoder=enc, history_len=HISTORY_LEN, pred_len=PRED_LEN,
            embed_dim=1024, num_heads=8, hidden_dim=512, scene_dim=64,
            freeze_encoder=True, use_gate=True, gate_init=0.5)
    load_checkpoint(m, CKPT, map_location='cpu'); return m.eval().to(DEVICE)


@torch.no_grad()
def main():
    model = build_model()
    lines, n_pred = [], 0
    for scene in VAL_SCENES:
        ds = TrajectoryDataset(data_root=WAYMO_ROOT, sequence_len=HISTORY_LEN+PRED_LEN,
                history_len=HISTORY_LEN, pred_len=PRED_LEN, voxel_res=VOXEL_RES,
                spatial_range=SPATIAL_RANGE, max_jump=5.0, scenes=[scene])
        poses = {t: load_pose(scene, t) for t in range(HISTORY_LEN + PRED_LEN)}
        for d in ds:
            oid = str(d['object_id'])
            std, mean, ref = d['norm_std'].numpy(), d['norm_mean'].numpy(), d['ref_center'].numpy()
            pf = model(d['inputs'].unsqueeze(0).to(DEVICE),
                       d['obj_history_flat'].unsqueeze(0).to(DEVICE), mode='predict')
            pred_rel = pf.cpu().view(PRED_LEN, 3).numpy() * std + mean

            for t in range(HISTORY_LEN):
                c = global_center(scene, t, oid)
                if c is not None:
                    lines.append(f'{scene} {oid} 0 {t} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}')
            for k, t in enumerate(range(HISTORY_LEN, HISTORY_LEN + PRED_LEN)):
                c = global_center(scene, t, oid)
                if c is not None:
                    lines.append(f'{scene} {oid} 1 {t} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}')
                center_t = pred_rel[k] + ref
                cp = (poses[t] @ np.append(center_t, 1.0))[:3]
                lines.append(f'{scene} {oid} 2 {t} {cp[0]:.4f} {cp[1]:.4f} {cp[2]:.4f}')
            n_pred += 1

    with open(OUT_TXT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'{n_pred} objetos, {len(lines)} pontos -> {OUT_TXT}')


if __name__ == '__main__':
    main()
