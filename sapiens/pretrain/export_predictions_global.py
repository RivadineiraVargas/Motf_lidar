"""
export_predictions_global.py  —  exporta trajetórias em coordenadas GLOBAIS
para o viewer C++ (show_point_cloud) desenhar as predições.

Formato de saída (predictions_global.txt), uma linha por ponto:
    <scene> <obj_id> <kind> <t> <x> <y> <z>
  kind: 0=histórico, 1=futuro real, 2=futuro predito
  coords globais (mundo). O C++ transforma p/ sensor frame com inv(pose).

Uso:
    conda activate sapiens_final
    python export_predictions_global.py attn       # default
    python export_predictions_global.py baseline
"""
import os
import sys
import numpy as np
import torch

WAYMO_ROOT    = '/home/lcad/lidar_sweep_viewer/waymo_10'
SCENE_ID      = '58d5f1b9e6a1a2f7'
HISTORY_LEN   = 5
PRED_LEN      = 5
VOXEL_RES     = 2.0
SPATIAL_RANGE = [-10, 10, -10, 10, -2, 4]

MODEL_TYPE = sys.argv[1] if len(sys.argv) > 1 else 'attn'
CHECKPOINTS = {
    'attn':     'work_dirs/trajectory_attn_gated/epoch_500.pth',
    'baseline': 'work_dirs/baseline_overfit_500/epoch_500.pth',
}
CHECKPOINT = CHECKPOINTS[MODEL_TYPE]
OUT_TXT    = f'/home/lcad/lidar_sweep_viewer/predictions_global.txt'


def load_pose(frame):
    path = os.path.join(WAYMO_ROOT, 'poses', SCENE_ID, f'{frame}.txt')
    with open(path) as f:
        rows = [list(map(float, l.split())) for l in f if len(l.split()) == 4]
    return np.array(rows) if len(rows) == 4 else np.eye(4)


def global_center(frame, obj_id):
    """Centro global (média dos 8 vértices) do objeto no frame, ou None."""
    path = os.path.join(WAYMO_ROOT, 'objs_bbox', SCENE_ID, str(frame), f'{obj_id}.txt')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        verts = [list(map(float, l.split())) for l in f if len(l.split()) == 3]
    if len(verts) != 8:
        return None
    return np.mean(verts, axis=0)


def build_model():
    from mmengine.runner import load_checkpoint
    if MODEL_TYPE == 'attn':
        from mmengine.config import Config
        from mmpretrain.models.trajectory_pred.trajectory_model_attn import \
            TrajectoryModelWithAttention
        from mmpretrain.models.backbones import mae_vit_4d
        cfg = Config.fromfile('configs/sapiens_mae/lidar/trajectory_attn_overfit.py')
        margs = {k: v for k, v in cfg.model.items() if k != 'type'}
        model = TrajectoryModelWithAttention(**margs)
    else:
        from mmpretrain.models.trajectory_pred.baseline_model import \
            BaselineTrajectoryModel
        model = BaselineTrajectoryModel(history_len=HISTORY_LEN, pred_len=PRED_LEN)
    load_checkpoint(model, CHECKPOINT, map_location='cpu')
    model.eval()
    return model


def main():
    from mmpretrain.datasets import TrajectoryDataset
    model = build_model()
    ds = TrajectoryDataset(
        data_root=WAYMO_ROOT, sequence_len=HISTORY_LEN + PRED_LEN,
        history_len=HISTORY_LEN, pred_len=PRED_LEN,
        voxel_res=VOXEL_RES, spatial_range=SPATIAL_RANGE,
    )
    # pose de cada frame (o dataset normaliza cada centro com inv(pose_t)).
    poses = {t: load_pose(t) for t in range(HISTORY_LEN + PRED_LEN)}

    lines = []
    n_pred = 0
    for i in range(len(ds)):
        d = ds[i]
        oid  = str(d['object_id'])
        std  = d['norm_std'].numpy()
        mean = d['norm_mean'].numpy()
        ref  = d['ref_center'].numpy()    # sensor frame 0

        with torch.no_grad():
            if MODEL_TYPE == 'attn':
                pf = model(d['inputs'].unsqueeze(0),
                           d['obj_history_flat'].unsqueeze(0), mode='predict')
            else:
                pf = model(d['obj_history_flat'].unsqueeze(0), mode='predict')
        pred_norm = pf.squeeze(0).view(PRED_LEN, 3).numpy()
        pred_rel  = pred_norm * std + mean   # relative[t] = centers[t] - ref

        # histórico e GT: centros globais REAIS dos arquivos de bbox
        for t in range(HISTORY_LEN):
            c = global_center(t, oid)
            if c is not None:
                lines.append(f'{SCENE_ID} {oid} 0 {t} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}')
        for k, t in enumerate(range(HISTORY_LEN, HISTORY_LEN + PRED_LEN)):
            c = global_center(t, oid)
            if c is not None:
                lines.append(f'{SCENE_ID} {oid} 1 {t} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}')
            # centers[t] = relative[t] + ref (sensor frame t) -> global via pose_t
            center_t = pred_rel[k] + ref
            cp = (poses[t] @ np.append(center_t, 1.0))[:3]
            lines.append(f'{SCENE_ID} {oid} 2 {t} {cp[0]:.4f} {cp[1]:.4f} {cp[2]:.4f}')
        n_pred += 1

    with open(OUT_TXT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'✅ {n_pred} objetos, {len(lines)} pontos -> {OUT_TXT}')


if __name__ == '__main__':
    main()
