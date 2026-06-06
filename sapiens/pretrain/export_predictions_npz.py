"""
export_predictions_npz.py  —  PASO 1 de la visualización 3D
Corre en sapiens_final. Usa el TrajectoryDataset (voxelización/normalización
idénticas al treino) + el modelo para predizir trayectorias, y exporta tudo a
um .npz que o passo 2 (Open3D, env lidar_mae) consome.

Uso:
    conda activate sapiens_final
    python export_predictions_npz.py            # modelo attn-gated (default)
    python export_predictions_npz.py baseline   # baseline MLP
"""
import sys
import numpy as np
import torch

WAYMO_ROOT    = '/home/lcad/lidar_sweep_viewer/waymo_10'
SCENE_ID      = '58d5f1b9e6a1a2f7'
HISTORY_LEN   = 5
PRED_LEN      = 5
VOXEL_RES     = 2.0
SPATIAL_RANGE = [-10, 10, -10, 10, -2, 4]
DISPLAY_FRAME = HISTORY_LEN - 1   # frame "atual" = último do histórico (4)

MODEL_TYPE = sys.argv[1] if len(sys.argv) > 1 else 'attn'
CHECKPOINTS = {
    'attn':     'work_dirs/trajectory_attn_gated/epoch_500.pth',
    'baseline': 'work_dirs/baseline_overfit_500/epoch_500.pth',
}
CHECKPOINT = CHECKPOINTS[MODEL_TYPE]
OUT_NPZ    = f'viz_predictions_{MODEL_TYPE}.npz'


def build_model():
    from mmengine.runner import load_checkpoint
    if MODEL_TYPE == 'attn':
        from mmengine.config import Config
        from mmpretrain.models.trajectory_pred.trajectory_model_attn import \
            TrajectoryModelWithAttention
        from mmpretrain.models.backbones import mae_vit_4d  # registra MAEViT4D
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

    obj_ids, hist_all, gt_all, pred_all = [], [], [], []
    for i in range(len(ds)):
        d = ds[i]
        std  = d['norm_std'].numpy()
        mean = d['norm_mean'].numpy()
        ref  = d['ref_center'].numpy()
        with torch.no_grad():
            if MODEL_TYPE == 'attn':
                pf = model(d['inputs'].unsqueeze(0),
                           d['obj_history_flat'].unsqueeze(0), mode='predict')
            else:
                pf = model(d['obj_history_flat'].unsqueeze(0), mode='predict')
        pred_norm   = pf.squeeze(0).view(PRED_LEN, 3).numpy()
        hist_norm   = d['obj_history_flat'].view(HISTORY_LEN, 3).numpy()
        future_norm = d['obj_future_flat'].view(PRED_LEN, 3).numpy()

        # desnormalizar -> sensor frame (mesmo frame das bins/bboxes)
        obj_ids.append(str(d['object_id']))
        hist_all.append(hist_norm   * std + mean + ref)
        gt_all.append(future_norm * std + mean + ref)
        pred_all.append(pred_norm   * std + mean + ref)

    np.savez(
        OUT_NPZ,
        obj_ids=np.array(obj_ids),
        history=np.array(hist_all, dtype=np.float32),
        future_gt=np.array(gt_all, dtype=np.float32),
        future_pred=np.array(pred_all, dtype=np.float32),
        scene_id=SCENE_ID,
        display_frame=DISPLAY_FRAME,
        model_type=MODEL_TYPE,
    )
    print(f'✅ {len(obj_ids)} objetos exportados -> {OUT_NPZ}')


if __name__ == '__main__':
    main()
