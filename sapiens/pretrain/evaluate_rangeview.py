"""
evaluate_rangeview.py — ADE/FDE del track RANGE-VIEW vs baseline (y referencia al
mejor de vóxels). 10 escenas, horizonte 3s, val = 2 escenas.

Uso: conda activate sapiens_gpu; cd sapiens/pretrain; python evaluate_rangeview.py
"""
import os, sys
import numpy as np
import torch
from mmengine.runner import load_checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mmpretrain.datasets.range_view import RangeViewTrajectoryDataset, num_tokens, patch_dim
from mmpretrain.models.trajectory_pred.baseline_model import BaselineTrajectoryModel
from mmpretrain.models.trajectory_pred.trajectory_model_attn import TrajectoryModelWithAttention
from mmpretrain.models.backbones.mae_vit_4d import MAEViT4D  # noqa

DATA_ROOT = '/home/lcad/lidar_sweep_viewer/waymo_clean'
HISTORY_LEN, PRED_LEN, SCENE_FRAMES = 5, 30, 5
CKPT_BASELINE = 'work_dirs/clean10_baseline/epoch_100.pth'
CKPT_RV       = 'work_dirs/clean10_rv_gated_init/epoch_100.pth'
VAL_SCENES = {'7e2f727866c69ea0', '82f90331a1dfe968'}
ALL = ['2a81f5233075e987','2e41fe6faf5cd2ea','367b072edc9822ea','394e61f27c2a1700',
       '4014ae5bcda2726f','41692b0ec7ff4123','4a2ef30000d19d90','4b60f9400a30ceaf',
       '7e2f727866c69ea0','82f90331a1dfe968']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_baseline():
    m = BaselineTrajectoryModel(history_len=HISTORY_LEN, pred_len=PRED_LEN, hidden_dim=512)
    load_checkpoint(m, CKPT_BASELINE, map_location='cpu'); return m.eval().to(DEVICE)


def load_rv():
    enc = dict(type='MAEViT4D', history_len=patch_dim(SCENE_FRAMES), embed_dim=1024,
               num_tokens=num_tokens(), arch='sapiens_0.3b', final_norm=True, mask_ratio=0.75)
    m = TrajectoryModelWithAttention(encoder=enc, history_len=HISTORY_LEN, pred_len=PRED_LEN,
            embed_dim=1024, num_heads=8, hidden_dim=512, scene_dim=64,
            freeze_encoder=True, use_gate=True, gate_init=0.5)
    load_checkpoint(m, CKPT_RV, map_location='cpu'); return m.eval().to(DEVICE)


@torch.no_grad()
def evaluate(model, ds, is_attn):
    res = {}
    for d in ds:
        sc = d['scene_name']
        h = d['obj_history_flat'].unsqueeze(0).to(DEVICE)
        fut = d['obj_future_flat']; mean = d['norm_mean']; std = d['norm_std']
        if is_attn:
            pf = model(d['inputs'].unsqueeze(0).to(DEVICE), h, mode='predict')
        else:
            pf = model(h, mode='predict')
        pred = (pf.cpu().view(PRED_LEN, 3) * std + mean).numpy()
        tgt = (fut.view(PRED_LEN, 3) * std + mean).numpy()
        dist = np.linalg.norm(pred[:, :2] - tgt[:, :2], axis=1)
        res.setdefault(sc, []).append((dist.mean(), dist[-1]))
    val = [v for s, vs in res.items() if s in VAL_SCENES for v in vs]
    allv = [v for vs in res.values() for v in vs]
    return (np.mean([a for a, _ in val]), np.mean([f for _, f in val]),
            np.mean([a for a, _ in allv]), np.mean([f for _, f in allv]))


def main():
    ds = RangeViewTrajectoryDataset(data_root=DATA_ROOT, sequence_len=HISTORY_LEN+PRED_LEN,
            history_len=HISTORY_LEN, pred_len=PRED_LEN, voxel_res=2.0,
            spatial_range=[-10,10,-10,10,-2,4], max_jump=5.0, scenes=ALL)
    print(f'{len(ds)} muestras')
    bva, bvf, bta, btf = evaluate(load_baseline(), ds, False)
    rv = load_rv(); rva, rvf, rta, rtf = evaluate(rv, ds, True)
    gate = float(torch.tanh(rv.scene_gate).item())

    print(f'\n{"="*58}')
    print(f'  RANGE-VIEW vs BASELINE — horizonte 3s (ADE/FDE, m)')
    print(f'{"="*58}')
    print(f'  {"":12} {"Baseline":>12}  {"Range-view":>12}')
    print(f'  Val ADE     {bva:>10.3f}m  {rva:>10.3f}m')
    print(f'  Val FDE     {bvf:>10.3f}m  {rvf:>10.3f}m')
    print(f'  Total ADE   {bta:>10.3f}m  {rta:>10.3f}m')
    print(f'  Total FDE   {btf:>10.3f}m  {rtf:>10.3f}m')
    print(f'{"="*58}')
    print(f'  gate aprendido: {gate:+.3f}   (arrancó en 0.5)')
    print(f'  Mejora Val ADE vs baseline: {(bva-rva)/bva*100:+.1f}%')
    print(f'  Ref. vóxel (mejor): Val ADE 1.303m (-35%)')
    print(f'{"="*58}\n')


if __name__ == '__main__':
    main()
