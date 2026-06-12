"""
evaluate_ade_fde.py — Evaluación ADE/FDE comparativa baseline vs MOTF gated

Uso:
    conda activate sapiens_gpu
    cd sapiens/pretrain
    python evaluate_ade_fde.py

Reporta ADE/FDE (en metros, en coord. sensor) para:
  - Dataset completo (10 escenas)
  - Split train (8 escenas) vs val (2 escenas)
  - Por escena
  - Valor del gate del modelo de atención
"""
import os
import sys
import torch
import numpy as np
from mmengine.runner import load_checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmpretrain.datasets.trajectory_dataset import TrajectoryDataset
from mmpretrain.models.trajectory_pred.baseline_model import BaselineTrajectoryModel
from mmpretrain.models.trajectory_pred.trajectory_model_attn import TrajectoryModelWithAttention
from mmpretrain.models.backbones.mae_vit_4d import MAEViT4D  # noqa: registra módulo

# ── Configuración ────────────────────────────────────────────────────────────

DATA_ROOT = '/home/lcad/lidar_sweep_viewer/waymo_10'

CKPT_BASELINE = 'work_dirs/baseline_multiescena/epoch_300.pth'
CKPT_ATTN     = 'work_dirs/trajectory_attn_multiescena/epoch_300.pth'
CKPT_AUG      = 'work_dirs/trajectory_attn_augmented/epoch_500.pth'
CKPT_NOGATE   = 'work_dirs/trajectory_attn_nogate/epoch_500.pth'
MAE_CKPT      = 'work_dirs/mae_encoder_pretrained.pth'

HISTORY_LEN   = 5
PRED_LEN      = 5
VOXEL_RES     = 2.0
SPATIAL_RANGE = [-10, 10, -10, 10, -2, 4]
NUM_VOXELS    = 300

TRAIN_SCENES = {
    '1a01176ef4830499', '1a1918cd002b6a94', '1a64c26c56412fc5',
    '1b20914ea0bc28bd', '1b88ea97e03be2bf', '1ba702a85f4fa15c',
    '1bb5aa0c95f2ce0e', '58d5f1b9e6a1a2f7',
}
VAL_SCENES = {'1a85be8ad06a056c', '1d9b8e390bed186f'}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Cargar modelos ───────────────────────────────────────────────────────────

def load_baseline():
    model = BaselineTrajectoryModel(
        history_len=HISTORY_LEN, pred_len=PRED_LEN, hidden_dim=512
    )
    load_checkpoint(model, CKPT_BASELINE, map_location='cpu')
    model.eval().to(DEVICE)
    return model


def load_attn(ckpt=CKPT_ATTN, use_gate=True):
    encoder_cfg = dict(
        type='MAEViT4D',
        history_len=HISTORY_LEN,
        embed_dim=1024,
        num_tokens=NUM_VOXELS,
        arch='sapiens_0.3b',
        final_norm=True,
        mask_ratio=0.75,
    )
    model = TrajectoryModelWithAttention(
        encoder=encoder_cfg,
        history_len=HISTORY_LEN,
        pred_len=PRED_LEN,
        embed_dim=1024,
        num_heads=8,
        hidden_dim=512,
        scene_dim=64,
        freeze_encoder=True,
        use_gate=use_gate,
    )
    load_checkpoint(model, ckpt, map_location='cpu')
    model.eval().to(DEVICE)
    return model

# ── Evaluación ───────────────────────────────────────────────────────────────

def eval_sample(model, data, is_attn):
    inputs           = data['inputs'].unsqueeze(0).to(DEVICE)
    obj_history_flat = data['obj_history_flat'].unsqueeze(0).to(DEVICE)
    obj_future_flat  = data['obj_future_flat'].unsqueeze(0)
    norm_mean        = data['norm_mean']
    norm_std         = data['norm_std']

    with torch.no_grad():
        if is_attn:
            pred_flat = model(inputs, obj_history_flat, mode='predict')
        else:
            pred_flat = model(obj_history_flat, mode='predict')

    pred_flat = pred_flat.cpu()
    pred   = pred_flat.view(PRED_LEN, 3) * norm_std + norm_mean   # desnorm
    target = obj_future_flat.view(PRED_LEN, 3) * norm_std + norm_mean

    pred   = pred.numpy()
    target = target.numpy()

    dist = np.linalg.norm(pred[:, :2] - target[:, :2], axis=1)  # solo XY
    ade  = float(dist.mean())
    fde  = float(dist[-1])
    return ade, fde


def evaluate(model, dataset, is_attn, label):
    results = {}   # scene -> list of (ade, fde)
    for i, data in enumerate(dataset):
        scene = data['scene_name']
        try:
            ade, fde = eval_sample(model, data, is_attn)
            results.setdefault(scene, []).append((ade, fde))
        except Exception as e:
            print(f'  [WARN] idx={i} scene={scene}: {e}')

    # Agregar por split
    all_ade, all_fde = [], []
    train_ade, train_fde = [], []
    val_ade, val_fde = [], []

    for scene, vals in sorted(results.items()):
        ades = [v[0] for v in vals]
        fdes = [v[1] for v in vals]
        all_ade += ades;  all_fde += fdes
        if scene in TRAIN_SCENES:
            train_ade += ades; train_fde += fdes
        elif scene in VAL_SCENES:
            val_ade += ades;   val_fde += fdes

    print(f'\n{"─"*58}')
    print(f'  {label}')
    print(f'{"─"*58}')
    print(f'  {"Split":<12} {"N":>5}  {"ADE (m)":>10}  {"FDE (m)":>10}')
    print(f'  {"─"*12}  {"─"*5}  {"─"*10}  {"─"*10}')

    def row(name, ades, fdes):
        if not ades:
            return
        print(f'  {name:<12} {len(ades):>5}  '
              f'{np.mean(ades):>8.3f} m  {np.mean(fdes):>8.3f} m')

    row('Train (8)',  train_ade, train_fde)
    row('Val   (2)',  val_ade,   val_fde)
    row('Total (10)', all_ade,   all_fde)

    print(f'\n  Por escena:')
    for scene, vals in sorted(results.items()):
        ades = [v[0] for v in vals]
        fdes = [v[1] for v in vals]
        split = 'train' if scene in TRAIN_SCENES else 'val  '
        print(f'    [{split}] {scene[:16]}  n={len(ades):>3}'
              f'  ADE={np.mean(ades):.3f}m  FDE={np.mean(fdes):.3f}m')

    return {
        'total_ade': np.mean(all_ade)  if all_ade  else None,
        'total_fde': np.mean(all_fde)  if all_fde  else None,
        'val_ade':   np.mean(val_ade)  if val_ade  else None,
        'val_fde':   np.mean(val_fde)  if val_fde  else None,
        'train_ade': np.mean(train_ade) if train_ade else None,
        'train_fde': np.mean(train_fde) if train_fde else None,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f'Dispositivo: {DEVICE}')

    print('\nCargando dataset...')
    dataset = TrajectoryDataset(
        data_root=DATA_ROOT,
        sequence_len=HISTORY_LEN + PRED_LEN,
        history_len=HISTORY_LEN,
        pred_len=PRED_LEN,
        voxel_res=VOXEL_RES,
        spatial_range=SPATIAL_RANGE,
        max_jump=5.0,
    )
    print(f'  {len(dataset)} muestras en total')

    print('\nCargando modelo Baseline...')
    baseline = load_baseline()

    print('Cargando modelo MOTF (gated attention, sin augmentación)...')
    attn = load_attn(CKPT_ATTN)
    gate_val = float(torch.tanh(attn.scene_gate).item())
    print(f'  Gate actual: tanh(scene_gate) = {gate_val:.4f}')

    print('Cargando modelo MOTF (gated attention + AUGMENTACIÓN)...')
    attn_aug = load_attn(CKPT_AUG)
    gate_aug = float(torch.tanh(attn_aug.scene_gate).item())
    print(f'  Gate actual: tanh(scene_gate) = {gate_aug:.4f}')

    print('Cargando modelo MOTF (SIN gate, escena siempre activa + aug)...')
    attn_ng = load_attn(CKPT_NOGATE, use_gate=False)
    print('  Gate: desactivado (escena siempre activa)')

    res_base = evaluate(baseline, dataset, is_attn=False, label='BASELINE')
    res_aug  = evaluate(attn_aug, dataset, is_attn=True,  label='MOTF — Gated + Augmentación')
    res_ng   = evaluate(attn_ng,  dataset, is_attn=True,  label='MOTF — SIN gate + Augmentación')

    # Tabla comparativa final
    print(f'\n{"═"*72}')
    print(f'  COMPARATIVA FINAL  (ADE / FDE en metros, solo XY)')
    print(f'{"═"*72}')
    print(f'  {"":15} {"Baseline":>12}  {"Gated+Aug":>12}  {"SinGate+Aug":>12}')
    print(f'  {"─"*15}  {"─"*12}  {"─"*12}  {"─"*12}')

    for split in ['train', 'val', 'total']:
        for metric in ['ade', 'fde']:
            key = f'{split}_{metric}'
            b = res_base.get(key)
            g = res_aug.get(key)
            n = res_ng.get(key)
            if b is None or g is None or n is None:
                continue
            label = f'{split.capitalize()} {metric.upper()}'
            best = min(b, g, n)
            def fmt(v):
                star = '*' if abs(v - best) < 1e-6 else ' '
                return f'{v:>9.3f}m{star}'
            print(f'  {label:<15} {fmt(b)}  {fmt(g)}  {fmt(n)}')

    print(f'{"═"*72}')
    print(f'  Gate (gated model): {gate_aug:+.4f}  |  SinGate: escena forzada activa')
    print(f'  * = mejor de los tres en esa métrica')
    print(f'  → Si "SinGate" supera al baseline en VAL, la escena APORTA.')
    print(f'    Si no, el problema es el horizonte (0.5s) y se necesita datos limpios.')
    print(f'{"═"*72}\n')


if __name__ == '__main__':
    main()
