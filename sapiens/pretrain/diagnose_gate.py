"""
diagnose_gate.py — Investiga por qué el scene_gate no abre.

3 pruebas:
  A. Gradiente: ¿llega gradiente a scene_gate? ¿de qué signo/magnitud?
  B. Gate forzado: forzar gate=1 y medir loss train/val vs gate=0.
     Si gate=1 NO baja la loss, la escena no aporta (gate cerrado es correcto).
  C. Linealidad: ¿cuánto error queda con un predictor de velocidad constante
     (extrapolación lineal del histórico)? Si el movimiento es ~lineal,
     el histórico basta y la escena es redundante.

Uso:
    conda activate sapiens_gpu
    cd sapiens/pretrain
    python diagnose_gate.py
"""
import os
import sys
import torch
import numpy as np
from mmengine.runner import load_checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmpretrain.datasets.trajectory_dataset import TrajectoryDataset
from mmpretrain.models.trajectory_pred.trajectory_model_attn import TrajectoryModelWithAttention
from mmpretrain.models.backbones.mae_vit_4d import MAEViT4D  # noqa

DATA_ROOT = '/home/lcad/lidar_sweep_viewer/waymo_10'
CKPT_AUG  = 'work_dirs/trajectory_attn_augmented/epoch_500.pth'

HISTORY_LEN, PRED_LEN = 5, 5
VOXEL_RES = 2.0
SPATIAL_RANGE = [-10, 10, -10, 10, -2, 4]
NUM_VOXELS = 300

TRAIN_SCENES = {
    '1a01176ef4830499', '1a1918cd002b6a94', '1a64c26c56412fc5',
    '1b20914ea0bc28bd', '1b88ea97e03be2bf', '1ba702a85f4fa15c',
    '1bb5aa0c95f2ce0e', '58d5f1b9e6a1a2f7',
}
VAL_SCENES = {'1a85be8ad06a056c', '1d9b8e390bed186f'}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def build_model(ckpt):
    encoder_cfg = dict(
        type='MAEViT4D', history_len=HISTORY_LEN, embed_dim=1024,
        num_tokens=NUM_VOXELS, arch='sapiens_0.3b', final_norm=True, mask_ratio=0.75,
    )
    model = TrajectoryModelWithAttention(
        encoder=encoder_cfg, history_len=HISTORY_LEN, pred_len=PRED_LEN,
        embed_dim=1024, num_heads=8, hidden_dim=512, scene_dim=64, freeze_encoder=True,
    )
    load_checkpoint(model, ckpt, map_location='cpu')
    return model.to(DEVICE)


def get_dataset():
    return TrajectoryDataset(
        data_root=DATA_ROOT, sequence_len=HISTORY_LEN + PRED_LEN,
        history_len=HISTORY_LEN, pred_len=PRED_LEN, voxel_res=VOXEL_RES,
        spatial_range=SPATIAL_RANGE, max_jump=5.0, augment=False,
    )


def collate(dataset, scene_set):
    """Junta todas las muestras de un split en tensores batch."""
    I, H, F = [], [], []
    for d in dataset:
        if d['scene_name'] not in scene_set:
            continue
        I.append(d['inputs'])
        H.append(d['obj_history_flat'])
        F.append(d['obj_future_flat'])
    return (torch.stack(I).to(DEVICE),
            torch.stack(H).to(DEVICE),
            torch.stack(F).to(DEVICE))


# ── Prueba A: gradiente que llega a scene_gate ───────────────────────────────
def test_gradient(model, batch):
    inputs, hist, fut = batch
    model.train()
    model.zero_grad()
    # forward manual en modo loss
    out = model(inputs, hist, mode='loss', obj_future_flat=fut)
    out['loss'].backward()
    g = model.scene_gate.grad
    print('\n── Prueba A: ¿llega gradiente a scene_gate? ──')
    print(f'  scene_gate (valor):     {model.scene_gate.item():+.6f}')
    print(f'  scene_gate.grad:        {g.item():+.6e}'
          if g is not None else '  scene_gate.grad: None (¡no fluye gradiente!)')
    print(f'  loss actual:            {out["loss"].item():.6f}')
    model.eval()


# ── Prueba B: forzar el gate y medir loss ────────────────────────────────────
@torch.no_grad()
def loss_at_gate(model, batch, gate_value):
    """Evalúa la loss MSE forzando tanh(scene_gate) = gate_value."""
    inputs, hist, fut = batch
    # backup
    orig = model.scene_gate.data.clone()
    # tanh(x)=v  =>  x = atanh(v); para v=0 -> 0, v=1 -> grande
    v = max(min(gate_value, 0.999), -0.999)
    model.scene_gate.data.fill_(float(np.arctanh(v)))
    pred = model(inputs, hist, mode='predict')
    loss = torch.nn.functional.mse_loss(pred, fut).item()
    model.scene_gate.data.copy_(orig)
    return loss


def test_forced_gate(model, train_batch, val_batch):
    print('\n── Prueba B: forzar gate y medir loss MSE (norm) ──')
    print(f'  {"gate":>8}  {"train loss":>12}  {"val loss":>12}')
    for gv in [0.0, 0.1, 0.3, 0.5, 0.8, 0.99]:
        lt = loss_at_gate(model, train_batch, gv)
        lv = loss_at_gate(model, val_batch, gv)
        print(f'  {gv:>8.2f}  {lt:>12.6f}  {lv:>12.6f}')
    print('  → si la loss NO baja al abrir el gate, la escena no aporta'
          ' (gate cerrado es óptimo)')


# ── Prueba C: predictor de velocidad constante (linealidad) ──────────────────
@torch.no_grad()
def test_linearity(dataset):
    """Compara el error de un extrapolador lineal (velocidad constante del
    histórico) contra el movimiento real. Si es bajo, el movimiento es ~lineal
    y la escena es redundante para horizonte corto."""
    print('\n── Prueba C: ¿el movimiento es lineal? (extrapolador vel. constante) ──')
    ade_lin, fde_lin = [], []
    for d in dataset:
        # trabajamos en espacio normalizado relativo (igual que el modelo)
        hist = d['obj_history_flat'].view(HISTORY_LEN, 3).numpy()
        fut  = d['obj_future_flat'].view(PRED_LEN, 3).numpy()
        # velocidad = media de diferencias del histórico
        vel = np.diff(hist, axis=0).mean(axis=0)   # (3,)
        last = hist[-1]
        pred = np.array([last + vel * (k + 1) for k in range(PRED_LEN)])
        dist = np.linalg.norm(pred[:, :2] - fut[:, :2], axis=1)
        ade_lin.append(dist.mean())
        fde_lin.append(dist[-1])
    print(f'  ADE extrapolador lineal (norm): {np.mean(ade_lin):.4f}')
    print(f'  FDE extrapolador lineal (norm): {np.mean(fde_lin):.4f}')
    print('  (en espacio normalizado; compara contra la loss MSE del modelo)')
    print('  → si el error lineal ya es bajo, el histórico basta para 0.5s')


def main():
    print(f'Dispositivo: {DEVICE}')
    dataset = get_dataset()
    model = build_model(CKPT_AUG)

    train_batch = collate(dataset, TRAIN_SCENES)
    val_batch   = collate(dataset, VAL_SCENES)
    print(f'\nTrain: {train_batch[0].shape[0]} muestras | '
          f'Val: {val_batch[0].shape[0]} muestras')

    test_gradient(model, train_batch)
    test_forced_gate(model, train_batch, val_batch)
    test_linearity(dataset)


if __name__ == '__main__':
    main()
