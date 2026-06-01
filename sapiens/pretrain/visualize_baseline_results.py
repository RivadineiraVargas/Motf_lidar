import torch
import numpy as np
import matplotlib.pyplot as plt
from mmpretrain.datasets import TrajectoryDataset
from mmpretrain.models.trajectory_pred.baseline_model import BaselineTrajectoryModel
from mmengine.runner import load_checkpoint

# Cargar modelo baseline (entrenado)
model = BaselineTrajectoryModel(history_len=5, pred_len=5)
checkpoint = torch.load('work_dirs/baseline_overfit_norm/epoch_200.pth', map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Dataset
dataset = TrajectoryDataset(
    data_root='/home/lcad/lidar_sweep_viewer/waymo_10',
    sequence_len=10,
    history_len=5,
    pred_len=5,
    voxel_res=2.0,
    spatial_range=[-10,10,-10,10,-2,4]
)

# Seleccionar algunos objetos para visualizar
indices = [0, 5, 10, 15]  # objetos con diferentes comportamientos

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
for idx, ax in zip(indices, axs.ravel()):
    data = dataset[idx]
    obj_history_flat = data['obj_history_flat'].unsqueeze(0)
    obj_future_flat = data['obj_future_flat'].unsqueeze(0)
    norm_mean = data['norm_mean']
    norm_std = data['norm_std']

    with torch.no_grad():
        pred_flat = model(obj_history_flat, mode='predict')
        pred = pred_flat.view(5,3) * norm_std + norm_mean
        target = obj_future_flat.view(5,3) * norm_std + norm_mean

    pred = pred.numpy()
    target = target.numpy()
    
    ax.plot(target[:,0], target[:,1], 'go-', label='Real', linewidth=2)
    ax.plot(pred[:,0], pred[:,1], 'ro-', label='Predicho', linewidth=2)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Objeto {idx}')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')

plt.tight_layout()
plt.savefig('baseline_trajectories.png', dpi=150)
plt.show()
