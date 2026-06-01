import torch
import numpy as np
import matplotlib.pyplot as plt
from mmpretrain.datasets import TrajectoryDataset
from mmpretrain.models.trajectory_pred import TrajectoryPredictionModel
from mmengine.config import Config
from mmengine.runner import load_checkpoint
import os

# Configuración
cfg = Config.fromfile('configs/sapiens_mae/lidar/trajectory_pred_overfit.py')
model_args = {k: v for k, v in cfg.model.items() if k != 'type'}
model = TrajectoryPredictionModel(**model_args)
checkpoint_path = 'work_dirs/trajectory_overfit_v2/epoch_200.pth'
load_checkpoint(model, checkpoint_path, map_location='cpu')
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

# Métricas
ade_list = []
fde_list = []

# Crear carpeta para gráficas
os.makedirs('evaluation_plots', exist_ok=True)

for idx in range(len(dataset)):
    data = dataset[idx]
    inputs = data['inputs'].unsqueeze(0)
    obj_history_flat = data['obj_history_flat'].unsqueeze(0)
    obj_future_flat = data['obj_future_flat'].unsqueeze(0)

    with torch.no_grad():
        pred_flat = model(inputs, obj_history_flat)
        pred = pred_flat.view(-1, 5, 3).squeeze(0).numpy()
        target = obj_future_flat.view(-1, 5, 3).squeeze(0).numpy()

    # Calcular ADE (Average Displacement Error)
    distances = np.linalg.norm(pred - target, axis=1)  # (5,)
    ade = np.mean(distances)
    fde = distances[-1]
    ade_list.append(ade)
    fde_list.append(fde)

    print(f"Objeto {idx}: ADE={ade:.4f}, FDE={fde:.4f}")

    # Graficar
    plt.figure()
    plt.plot(target[:,0], target[:,1], 'go-', label='Real')
    plt.plot(pred[:,0], pred[:,1], 'ro-', label='Predicho')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title(f'Trayectoria objeto {idx} - ADE={ade:.4f}')
    plt.savefig(f'evaluation_plots/obj_{idx}.png')
    plt.close()

print(f"ADE medio: {np.mean(ade_list):.4f} ± {np.std(ade_list):.4f}")
print(f"FDE medio: {np.mean(fde_list):.4f} ± {np.std(fde_list):.4f}")
