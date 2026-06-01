import torch
import matplotlib.pyplot as plt
from mmpretrain.datasets import TrajectoryDataset
from mmpretrain.models.trajectory_pred.trajectory_model_attn import TrajectoryModelWithAttention
from mmengine.config import Config
from mmengine.runner import load_checkpoint

# Configuración (ajusta según el modelo que quieras mostrar)
cfg = Config.fromfile('configs/sapiens_mae/lidar/trajectory_attn_overfit.py')
model_args = {k: v for k, v in cfg.model.items() if k != 'type'}
model = TrajectoryModelWithAttention(**model_args)
checkpoint_path = 'work_dirs/trajectory_attn_overfit/epoch_200.pth'  # o el que tengas
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

# Tomar el primer objeto
data = dataset[0]
inputs = data['inputs'].unsqueeze(0)
obj_history_flat = data['obj_history_flat'].unsqueeze(0)
obj_future_flat = data['obj_future_flat'].unsqueeze(0)
norm_mean = data['norm_mean'].unsqueeze(0)
norm_std = data['norm_std'].unsqueeze(0)

with torch.no_grad():
    pred_flat = model(inputs, obj_history_flat, mode='predict')
    pred = pred_flat.view(-1, 5, 3) * norm_std + norm_mean
    target = obj_future_flat.view(-1, 5, 3) * norm_std + norm_mean

pred = pred.squeeze(0).numpy()
target = target.squeeze(0).numpy()

# Graficar
plt.figure(figsize=(6,6))
plt.plot(target[:,0], target[:,1], 'go-', label='Real', linewidth=2, markersize=8)
plt.plot(pred[:,0], pred[:,1], 'ro-', label='Predicho', linewidth=2, markersize=8)
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.legend()
plt.title('Trayectoria de objeto (waymo_10)')
plt.grid(True)
plt.savefig('trayectoria_ejemplo.png', dpi=150)
plt.close()
print("Gráfica guardada como trayectoria_ejemplo.png")
