import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mmpretrain.datasets import TrajectoryDataset
from mmpretrain.models.trajectory_pred.baseline_model import BaselineTrajectoryModel
from mmengine.config import Config
from mmengine.runner import load_checkpoint
import os

# Configuración (ajusta rutas si es necesario)
config_file = 'configs/sapiens_mae/lidar/baseline_overfit.py'
checkpoint_dir = 'work_dirs/baseline_overfit_norm'
from mmpretrain.models.trajectory_pred.baseline_model import BaselineTrajectoryModel as Modelo

cfg = Config.fromfile(config_file)
model_args = {k: v for k, v in cfg.model.items() if k != 'type'}
model = Modelo(**model_args)
checkpoint_path = os.path.join(checkpoint_dir, 'epoch_200.pth')
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

data = dataset[0]
inputs = data['inputs'].unsqueeze(0)
obj_history_flat = data['obj_history_flat'].unsqueeze(0)
obj_future_flat = data['obj_future_flat'].unsqueeze(0)
norm_mean = data['norm_mean'].unsqueeze(0)
norm_std = data['norm_std'].unsqueeze(0)

# Predicción
with torch.no_grad():
    pred_flat = model(obj_history_flat, mode='predict')
    pred_rel = pred_flat.view(-1, 5, 3) * norm_std + norm_mean
    target_rel = obj_future_flat.view(-1, 5, 3) * norm_std + norm_mean

pred_rel = pred_rel.squeeze(0).numpy()
target_rel = target_rel.squeeze(0).numpy()

# Cargar pose y transformar a global
pose_path = '/home/lcad/lidar_sweep_viewer/waymo_10/poses/58d5f1b9e6a1a2f7/0.txt'
pose = np.loadtxt(pose_path)

target_hom = np.hstack([target_rel, np.ones((5,1))])
pred_hom = np.hstack([pred_rel, np.ones((5,1))])
target_global = (pose @ target_hom.T).T[:, :3]
pred_global = (pose @ pred_hom.T).T[:, :3]

# Cargar y transformar LiDAR
lidar_path = '/home/lcad/lidar_sweep_viewer/waymo_10/bin_files/58d5f1b9e6a1a2f7/0.bin'
points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
lidar_sensor = points[:, :3]
lidar_hom = np.hstack([lidar_sensor, np.ones((lidar_sensor.shape[0],1))])
lidar_global = (pose @ lidar_hom.T).T[:, :3]

# Filtrar puntos cerca de la trayectoria
centro = np.mean([target_global, pred_global], axis=(0,1))[:2]
distancias = np.linalg.norm(lidar_global[:, :2] - centro, axis=1)
mask = distancias < 150
lidar_filtrado = lidar_global[mask]
print(f"Puntos LiDAR usados: {len(lidar_filtrado)}")

# Submuestreo para visualización
step = max(1, len(lidar_filtrado) // 5000)

# Crear figura
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(lidar_filtrado[::step, 0], lidar_filtrado[::step, 1], 
           c='lightgray', s=1, alpha=0.5, label='LiDAR')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Predicción de trayectoria (baseline) - 5 puntos futuros')
ax.grid(True, linestyle='--', alpha=0.5)
ax.axis('equal')

# Ajustar zoom
margin = 15
x_min = min(target_global[:,0].min(), pred_global[:,0].min()) - margin
x_max = max(target_global[:,0].max(), pred_global[:,0].max()) + margin
y_min = min(target_global[:,1].min(), pred_global[:,1].min()) - margin
y_max = max(target_global[:,1].max(), pred_global[:,1].max()) + margin
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Dibujar trayectorias completas (estático)
ax.plot(target_global[:,0], target_global[:,1], 'g--x', linewidth=3, markersize=10, label='Real')
ax.plot(pred_global[:,0], pred_global[:,1], 'r-o', linewidth=3, markersize=6, label='Predicho')
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig('presentacion_trayectoria.png', dpi=150)
print(" Imagen guardada como 'presentacion_trayectoria.png'")

# Crear animación (opcional, si quieres video)
from matplotlib.animation import FuncAnimation
fig2, ax2 = plt.subplots(figsize=(10, 10))
ax2.scatter(lidar_filtrado[::step, 0], lidar_filtrado[::step, 1], 
            c='lightgray', s=1, alpha=0.5, label='LiDAR')
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
ax2.set_title('Evolución de la predicción')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.axis('equal')
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(y_min, y_max)
line_real, = ax2.plot([], [], 'g--x', linewidth=3, markersize=10, label='Real')
line_pred, = ax2.plot([], [], 'r-o', linewidth=3, markersize=6, label='Predicho')
ax2.legend(loc='upper left')
time_text = ax2.text(0.02, 0.98, '', transform=ax2.transAxes, fontsize=12, verticalalignment='top')

def init():
    line_real.set_data([], [])
    line_pred.set_data([], [])
    time_text.set_text('')
    return line_real, line_pred, time_text

def update(frame):
    frame += 1
    line_real.set_data(target_global[:frame, 0], target_global[:frame, 1])
    line_pred.set_data(pred_global[:frame, 0], pred_global[:frame, 1])
    time_text.set_text(f'Instante {frame}/5')
    return line_real, line_pred, time_text

ani = FuncAnimation(fig2, update, frames=5, init_func=init, interval=1000, blit=True)
ani.save('presentacion_animacion.mp4', writer='ffmpeg', fps=1, dpi=150)
print("Video guardado como 'presentacion_animacion.mp4'")
