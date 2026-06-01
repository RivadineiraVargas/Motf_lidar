import torch
import numpy as np
import matplotlib.pyplot as plt
from mmpretrain.datasets import TrajectoryDataset
from mmpretrain.models.trajectory_pred.baseline_model import BaselineTrajectoryModel
from mmengine.config import Config
from mmengine.runner import load_checkpoint
import os

# ============================================
# 1. CONFIGURACIÓN - AJUSTA SEGÚN TUS ARCHIVOS
# ============================================
# Elige qué modelo usar: baseline (recomendado) o atención
USAR_BASELINE = True  # Cambia a False si quieres usar el modelo con atención

if USAR_BASELINE:
    config_file = 'configs/sapiens_mae/lidar/baseline_overfit.py'
    checkpoint_dir = 'work_dirs/baseline_overfit_norm'
    from mmpretrain.models.trajectory_pred.baseline_model import BaselineTrajectoryModel as Modelo
else:
    config_file = 'configs/sapiens_mae/lidar/trajectory_attn_overfit.py'
    checkpoint_dir = 'work_dirs/trajectory_attn_overfit'
    from mmpretrain.models.trajectory_pred.trajectory_model_attn import TrajectoryModelWithAttention as Modelo

# Cargar configuración y modelo
cfg = Config.fromfile(config_file)
model_args = {k: v for k, v in cfg.model.items() if k != 'type'}
model = Modelo(**model_args)
checkpoint_path = os.path.join(checkpoint_dir, 'epoch_200.pth')  # Ajusta si usaste otra época
load_checkpoint(model, checkpoint_path, map_location='cpu')
model.eval()

# ============================================
# 2. CARGAR DATOS DEL PRIMER OBJETO
# ============================================
dataset = TrajectoryDataset(
    data_root='/home/lcad/lidar_sweep_viewer/waymo_10',
    sequence_len=10,
    history_len=5,
    pred_len=5,
    voxel_res=2.0,
    spatial_range=[-10,10,-10,10,-2,4]
)

data = dataset[0]  # primer objeto
inputs = data['inputs'].unsqueeze(0)
obj_history_flat = data['obj_history_flat'].unsqueeze(0)
obj_future_flat = data['obj_future_flat'].unsqueeze(0)
norm_mean = data['norm_mean'].unsqueeze(0)
norm_std = data['norm_std'].unsqueeze(0)
ref_center = data['ref_center'].numpy()

# ============================================
# 3. OBTENER PREDICCIÓN
# ============================================
with torch.no_grad():
    if USAR_BASELINE:
        pred_flat = model(obj_history_flat, mode='predict')
    else:
        pred_flat = model(inputs, obj_history_flat, mode='predict')
    pred = pred_flat.view(-1, 5, 3) * norm_std + norm_mean
    target = obj_future_flat.view(-1, 5, 3) * norm_std + norm_mean

pred_rel = pred.squeeze(0).numpy()
target_rel = target.squeeze(0).numpy()

pred_abs = pred_rel + ref_center
target_abs = target_rel + ref_center

# ============================================
# 4. CARGAR NUBE DE PUNTOS DEL PRIMER FRAME
# ============================================
scene_bin = os.path.join('/home/lcad/lidar_sweep_viewer/waymo_10/bin_files', '58d5f1b9e6a1a2f7')
points = np.fromfile(os.path.join(scene_bin, '0.bin'), dtype=np.float32).reshape(-1, 4)
# Tomamos solo las coordenadas x,y,z
lidar_points = points[:, :3]

distances = np.linalg.norm(lidar_points - ref_center, axis=1)
mask = distances < 30
filtered_points = lidar_points[mask]
print(f"Puntos LiDAR totales: {len(lidar_points)}, después de filtro: {len(filtered_points)}")
print("ref_center:", ref_center)
print("Pontos LiDAR - min x:", lidar_points[:,0].min(), "max x:", lidar_points[:,0].max())
print("Pontos LiDAR - min y:", lidar_points[:,1].min(), "max y:", lidar_points[:,1].max())
print("Distâncias ao ref_center - min:", distances.min(), "max:", distances.max())
print("Pontos após filtro:", len(filtered_points))
if len(filtered_points) == 0:
    print("Nenhum ponto próximo! Usando todos os pontos.")
    filtered_points = lidar_points
# ============================================
# 5. VISUALIZACIÓN 2D (VISTA SUPERIOR)
# ============================================
print("Forma de pred:", pred.shape)
print("Forma de target:", target.shape)
print("Primeros 3 puntos predichos:\n", pred[:3])
print("Primeros 3 puntos reales:\n", target[:3])
print("Rango X real:", target[:,0].min(), "-", target[:,0].max())
print("Rango Y real:", target[:,1].min(), "-", target[:,1].max())

plt.figure(figsize=(10, 8))

# Mostrar una muestra de puntos LiDAR (para no saturar)
plt.scatter(lidar_points[::10, 0], lidar_points[::10, 1], 
            c='lightgray', s=1, alpha=0.5, label='LiDAR puntos')

# Trayectorias
plt.plot(target_abs[:,0], target_abs[:,1], 'g-o', linewidth=2, markersize=5, label='Real')
plt.plot(pred_abs[:,0], pred_abs[:,1], 'r-o', linewidth=2, markersize=5, label='Predicho')

# Marcar inicio y fin
plt.scatter(target[0,0], target[0,1], c='green', s=200, marker='o', edgecolors='black', zorder=5)
plt.scatter(target[-1,0], target[-1,1], c='green', s=200, marker='s', edgecolors='black', zorder=5)
plt.scatter(pred[0,0], pred[0,1], c='red', s=200, marker='o', edgecolors='black', zorder=5)
plt.scatter(pred[-1,0], pred[-1,1], c='red', s=200, marker='s', edgecolors='black', zorder=5)

x_min = min(target[:,0].min(), pred[:,0].min()) - 5
x_max = max(target[:,0].max(), pred[:,0].max()) + 5
y_min = min(target[:,1].min(), pred[:,1].min()) - 5
y_max = max(target[:,1].max(), pred[:,1].max()) + 5

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Trayectoria sobre nube LiDAR (vista superior)')
plt.legend()
plt.grid(True)
plt.axis('equal')  # para que las proporciones sean reales

# Guardar imagen
plt.savefig('resultado_visual.png', dpi=150, bbox_inches='tight')
print("✅ Imagen guardada como 'resultado_visual.png'")
