import torch
import numpy as np
import matplotlib.pyplot as plt
from mmpretrain.datasets import TrajectoryDataset
from mmpretrain.models.trajectory_pred.baseline_model import BaselineTrajectoryModel
from mmengine.config import Config
from mmengine.runner import load_checkpoint
import os

# ============================================
# 1. CONFIGURAÇÃO
# ============================================
USAR_BASELINE = True

if USAR_BASELINE:
    config_file = 'configs/sapiens_mae/lidar/baseline_overfit.py'
    checkpoint_dir = 'work_dirs/baseline_overfit_norm'
    from mmpretrain.models.trajectory_pred.baseline_model import BaselineTrajectoryModel as Modelo
else:
    config_file = 'configs/sapiens_mae/lidar/trajectory_attn_overfit.py'
    checkpoint_dir = 'work_dirs/trajectory_attn_overfit'
    from mmpretrain.models.trajectory_pred.trajectory_model_attn import TrajectoryModelWithAttention as Modelo

cfg = Config.fromfile(config_file)
model_args = {k: v for k, v in cfg.model.items() if k != 'type'}
model = Modelo(**model_args)
checkpoint_path = os.path.join(checkpoint_dir, 'epoch_200.pth')
load_checkpoint(model, checkpoint_path, map_location='cpu')
model.eval()

# ============================================
# 2. CARREGAR DATASET
# ============================================
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

# ============================================
# 3. PREDIÇÃO
# ============================================
with torch.no_grad():
    if USAR_BASELINE:
        pred_flat = model(obj_history_flat, mode='predict')
    else:
        pred_flat = model(inputs, obj_history_flat, mode='predict')
    pred_rel = pred_flat.view(-1, 5, 3) * norm_std + norm_mean
    target_rel = obj_future_flat.view(-1, 5, 3) * norm_std + norm_mean

pred_rel = pred_rel.squeeze(0).numpy()
target_rel = target_rel.squeeze(0).numpy()

# ============================================
# 4. TRANSFORMAR PARA GLOBAL USANDO A POSE
# ============================================
pose_path = '/home/lcad/lidar_sweep_viewer/waymo_10/poses/58d5f1b9e6a1a2f7/0.txt'
pose = np.loadtxt(pose_path)

target_hom = np.hstack([target_rel, np.ones((5,1))])
pred_hom = np.hstack([pred_rel, np.ones((5,1))])
target_global = (pose @ target_hom.T).T[:, :3]
pred_global = (pose @ pred_hom.T).T[:, :3]

# ============================================
# 5. CARREGAR E TRANSFORMAR LIDAR
# ============================================
lidar_path = '/home/lcad/lidar_sweep_viewer/waymo_10/bin_files/58d5f1b9e6a1a2f7/0.bin'
points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
lidar_sensor = points[:, :3]
lidar_hom = np.hstack([lidar_sensor, np.ones((lidar_sensor.shape[0],1))])
lidar_global = (pose @ lidar_hom.T).T[:, :3]

# ============================================
# 6. FILTRAR LIDAR PRÓXIMO À TRAJETÓRIA
# ============================================
centro_x = np.mean([target_global[:,0].mean(), pred_global[:,0].mean()])
centro_y = np.mean([target_global[:,1].mean(), pred_global[:,1].mean()])
centro = np.array([centro_x, centro_y])
distancias = np.linalg.norm(lidar_global[:, :2] - centro, axis=1)
raio = 100  # metros
mask = distancias < raio
lidar_filtrado = lidar_global[mask]
print(f"Pontos LiDAR após filtro: {len(lidar_filtrado)} de {len(lidar_global)}")

# ============================================
# 7. VISUALIZAÇÃO
# ============================================
plt.figure(figsize=(10,8))

step = max(1, len(lidar_filtrado)//5000)
plt.scatter(lidar_filtrado[::step, 0], lidar_filtrado[::step, 1], 
            c='lightgray', s=1, alpha=0.5, label='LiDAR pontos')

plt.plot(target_global[:,0], target_global[:,1], 'g-o', linewidth=2, markersize=6, label='Real')
plt.plot(pred_global[:,0], pred_global[:,1], 'r-o', linewidth=2, markersize=4, label='Predito')

plt.scatter(target_global[0,0], target_global[0,1], c='green', s=200, marker='o', edgecolors='black')
plt.scatter(target_global[-1,0], target_global[-1,1], c='green', s=200, marker='s', edgecolors='black')
plt.scatter(pred_global[0,0], pred_global[0,1], c='red', s=200, marker='o', edgecolors='black')
plt.scatter(pred_global[-1,0], pred_global[-1,1], c='red', s=200, marker='s', edgecolors='black')

x_min = min(target_global[:,0].min(), pred_global[:,0].min()) - 5
x_max = max(target_global[:,0].max(), pred_global[:,0].max()) + 5
y_min = min(target_global[:,1].min(), pred_global[:,1].min()) - 5
y_max = max(target_global[:,1].max(), pred_global[:,1].max()) + 5

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Trajetória sobre nuvem LiDAR (zoom)')
plt.legend()
plt.grid(True)
plt.axis('equal')

plt.savefig('resultado_zoom.png', dpi=150, bbox_inches='tight')
print("✅ Imagem salva como 'resultado_zoom.png'")
