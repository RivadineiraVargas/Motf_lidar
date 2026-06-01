import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mmpretrain.datasets import TrajectoryDataset
from mmpretrain.models.trajectory_pred.baseline_model import BaselineTrajectoryModel
from mmengine.config import Config
from mmengine.runner import load_checkpoint
import os

# Configuração (igual ao script anterior)
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

with torch.no_grad():
    if USAR_BASELINE:
        pred_flat = model(obj_history_flat, mode='predict')
    else:
        pred_flat = model(inputs, obj_history_flat, mode='predict')
    pred_rel = pred_flat.view(-1, 5, 3) * norm_std + norm_mean
    target_rel = obj_future_flat.view(-1, 5, 3) * norm_std + norm_mean

pred_rel = pred_rel.squeeze(0).numpy()
target_rel = target_rel.squeeze(0).numpy()

pose_path = '/home/lcad/lidar_sweep_viewer/waymo_10/poses/58d5f1b9e6a1a2f7/0.txt'
pose = np.loadtxt(pose_path)

target_hom = np.hstack([target_rel, np.ones((5,1))])
pred_hom = np.hstack([pred_rel, np.ones((5,1))])
target_global = (pose @ target_hom.T).T[:, :3]
pred_global = (pose @ pred_hom.T).T[:, :3]

lidar_path = '/home/lcad/lidar_sweep_viewer/waymo_10/bin_files/58d5f1b9e6a1a2f7/0.bin'
points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
lidar_sensor = points[:, :3]
lidar_hom = np.hstack([lidar_sensor, np.ones((lidar_sensor.shape[0],1))])
lidar_global = (pose @ lidar_hom.T).T[:, :3]

# Filtrar pontos próximos à trajetória
centro = target_global.mean(axis=0)[:2]
distancias = np.linalg.norm(lidar_global[:, :2] - centro, axis=1)
raio = 200
mask = distancias < raio
lidar_filtrado = lidar_global[mask]
step = max(1, len(lidar_filtrado) // 5000)  # <-- definição de step


# Configurar a animação
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(lidar_filtrado[::step, 0], lidar_filtrado[::step, 1], 
           c='lightgray', s=1, alpha=0.5, label='LiDAR')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Evolución de la trayectoria')
ax.grid(True)
ax.axis('equal')

# Ajustar límites con margen
x_min = min(target_global[:,0].min(), pred_global[:,0].min()) - 20
x_max = max(target_global[:,0].max(), pred_global[:,0].max()) + 20
y_min = min(target_global[:,1].min(), pred_global[:,1].min()) - 20
y_max = max(target_global[:,1].max(), pred_global[:,1].max()) + 20
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Líneas vacías para la animación
line_real, = ax.plot([], [], 'b-o', linewidth=2, markersize=9, label='Real')
line_pred, = ax.plot([], [], 'r-o', linewidth=2, markersize=8, label='Predicho')
#ax.legend()

def init():
    line_real.set_data([], [])
    line_pred.set_data([], [])
    return line_real, line_pred

def update(frame):
    # frame va de 1 a 5 (o hasta el total de puntos)
    line_real.set_data(target_global[:frame, 0], target_global[:frame, 1])
    line_pred.set_data(pred_global[:frame, 0], pred_global[:frame, 1])
    return line_real, line_pred

ani = animation.FuncAnimation(fig, update, frames=5, init_func=init,
                              interval=500, blit=True, repeat=True)
ani.save('trayectoria.gif', writer='pillow', fps=2)

with open('/home/lcad/lidar_sweep_viewer/sapiens/pretrain/trajectory.txt', 'w') as f:
    f.write('# real_x real_y real_z pred_x pred_y pred_z\n')
    for i in range(5):
        f.write(f"{target_global[i,0]} {target_global[i,1]} {target_global[i,2]} {pred_global[i,0]} {pred_global[i,1]} {pred_global[i,2]}\n")
print("Archivo trajectory.txt generado")

print("✅ GIF guardado como 'trayectoria.gif'")
