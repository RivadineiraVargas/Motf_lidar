import open3d as o3d
import numpy as np
import torch
from mmpretrain.datasets import TrajectoryDataset
from mmpretrain.models.trajectory_pred.baseline_model import BaselineTrajectoryModel

# Cargar modelo (opcional, podemos solo dibujar las reales)
dataset = TrajectoryDataset(...)  # mismo que antes

# Tomar un objeto
data = dataset[0]
scene_bin = f"/home/lcad/lidar_sweep_viewer/waymo_10/bin_files/58d5f1b9e6a1a2f7/0.bin"
points = np.fromfile(scene_bin, dtype=np.float32).reshape(-1, 4)

# Crear nube de puntos
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])
pcd.colors = o3d.utility.Vector3dVector(np.tile([0.5,0.5,0.5], (points.shape[0],1)))

# Trayectoria real (desnormalizada)
target = data['obj_future_flat'].view(5,3) * data['norm_std'] + data['norm_mean']
target = target.numpy()

# Línea de la trayectoria
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(target)
line_set.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(4)])
line_set.colors = o3d.utility.Vector3dVector([[0,1,0]]*4)

# Visualizar
o3d.visualization.draw_geometries([pcd, line_set])
