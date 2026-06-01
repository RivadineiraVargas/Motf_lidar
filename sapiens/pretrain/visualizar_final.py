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

# Precisamos também da história para mostrar a trajetória completa
# A história já está nos centros do dataset? Vamos carregar os centros completos do objeto
centers = data['centers']  # isso é uma lista de numpy arrays? Precisamos verificar. No dataset, centers é uma lista.
# Como centers é uma lista de numpy arrays, vamos convertê-la
centers = np.array(data['centers'])  # (10,3)

# Os centros já estão no sensor frame, sem normalização? No dataset, centers é guardado em sensor frame (após transformação).
# Vamos confirmar: no __getitem__, centers é a lista de centros absolutos em sensor frame (após poses). Portanto, podemos usar diretamente.
# Mas precisamos normalizar? centers são absolutos, e nossa trajetória futura predita também é absoluta (após ref_center + deslocamentos).
# Na verdade, target_rel e pred_rel são deslocamentos em relação ao primeiro frame. Para obter a trajetória absoluta completa, podemos fazer:
#   full_abs = ref_center + (deslocamentos acumulados). Mas temos os centros originais em centers.
# Vamos usar centers diretamente para a trajetória real completa.

# Portanto:
history_abs = centers[:5]    # primeiros 5 frames (história)
future_abs = centers[5:10]   # próximos 5 (futuro real)

# Para a trajetória predita, temos pred_rel que são deslocamentos em relação ao primeiro frame. Para obter absolutos:
ref_center = centers[0]  # primeiro frame
pred_abs = ref_center + np.cumsum(pred_rel, axis=0)  # integração dos deslocamentos (se forem incrementais)
# Mas nossos pred_rel são as posições futuras absolutas? Na verdade, no modelo, a saída são as posições absolutas (target_rel) no sistema normalizado. Após desnormalizar, temos deslocamentos? Não, target_rel já são as posições futuras absolutas no sensor frame (porque usamos ref_center? Vamos ver: no dataset, obj_future_flat são os centros normalizados, e depois de desnormalizar obtemos as posições relativas ao primeiro frame? Confuso. Melhor usar diretamente os centros do dataset.

# Na verdade, temos no dataset: centers são as posições absolutas em sensor frame. Então vamos usar future_abs como real. Para a predita, precisamos transformar pred_rel em absolutas. Como pred_rel foram obtidas pela desnormalização de pred_flat, e pred_flat foi treinado para prever obj_future_flat, que são os centros normalizados (deslocamentos). Portanto, pred_rel são deslocamentos em relação ao primeiro frame. Então pred_abs = ref_center + pred_rel.

ref_center = centers[0]  # (3,)
pred_abs = ref_center + pred_rel

# Agora temos:
# - history_abs: (5,3) trajetória passada real
# - future_abs: (5,3) trajetória futura real
# - pred_abs: (5,3) trajetória futura predita

# Transformar todos para global usando a pose
history_hom = np.hstack([history_abs, np.ones((5,1))])
future_hom = np.hstack([future_abs, np.ones((5,1))])
pred_hom = np.hstack([pred_abs, np.ones((5,1))])

history_global = (pose @ history_hom.T).T[:, :3]
future_global = (pose @ future_hom.T).T[:, :3]
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
centro_x = np.mean([history_global[:,0].mean(), future_global[:,0].mean(), pred_global[:,0].mean()])
centro_y = np.mean([history_global[:,1].mean(), future_global[:,1].mean(), pred_global[:,1].mean()])
centro = np.array([centro_x, centro_y])
distancias = np.linalg.norm(lidar_global[:, :2] - centro, axis=1)
raio = 150  # metros
mask = distancias < raio
lidar_filtrado = lidar_global[mask]
print(f"Pontos LiDAR após filtro: {len(lidar_filtrado)} de {len(lidar_global)}")

# ============================================
# 7. VISUALIZAÇÃO AVANÇADA
# ============================================
plt.figure(figsize=(12,10))

# Pontos LiDAR (cinza claro, pequenos)
step = max(1, len(lidar_filtrado)//8000)
plt.scatter(lidar_filtrado[::step, 0], lidar_filtrado[::step, 1], 
            c='lightgray', s=1, alpha=0.5, label='LiDAR pontos')

# Trajetória história (tracejado preto)
plt.plot(history_global[:,0], history_global[:,1], 'k--', linewidth=2, label='História (passado)')

# Trajetória futura real (azul sólido, linha grossa)
plt.plot(future_global[:,0], future_global[:,1], 'b-', linewidth=4, label='Futuro real (azul)')

# Trajetória futura predita (vermelho sólido, linha grossa)
plt.plot(pred_global[:,0], pred_global[:,1], 'r-', linewidth=4, label='Futuro predito (vermelho)')

# Marcadores de início e fim
plt.scatter(future_global[0,0], future_global[0,1], c='blue', s=200, marker='o', edgecolors='black', label='Início futuro real')
plt.scatter(future_global[-1,0], future_global[-1,1], c='blue', s=200, marker='s', edgecolors='black', label='Fim futuro real')
plt.scatter(pred_global[0,0], pred_global[0,1], c='red', s=200, marker='o', edgecolors='black', label='Início predito')
plt.scatter(pred_global[-1,0], pred_global[-1,1], c='red', s=200, marker='s', edgecolors='black', label='Fim predito')

# Ajustar zoom (margem de 30 metros)
x_min = min(history_global[:,0].min(), future_global[:,0].min(), pred_global[:,0].min()) - 30
x_max = max(history_global[:,0].max(), future_global[:,0].max(), pred_global[:,0].max()) + 30
y_min = min(history_global[:,1].min(), future_global[:,1].min(), pred_global[:,1].min()) - 30
y_max = max(history_global[:,1].max(), future_global[:,1].max(), pred_global[:,1].max()) + 30

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Trajetória completa: história, futuro real e predito')
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.axis('equal')

plt.savefig('resultado_final2.png', dpi=150, bbox_inches='tight')
print("✅ Imagem salva como 'resultado_final2.png'")
