# _clean10_base.py — parámetros compartidos del experimento waymo_clean FASE 1
# Protocolo de Claudine: 10 escenas (luego 100, luego 1000). 8 train / 2 val.
# No ejecutable por sí solo; lo importan baseline/gated/nogate via _base_.

history_len   = 5
pred_len      = 30          # 3.0s de horizonte
voxel_res     = 2.0
spatial_range = [-10, 10, -10, 10, -2, 4]
num_voxels    = 10 * 10 * 3   # 300

DATA_ROOT = '/home/lcad/lidar_sweep_viewer/waymo_clean'

# 10 escenas limpias (subconjunto de las 25): 8 train / 2 val
train_scenes = [
    '2a81f5233075e987', '2e41fe6faf5cd2ea', '367b072edc9822ea', '394e61f27c2a1700',
    '4014ae5bcda2726f', '41692b0ec7ff4123', '4a2ef30000d19d90', '4b60f9400a30ceaf',
]
val_scenes = [
    '7e2f727866c69ea0', '82f90331a1dfe968',
]
