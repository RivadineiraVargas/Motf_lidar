# _clean25_base.py — parámetros compartidos del experimento waymo_clean (25 escenas, 3s)
# No es un config ejecutable por sí solo; lo importan baseline/gated/nogate.

history_len   = 5
pred_len      = 30          # 3.0s de horizonte (vs 0.5s antes)
voxel_res     = 2.0
spatial_range = [-10, 10, -10, 10, -2, 4]
num_voxels    = 10 * 10 * 3   # 300

DATA_ROOT = '/home/lcad/lidar_sweep_viewer/waymo_clean'

# 25 escenas con LiDAR -> 20 train / 5 val
train_scenes = [
    '2a81f5233075e987', '2e41fe6faf5cd2ea', '367b072edc9822ea', '394e61f27c2a1700',
    '4014ae5bcda2726f', '41692b0ec7ff4123', '4a2ef30000d19d90', '4b60f9400a30ceaf',
    '7e2f727866c69ea0', '82f90331a1dfe968', '8e0342468563ae5e', '92ab54c34f237728',
    '9e897ff552287bea', '9ea216a54ee07b49', '9fffe68876965f2e', 'a20f67087b9a288',
    'aaccfa0a1132fb83', 'adce80bac21c1895', 'ae3d6f946b8e7871', 'd2399ea6a028ecb2',
]
val_scenes = [
    'db4edc9bd0c9d18c', 'e52c6a9366981ad', 'e75176fd226ea04a',
    'f2ca03b1434a27e4', 'f7cc90b8f4611d4d',
]
