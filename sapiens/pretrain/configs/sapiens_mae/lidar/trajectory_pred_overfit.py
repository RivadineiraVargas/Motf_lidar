_base_ = [
    '../../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=[
        'mmpretrain.models.backbones.mae_vit_4d',
        'mmpretrain.datasets.lidar_sequence',
        'mmpretrain.datasets.trajectory_dataset',
        'mmpretrain.models.selfsup.mae_4d',
        'mmpretrain.models.trajectory_pred.trajectory_model'
    ],
    allow_failed_imports=False)

# Parámetros
history_len = 5
pred_len = 5
voxel_res = 2.0
spatial_range = [-10, 10, -10, 10, -2, 4]

grid_x = int((spatial_range[1] - spatial_range[0]) / voxel_res)
grid_y = int((spatial_range[3] - spatial_range[2]) / voxel_res)
grid_z = int((spatial_range[5] - spatial_range[4]) / voxel_res)
num_voxels = grid_x * grid_y * grid_z

# Dataset
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='TrajectoryDataset',
        data_root='/home/lcad/lidar_sweep_viewer/waymo_10',
        history_len=history_len,
        sequence_len=10,
        pred_len=pred_len,
        voxel_res=voxel_res,
        spatial_range=spatial_range,
        pipeline=[]
    )
)

# Modelo: encoder pre-entrenado + decoder de trayectorias
# Cargamos el encoder desde un checkpoint (ajusta la ruta)
#load_from = '/home/lcad/lidar_sweep_viewer/sapiens/pretrain/work_dirs/lidar_overfit/epoch_4000.pth'  # o el último checkpoint

#load_from = '/home/lcad/lidar_sweep_viewer/sapiens/pretrain/work_dirs/mae_pretrain/epoch_200.pth'  # o el último

model = dict(
    type='TrajectoryPredictionModel',
    encoder=dict(
        type='MAEViT4D',
        history_len=history_len,
        embed_dim=1024,
        num_tokens=num_voxels,
        arch='sapiens_0.3b',
        img_size=1024,
        patch_size=16,
        #in_chans=history_len,
        final_norm=True,
        mask_ratio=0.75,   # no se usa en inferencia, pero necesario para construcción
        #init_cfg=dict(type='Pretrained', checkpoint=load_from)  # Cargar pesos del encoder
    ),
    history_len=history_len,
    pred_len=pred_len,
    in_channels=1024,   # según la salida real del encoder (verificar)
    hidden_dim=512,
)

# Optimizador
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05),
    clip_grad=dict(max_norm=3.0, norm_type=2)
)

# Runtime
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200)  # pocas épocas para overfit
default_hooks = dict(
    checkpoint=dict(interval=50, max_keep_ckpts=2),
    logger=dict(interval=1),
    runtime_info=None   # desactivado por ahora
)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

work_dir = './work_dirs/trajectory_overfit_v2'
