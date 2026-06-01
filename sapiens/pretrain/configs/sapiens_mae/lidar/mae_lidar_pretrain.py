_base_ = [
    '../../_base_/default_runtime.py',
]

custom_imports = dict(imports=['mmpretrain.models.backbones.mae_vit_4d', 'mmpretrain.datasets.lidar_sequence', 'mmpretrain.models.selfsup.mae_4d', 'mmpretrain.models.heads.mae_head_4d'], allow_failed_imports=False)
# Parámetros LiDAR
sequence_len = 10        # Era 200
history_len = 10         # 5 seg para pre-train
voxel_res = 2.0           # 0.5 metros por vóxel
spatial_range = [-10, 10, -10, 10, -2, 4]  # xmin, xmax, ymin, ymax, zmin, zmax

# Calcular dimensiones de la rejilla
grid_x = int((spatial_range[1] - spatial_range[0]) / voxel_res)
grid_y = int((spatial_range[3] - spatial_range[2]) / voxel_res)
grid_z = int((spatial_range[5] - spatial_range[4]) / voxel_res)
num_voxels = grid_x * grid_y * grid_z

# Modelo Sapiens (0.3b)
model_name = 'sapiens_0.3b'
embed_dim = 1024
num_layers = 24
mask_ratio = 0.75

# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='LidarSequenceDataset',
        data_root='/home/lcad/lidar_sweep_viewer/waymo_10',  # Ajusta a tu ruta
        sequence_len=sequence_len,
        history_len=history_len,
        voxel_res=voxel_res,
        spatial_range=spatial_range,
        mask_ratio=mask_ratio,
        pipeline=[]  # No necesitamos pipeline de imagen
    )
)

# -------------------------------------------------------------------
# Modelo
# -------------------------------------------------------------------
model = dict(
    type='MAE4D',
    backbone=dict(
        type='MAEViT4D',
        history_len=history_len,
        embed_dim=embed_dim,#embed_dim
        num_tokens=num_voxels,  # Importante para inicializar pos_embed
        arch=model_name, #'b'
        img_size=1024,
        patch_size=16,
        #depth=24,
        #in_chans=history_len,
        final_norm=True,
        mask_ratio=mask_ratio  # Se usa en el cálculo de pérdida
    ),
    neck=dict(
        type='MAEPretrainDecoder',
        #_delete_=True,
        embed_dim=embed_dim,
	num_patches=300,#num_voxels
        patch_size=16,  # No usado realmente, pero necesario para compatibilidad
        #num_patches=num_voxels,
	decoder_embed_dim=512,
	decoder_depth=8,
	decoder_num_heads=16,
	mlp_ratio=4.,
	init_cfg=None,  # Número de tokens
    ),
    head=dict(
        type='MAEPretrainHead4D',
        #loss=dict(type='PixelReconstructionLoss', criterion='L2'),
        history_len=history_len,
        #patch_size=16,
        #norm_pix=False,
        in_channels=768,##### embed_dim
        #init_cfg=None,

    ),
    init_cfg=None,
)

data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[0., 0., 0.],
    std=[1., 1., 1.],
    to_rgb=False)

# -------------------------------------------------------------------
# Optimizador
# -------------------------------------------------------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        weight_decay=0.05,
    ),
    clip_grad=dict(max_norm=3.0, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'bias': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0)
        }
    )
)

# -------------------------------------------------------------------
# Runtime
# -------------------------------------------------------------------
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=4000)

default_hooks = dict(
    checkpoint=dict(interval=200, max_keep_ckpts=2),
    logger=dict(interval=1),
    runtime_info=None, 
)

randomness = dict(seed=0, diff_rank_seed=True)
resume = True
auto_scale_lr = dict(enable=False)

# -------------------------------------------------------------------
# Entorno
# -------------------------------------------------------------------
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
