# mae_lidar_10_overfit.py — versão corrigida
_base_ = ['../../_base_/default_runtime.py']

custom_imports = dict(
    imports=[
        'mmpretrain.models.backbones.mae_vit_4d',
        'mmpretrain.datasets.lidar_sequence',
        'mmpretrain.models.selfsup.mae_4d',
        'mmpretrain.models.heads.mae_head_4d',
    ],
    allow_failed_imports=False
)

# ── Parâmetros LiDAR ──────────────────────────────────────────────
history_len   = 5
sequence_len  = 10
voxel_res     = 2.0
spatial_range = [-10, 10, -10, 10, -2, 4]

grid_x     = int((spatial_range[1] - spatial_range[0]) / voxel_res)
grid_y     = int((spatial_range[3] - spatial_range[2]) / voxel_res)
grid_z     = int((spatial_range[5] - spatial_range[4]) / voxel_res)
num_voxels = grid_x * grid_y * grid_z   # 300

embed_dim  = 1024
mask_ratio = 0.75

# ── Dataset ───────────────────────────────────────────────────────
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='LidarSequenceDataset',
        data_root='/home/lcad/lidar_sweep_viewer/waymo_10',
        sequence_len=sequence_len,
        history_len=history_len,
        voxel_res=voxel_res,
        spatial_range=spatial_range,
        mask_ratio=mask_ratio,
        pipeline=[],
    )
)

# ── Preprocessor — nível superior, FORA do model ─────────────────
data_preprocessor = dict(type='BaseDataPreprocessor')

# ── Modelo ────────────────────────────────────────────────────────
model = dict(
    type='MAE4D',
    backbone=dict(
        type='MAEViT4D',
        history_len=history_len,
        embed_dim=embed_dim,
        num_tokens=num_voxels,
        arch='sapiens_0.3b',
        mask_ratio=mask_ratio,
        final_norm=True,
    ),
    neck=dict(
        type='MAEPretrainDecoder',
        num_patches=num_voxels,
        patch_size=1,
        in_chans=history_len,
        embed_dim=embed_dim,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
        init_cfg=None,
    ),
    head=dict(
        type='MAEPretrainHead4D',
        history_len=history_len,
        in_channels=512,
    ),
    init_cfg=None,
)

# ── Otimizador ────────────────────────────────────────────────────
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05),
    clip_grad=dict(max_norm=3.0, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'bias': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        }
    )
)

# ── Runtime ───────────────────────────────────────────────────────
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=4000)

default_hooks = dict(
    checkpoint=dict(interval=200, max_keep_ckpts=3),
    logger=dict(interval=50),
)

work_dir   = './work_dirs/mae_lidar_10_overfit'
randomness = dict(seed=0)
# resume = True

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)