# GENERADO por run_fase1_cv.sh — pre-entrenamiento MAE 4D del fold 0.
# val RETENIDA del fold 0: ['7e2f727866c69ea0', '82f90331a1dfe968']
# El MAE de este fold se pre-entrena SOLO en sus 8 escenas de train;
# usarlo en otro fold seria FUGA auto-supervisada.
# mae_clean10_pretrain.py — Re-pretraining del MAE en 8 escenas train (protocolo waymo_10 de Claudine)
# Basado en mae_lidar_10_overfit.py (el config PROBADO que sí corre).
# Cambios: data_root=waymo_clean, shuffle=True, 1000 épocas, work_dir nuevo.
# IMPORTANTE: history_len=5 para que el encoder sea compatible con los
# trajectory models (que construyen el encoder con history_len=5).
#
# Lanzar:  conda activate sapiens_gpu
#          cd sapiens/pretrain
#          python tools/train.py configs/sapiens_mae/lidar/mae_clean10_pretrain.py
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

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),   # shuffle ON (25 escenas)
    dataset=dict(
        type='LidarSequenceDataset',
        data_root='/home/lcad/lidar_sweep_viewer/waymo_clean',   # DATOS LIMPIOS
        sequence_len=sequence_len,
        history_len=history_len,
        voxel_res=voxel_res,
        spatial_range=spatial_range,
        mask_ratio=mask_ratio,
        pipeline=[],
        scenes=[
            '2a81f5233075e987',
            '2e41fe6faf5cd2ea',
            '367b072edc9822ea',
            '394e61f27c2a1700',
            '4014ae5bcda2726f',
            '41692b0ec7ff4123',
            '4a2ef30000d19d90',
            '4b60f9400a30ceaf',
        ],
    )
)

data_preprocessor = dict(type='BaseDataPreprocessor')

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

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1000)

default_hooks = dict(
    checkpoint=dict(interval=200, max_keep_ckpts=3),
    logger=dict(interval=10),
)

work_dir   = './work_dirs/f1cv/mae_fold0'
randomness = dict(seed=0)
resume     = False

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
