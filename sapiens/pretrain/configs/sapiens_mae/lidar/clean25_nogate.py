# clean25_nogate.py — MOTF SIN gate (escena siempre activa), horizonte 3s, datos limpios (waymo_clean, 25 escenas)
_base_ = ['../../_base_/default_runtime.py', '_clean25_base.py']

custom_imports = dict(
    imports=[
        'mmpretrain.datasets.trajectory_dataset',
        'mmpretrain.models.backbones.mae_vit_4d',
        'mmpretrain.models.trajectory_pred.trajectory_model_attn',
    ],
    allow_failed_imports=False
)

history_len   = 5
pred_len      = 30
voxel_res     = 2.0
spatial_range = [-10, 10, -10, 10, -2, 4]
num_voxels    = 300

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='TrajectoryDataset',
        data_root='/home/lcad/lidar_sweep_viewer/waymo_clean',
        sequence_len=history_len + pred_len,
        history_len=history_len,
        pred_len=pred_len,
        voxel_res=voxel_res,
        spatial_range=spatial_range,
        augment=True,
        scenes=[
            '2a81f5233075e987', '2e41fe6faf5cd2ea', '367b072edc9822ea', '394e61f27c2a1700',
            '4014ae5bcda2726f', '41692b0ec7ff4123', '4a2ef30000d19d90', '4b60f9400a30ceaf',
            '7e2f727866c69ea0', '82f90331a1dfe968', '8e0342468563ae5e', '92ab54c34f237728',
            '9e897ff552287bea', '9ea216a54ee07b49', '9fffe68876965f2e', 'a20f67087b9a288',
            'aaccfa0a1132fb83', 'adce80bac21c1895', 'ae3d6f946b8e7871', 'd2399ea6a028ecb2',
        ],
    ),
)

model = dict(
    type='TrajectoryModelWithAttention',
    encoder=dict(
        type='MAEViT4D',
        history_len=history_len,
        embed_dim=1024,
        num_tokens=num_voxels,
        arch='sapiens_0.3b',
        final_norm=True,
        mask_ratio=0.75,
    ),
    history_len=history_len,
    pred_len=pred_len,
    embed_dim=1024,
    num_heads=8,
    hidden_dim=512,
    scene_dim=64,
    freeze_encoder=True,
    use_gate=False,
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=1e-4)
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100)

default_hooks = dict(
    checkpoint=dict(interval=50, max_keep_ckpts=2),
    logger=dict(interval=20),
)

work_dir = './work_dirs/clean25_nogate'
load_from = './work_dirs/mae_encoder_pretrained.pth'
