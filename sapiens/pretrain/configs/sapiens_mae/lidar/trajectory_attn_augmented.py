# trajectory_attn_augmented.py
# MOTF gated + augmentación por rotación (0/90/180/270° + flip XY).
# Mismo split 8 train / 2 val que multiescena.
# Con augment=True el dataset multiplica la variedad efectiva ~8x por época.
_base_ = ['../../_base_/default_runtime.py']

custom_imports = dict(
    imports=[
        'mmpretrain.datasets.trajectory_dataset',
        'mmpretrain.models.backbones.mae_vit_4d',
        'mmpretrain.models.trajectory_pred.trajectory_model_attn',
    ],
    allow_failed_imports=False
)

history_len   = 5
pred_len      = 5
voxel_res     = 2.0
spatial_range = [-10, 10, -10, 10, -2, 4]
num_voxels    = 10 * 10 * 3   # 300

train_scenes = [
    '1a01176ef4830499', '1a1918cd002b6a94', '1a64c26c56412fc5',
    '1b20914ea0bc28bd', '1b88ea97e03be2bf', '1ba702a85f4fa15c',
    '1bb5aa0c95f2ce0e', '58d5f1b9e6a1a2f7',
]
val_scenes = ['1a85be8ad06a056c', '1d9b8e390bed186f']

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='TrajectoryDataset',
        data_root='/home/lcad/lidar_sweep_viewer/waymo_10',
        sequence_len=history_len + pred_len,
        history_len=history_len,
        pred_len=pred_len,
        voxel_res=voxel_res,
        spatial_range=spatial_range,
        scenes=train_scenes,
        augment=True,
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
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=1e-4)
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=500)

default_hooks = dict(
    checkpoint=dict(interval=100, max_keep_ckpts=2),
    logger=dict(interval=10),
)

work_dir = './work_dirs/trajectory_attn_augmented'
load_from = './work_dirs/mae_encoder_pretrained.pth'
