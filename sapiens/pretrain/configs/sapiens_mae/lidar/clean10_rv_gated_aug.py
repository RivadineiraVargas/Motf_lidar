# clean10_rv_gated_aug.py — MOTF gated sobre RANGE-VIEW (track range-view).
# Mesmo modelo/gate/horizonte que o track de vóxels, mas a cena é a range-view
# (tokens 128x1280) em vez de vóxels (300x5). Encoder reusado (Linear 1280->1024).
_base_ = ['../../_base_/default_runtime.py']

custom_imports = dict(
    imports=[
        'mmpretrain.datasets.range_view',
        'mmpretrain.models.backbones.mae_vit_4d',
        'mmpretrain.models.trajectory_pred.trajectory_model_attn',
    ],
    allow_failed_imports=False
)

history_len  = 5            # histórico do objeto (posições)
pred_len     = 30           # 3 s
scene_frames = 5            # frames de range-view empilhados
n_tokens     = 128
tok_dim      = 16 * 16 * scene_frames    # 1280

train_scenes = [
    '2a81f5233075e987', '2e41fe6faf5cd2ea', '367b072edc9822ea', '394e61f27c2a1700',
    '4014ae5bcda2726f', '41692b0ec7ff4123', '4a2ef30000d19d90', '4b60f9400a30ceaf',
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RangeViewTrajectoryDataset',
        data_root='/home/lcad/lidar_sweep_viewer/waymo_clean',
        sequence_len=history_len + pred_len,
        history_len=history_len,
        pred_len=pred_len,
        voxel_res=2.0,                       # (ignorado pela range-view; req. p/ __init__)
        spatial_range=[-10, 10, -10, 10, -2, 4],
        scenes=train_scenes,
        augment=True,   # azimut-shift + rotación consistente de trayectoria
    ),
)

model = dict(
    type='TrajectoryModelWithAttention',
    encoder=dict(
        type='MAEViT4D',
        history_len=tok_dim,          # = patch_dim: Linear(1280 -> embed_dim)
        embed_dim=1024,
        num_tokens=n_tokens,          # 128
        arch='sapiens_0.3b',
        final_norm=True,
        mask_ratio=0.75,
    ),
    history_len=history_len,          # 5 (histórico do objeto)
    pred_len=pred_len,
    embed_dim=1024,
    num_heads=8,
    hidden_dim=512,
    scene_dim=64,
    freeze_encoder=True,
    use_gate=True,
    gate_init=0.5,
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=1e-4)
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100)
default_hooks = dict(checkpoint=dict(interval=50, max_keep_ckpts=2),
                     logger=dict(interval=20))

work_dir = './work_dirs/clean10_rv_gated_aug'
load_from = './work_dirs/mae_encoder_rangeview.pth'
