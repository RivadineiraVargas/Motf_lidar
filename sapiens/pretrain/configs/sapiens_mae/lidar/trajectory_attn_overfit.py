# trajectory_attn_overfit.py — versão corrigida
_base_ = ['../../_base_/default_runtime.py']

custom_imports = dict(
    imports=[
        'mmpretrain.datasets.trajectory_dataset',
        'mmpretrain.models.backbones.mae_vit_4d',
        'mmpretrain.models.trajectory_pred.trajectory_model_attn',
    ],
    allow_failed_imports=False
)

# ── Parâmetros ────────────────────────────────────────────────────
history_len   = 5       # deve ser igual ao pré-treino
pred_len      = 5
voxel_res     = 2.0
spatial_range = [-10, 10, -10, 10, -2, 4]

grid_x     = int((spatial_range[1] - spatial_range[0]) / voxel_res)
grid_y     = int((spatial_range[3] - spatial_range[2]) / voxel_res)
grid_z     = int((spatial_range[5] - spatial_range[4]) / voxel_res)
num_voxels = grid_x * grid_y * grid_z   # 300

# ── Dataset ───────────────────────────────────────────────────────
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='TrajectoryDataset',
        data_root='/home/lcad/lidar_sweep_viewer/waymo_10',
        sequence_len=10,
        history_len=history_len,
        pred_len=pred_len,
        voxel_res=voxel_res,
        spatial_range=spatial_range,
    ),
)

# ── Modelo ────────────────────────────────────────────────────────
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

# ── Otimizador ────────────────────────────────────────────────────
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=1e-4)
)

# ── Runtime ───────────────────────────────────────────────────────
# Run completo: arquitetura gated (LayerNorm + proj 64 + gate). Objetivo:
# confirmar que chega a ~baseline (~0.44) sem colapsar.
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=500)

default_hooks = dict(
    checkpoint=dict(interval=100, max_keep_ckpts=2),
    logger=dict(interval=10),
)

work_dir = './work_dirs/trajectory_attn_gated'

load_from = './work_dirs/mae_encoder_pretrained.pth'