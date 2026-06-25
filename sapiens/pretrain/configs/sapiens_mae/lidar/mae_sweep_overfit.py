# mae_sweep_overfit.py — Validación del ENCODER (plano de Claudine, Sec. 3-4).
# Arquitectura REDUCIDA (6 capas, embed 256) + overfit en N sweeps individuales.
# Objetivo: demostrar que el MAE aprende/overfittea (loss -> ~0) en pocos sweeps,
# con reconstrucción visualizable. Cambiar MAX_SWEEPS para 10 / 100 / 1000.
_base_ = ['../../_base_/default_runtime.py']

custom_imports = dict(
    imports=[
        'mmpretrain.models.backbones.mae_vit_4d',
        'mmpretrain.datasets.range_view',
        'mmpretrain.models.selfsup.mae_4d',
        'mmpretrain.models.heads.mae_head_4d',
    ],
    allow_failed_imports=False
)

MAX_SWEEPS = 10             # 10 / 100 / 1000 (plano de Claudine)
n_tokens   = 128
tok_dim    = 16 * 16       # 256 (1 frame, canal rango)
embed_dim  = 256           # encoder REDUCIDO
mask_ratio = 0.75

# Arquitectura pequeña (vs sapiens_0.3b de 24 capas/1024)
small_arch = dict(embed_dims=embed_dim, num_layers=6, num_heads=4,
                  feedforward_channels=embed_dim * 4)

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RangeSweepDataset',
        data_root='/home/lcad/lidar_sweep_viewer/waymo_clean',
        max_sweeps=MAX_SWEEPS,
        pipeline=[],
    )
)

data_preprocessor = dict(type='BaseDataPreprocessor')

model = dict(
    type='MAE4D',
    backbone=dict(
        type='MAEViT4D',
        history_len=tok_dim,          # patch_embed: Linear(256 -> embed_dim)
        embed_dim=embed_dim,
        num_tokens=n_tokens,
        arch=small_arch,              # ← arquitectura reducida
        mask_ratio=mask_ratio,
        final_norm=True,
    ),
    neck=dict(
        type='MAEPretrainDecoder',
        num_patches=n_tokens,
        patch_size=1,
        in_chans=tok_dim,
        embed_dim=embed_dim,
        decoder_embed_dim=128,
        decoder_depth=4,
        decoder_num_heads=4,
        mlp_ratio=4.,
        init_cfg=None,
    ),
    head=dict(type='MAEPretrainHead4D', history_len=tok_dim, in_channels=128),
    init_cfg=None,
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=0.05),
    clip_grad=dict(max_norm=3.0, norm_type=2),
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=2000)
default_hooks = dict(checkpoint=dict(interval=500, max_keep_ckpts=2),
                     logger=dict(interval=50), runtime_info=None)
randomness = dict(seed=0)
resume = False
work_dir = './work_dirs/mae_sweep_overfit10'
env_cfg = dict(cudnn_benchmark=True,
               mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
               dist_cfg=dict(backend='nccl'))
