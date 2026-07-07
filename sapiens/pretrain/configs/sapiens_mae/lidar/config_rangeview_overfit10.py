# config_rangeview_overfit10.py — Enfoque del COLEGA (MAE de imágenes Sapiens
# estándar sobre range-PNG) + arquitectura REDUCIDA (Claudine Sec. 4) + datos LIMPIOS.
# Overfit en 10 sweeps. Imagen cuadrada 512 para masking super-patch limpio.
_base_ = ['../../_base_/default_runtime.py']

img_size = 512
patch_size = 16          # 512/16 = 32 -> 32x32 = 1024 patches (super-patch 2x2 OK)
embed_dim = 384          # encoder REDUCIDO (vs 1024 del colega)

# Preproc estilo colega: trata el gris como RGB con stats ImageNet
data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(img_size, img_size), backend='pillow', interpolation='bicubic'),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='CustomDataset',
        data_root='/home/lcad/lidar_sweep_viewer/waymo_clean/range_png',
        with_label=False,
        pipeline=train_pipeline,
    ),
)

# test_dataloader: requerido por MAEInferencer para la viz de reconstrucción
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(img_size, img_size), backend='pillow', interpolation='bicubic'),
    dict(type='PackInputs'),
]
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='CustomDataset',
        data_root='/home/lcad/lidar_sweep_viewer/waymo_clean/range_png',
        with_label=False,
        pipeline=test_pipeline,
    ),
)

model = dict(
    type='MAE',
    backbone=dict(
        type='MAEViT',
        arch=dict(embed_dims=embed_dim, num_layers=6, num_heads=6,
                  feedforward_channels=embed_dim * 4),   # REDUCIDA: 6 capas
        img_size=img_size,
        patch_size=patch_size,
        mask_ratio=0.5),                                  # como el colega
    neck=dict(
        type='MAEPretrainDecoder',
        num_patches=(img_size // patch_size) ** 2,        # 1024 (faltaba)
        patch_size=patch_size,
        in_chans=3,
        embed_dim=embed_dim,
        decoder_embed_dim=192,
        decoder_depth=4,
        decoder_num_heads=6,
        mlp_ratio=4.),
    head=dict(
        type='MAEPretrainHead',
        norm_pix=False,                                   # como el colega (evita div/0)
        patch_size=patch_size,
        loss=dict(type='PixelReconstructionLoss', criterion='L2')),
    init_cfg=[
        dict(type='Xavier', layer='Linear', distribution='uniform'),
        dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
    ])

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1.5e-4, betas=(0.9, 0.95), weight_decay=0.05),
    clip_grad=dict(max_norm=1.0))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=2000)
default_hooks = dict(checkpoint=dict(interval=500, max_keep_ckpts=2),
                     logger=dict(interval=20))
randomness = dict(seed=0)
work_dir = './work_dirs/rv_img_overfit10'
