# config_rangeview_rect_overfit100.py — Paso 2 del protocolo 10/100/1000 (Claudine).
# Igual que config_rangeview_rect_overfit10 pero con 100 sweeps MULTI-ESCENA
# (24 escenas, generados por utilities/make_rect_png_100.py; la escena
# 82f90331a1dfe968 queda EXCLUIDA como test no-visto).
# batch 4 (a 10 sweeps batch 2 usaba 2.2GB/8GB) -> 25 iters/epoca.
_base_ = ['../../_base_/default_runtime.py']

img_height = 1024
img_width = 2650
patch_height = 16
patch_width = 25
grid_h = img_height // patch_height   # 64
grid_w = img_width // patch_width     # 106
num_patches = grid_h * grid_w         # 6784
embed_dim = 384                       # encoder REDUCIDO (Claudine)

data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(img_width, img_height), backend='pillow', interpolation='bicubic'),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='CustomDataset',
        data_root='/home/lcad/lidar_sweep_viewer/waymo_clean/range_png_rect/train100',
        with_label=False,
        pipeline=train_pipeline,
    ),
)

model = dict(
    type='MAE',
    backbone=dict(
        type='MAEViT',
        arch=dict(embed_dims=embed_dim, num_layers=6, num_heads=6,
                  feedforward_channels=embed_dim * 4),
        img_size=(img_height, img_width),
        patch_size=(patch_height, patch_width),
        mask_ratio=0.5),
    neck=dict(
        type='MAEPretrainDecoder',
        num_patches=num_patches,
        patch_resolution=(grid_h, grid_w),          # grid rectangular (pos embed)
        patch_size=patch_height,
        in_chans=3,
        embed_dim=embed_dim,
        decoder_embed_dim=192,
        decoder_depth=4,
        decoder_num_heads=6,
        mlp_ratio=4.,
        predict_feature_dim=patch_height * patch_width * 3),  # 1200, patch rectangular
    head=dict(
        type='MAEPretrainHead',
        norm_pix=False,
        patch_height=patch_height,
        patch_width=patch_width,
        img_height=img_height,
        img_width=img_width,
        loss=dict(type='PixelReconstructionLoss', criterion='L2')),
    init_cfg=[
        dict(type='Xavier', layer='Linear', distribution='uniform'),
        dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
    ])

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1.5e-4, betas=(0.9, 0.95), weight_decay=0.05),
    clip_grad=dict(max_norm=1.0))

# 1000 ep: loss 2.07->0.39 sin meseta; val 0.79 / unseen 3.16 ya mejores que el
# modelo de 10 sweeps (0.99 / 3.39) -> extendido a 3000 para seguir bajando.
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=3000)
default_hooks = dict(checkpoint=dict(interval=100, max_keep_ckpts=2),
                     logger=dict(interval=25))
randomness = dict(seed=0)
work_dir = './work_dirs/rv_rect_overfit100'
