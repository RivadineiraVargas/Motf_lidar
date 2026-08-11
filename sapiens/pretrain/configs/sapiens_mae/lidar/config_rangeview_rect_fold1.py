# config_rangeview_rect_fold1.py — Re-pre-entrenamiento del encoder MAE SOLO
# en las 20 escenas de TRAIN del fold 1 (sin fuga de las 5 retenidas).
# Parte de completar la validacion cruzada de 5 folds: el efecto medido a 3s
# resulto depender del split (fold 0: -20.4%, p=0.0006; fold 4: nulo), asi que
# hacen falta los folds 1-3 para una respuesta promediada sobre los 5.
# Datos: 100 PNG (20 escenas x 5
# sweeps) generados por utilities/make_rect_png_scenes.py.
# Idéntico en arquitectura al overfit100 validado; solo cambian data_root,
# work_dir, max_epochs y max_keep_ckpts (protege más checkpoints).
_base_ = ['../../_base_/default_runtime.py']

img_height = 1024
img_width = 2650
patch_height = 16
patch_width = 25
grid_h = img_height // patch_height   # 64
grid_w = img_width // patch_width     # 106
num_patches = grid_h * grid_w         # 6784
embed_dim = 384

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
        data_root='/home/lcad/lidar_sweep_viewer/waymo_clean/range_png_rect/fold1_train',
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
        patch_resolution=(grid_h, grid_w),
        patch_size=patch_height,
        in_chans=3,
        embed_dim=embed_dim,
        decoder_embed_dim=192,
        decoder_depth=4,
        decoder_num_heads=6,
        mlp_ratio=4.,
        predict_feature_dim=patch_height * patch_width * 3),
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

# 1000 ép (~45s/ép = ~12.5h, cabe de noche). El overfit100 mostró que la
# generalización pica ~ép1000. Checkpoints c/250 protegen ante interrupción.
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1000)
default_hooks = dict(checkpoint=dict(interval=250, max_keep_ckpts=6),
                     logger=dict(interval=25))
randomness = dict(seed=0)
work_dir = './work_dirs/rv_rect_fold1'
