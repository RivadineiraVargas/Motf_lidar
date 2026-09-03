# rect_overfit10_val.py — la prueba de 10 sweeps de Claudine (ítem 5), INSTRUMENTADA.
#
# QUE CAMBIA respecto de config_rangeview_rect_overfit10.py, y por que. Los tres
# cambios salen de medir el estado del 02/09/2026, no de suponer nada:
#
# 1. data_root APUNTA A UNA CARPETA CON LAS 10 IMAGENES, no a la raiz.
#    El original apunta a `range_png_rect/` y CustomDataset recorre
#    subdirectorios. El 28/06 ahi solo estaba `train/`, asi que la corrida
#    original vio 10 imagenes (su log dice [6000][5/5] con batch_size=2, o sea
#    10). Despues aparecieron train100, val, unseen y cinco fold*_train:
#    construyendo el dataset HOY con ese config salen 612 imagenes, e incluyen
#    `val/` y `unseen/` — fuga directa en el split contra el que se mide.
#
# 2. LAS 10 IMAGENES AHORA ESTAN A 1024 FILAS, como todo lo demas.
#    `train/` estaba en 2650x64 (filas nativas) mientras val/unseen/train100 y
#    los fold*_train estan en 2650x1024. El pipeline reescala a 1024 con
#    BICUBIC; el generador (make_rect_png_scenes.py) escribe las de 1024 con
#    INTER_NEAREST. O sea que el modelo entrenaba sobre gradientes suavizados y
#    se evaluaba sobre bloques duros de 16 filas repetidas: dominios distintos.
#    Es la explicacion mas simple del item 11 (10sw 3.39 contra 3.52 sin
#    entrenar, casi nada) y de por que el de 100 sweeps va mejor (3.16): su
#    train100 ya estaba en 1024, entrenaba y evaluaba en el mismo dominio.
#
# 3. SE GUARDAN TODOS LOS CHECKPOINTS.
#    El original tiene max_keep_ckpts=2 con interval=500: en disco quedaron solo
#    epoch_5500 y epoch_6000. El item 6 (100 sweeps) encontro que "la
#    generalizacion pica ~ep1000 y luego memoriza", asi que el 3.39 medido en la
#    epoca 6000 es, muy probablemente, el modelo ya memorizado — y no se podia
#    verificar porque los checkpoints del pico estaban borrados.
#
# LAS 10 IMAGENES SON LAS MISMAS de la prueba original: escena 2a81f5233075e987,
# sweeps {0,1,2,3,4,5,6,7,8,10}. Identificadas comparando pixel a pixel contra
# los .npy de range_files (diferencia media 0.20 sobre 255). Falta el sweep 9
# porque el generador ordeno los .npy como STRINGS: 0, 1, 10, 2, 3... El item 5
# decia "10 sweeps de la escena 2a81" y eso es correcto, pero no son los
# sweeps 0-9 sino 0-8 mas el 10.
#
# EL RETENIDO, que antes era n=1. Se mide con curva_overfit10.py sobre:
#   ov10_val_intra   — 2a81 sweep 9: el UNICO sweep de la escena de train que el
#                      modelo no vio. Mide memorizacion de la escena.
#   ov10_val_escenas — 5 escenas nunca vistas x 11 sweeps = 55 imagenes. Mide lo
#                      que de verdad importa: si cruza entre ESCENAS.
# Todo lo demas —arquitectura reducida 384/6 (item 2 de Claudine), mask_ratio
# 0.5, lr 1.5e-4, 6000 epocas, seed 0— es identico al original.
_base_ = ['../../_base_/default_runtime.py']

img_height = 1024
img_width = 2650
patch_height = 16
patch_width = 25
grid_h = img_height // patch_height   # 64
grid_w = img_width // patch_width     # 106
num_patches = grid_h * grid_w         # 6784
embed_dim = 384                       # encoder REDUCIDO (Claudine, item 2)

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
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='CustomDataset',
        # carpeta con EXACTAMENTE las 10 imagenes, ya a 1024 (ver nota 1 y 2)
        data_root='/home/lcad/lidar_sweep_viewer/waymo_clean/range_png_rect/ov10_train',
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

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=6000)
# interval 250 y SIN max_keep_ckpts: la curva de generalizacion es el producto
# de esta corrida, y con max_keep_ckpts=2 se perdia justo la zona del pico.
default_hooks = dict(checkpoint=dict(interval=250),
                     logger=dict(interval=20))
randomness = dict(seed=0)
work_dir = './work_dirs/rect_ov10_val'
