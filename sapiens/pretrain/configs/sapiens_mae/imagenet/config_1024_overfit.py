# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

_base_ = [
    '../../_base_/models/mae_vit-base-p16.py',
    '../../_base_/default_runtime.py',
]

# --- CONFIGURAÇÕES GERAIS ---
patch_size = 16
image_size = 1024

# Visualizar e Salvar com frequência para monitorar o teste
vis_every_iters = 1000
save_every_epochs = 200

# Modelo Escolhido
model_name = 'sapiens_0.3b'; embed_dim=1024; num_layers=24 # <--- CORRIGIDO: Garante o uso do modelo 0.3b
# model_name = 'sapiens_0.6b'; embed_dim=1280; num_layers=32
# model_name = 'sapiens_1b'; embed_dim=1536; num_layers=40
# model_name = 'sapiens_2b'; embed_dim=1920; num_layers=48 # <--- Comentado para desativar

num_patches = (image_size // patch_size) ** 2

custom_imports = dict(
    imports=['mmpretrain.datasets', 'mmpretrain.visualization'],
    allow_failed_imports=False
)

# --- PREPROCESSAMENTO ---
data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    # SUBSTITUIR RandomResizedCrop por Resize fixo
    dict(
        type='Resize',
        scale=(image_size, image_size), # Força tamanho exato 1024x1024
        backend='pillow',
        interpolation='bicubic'),
    dict(type='PackInputs')
]

# [ALTERADO] Dataloader para ler pasta local e Batch pequeno
train_dataloader = dict(
    batch_size=10, # <--- Ideal: tamanho do batch igual ao do dataset para overfitting
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False), # Shuffle False ajuda no debug visual
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='CustomDataset', # <--- Usa pasta genérica de imagens
        data_root='/dados/hendrix/customData', # <--- Certifique-se que suas 10 imagens estão aqui
        pipeline=train_pipeline
    )
)

# --- MODELO ---
model = dict(
    backbone=dict(
        type='MAEViT', 
        arch=model_name, 
        patch_size=patch_size, 
        img_size=image_size, 
        final_norm=True, 
        mask_ratio=0.75 # Baixo para facilitar, mas > 0 para evitar erro de cálculo
    ),
    neck=dict(
        type='MAEPretrainDecoder',
        embed_dim=embed_dim,
        patch_size=patch_size,
        num_patches=num_patches),
    head=dict(
        type='MAEPretrainHead',
        patch_size=patch_size,
        norm_pix=False # Desligado para evitar divisão por zero em patches lisos
    ))

# --- OPTIMIZER (ESTÁVEL) ---
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', 
        lr=1e-4, #5e-4,      # 1e-4 é seguro para batch size pequeno
        weight_decay=0.05, 
    ),
    # OBRIGATÓRIO PARA EVITAR NAN
    clip_grad=dict(max_norm=3.0, norm_type=2), 
    
    paramwise_cfg=dict(
        custom_keys={
            'bias': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0)
        }
    )
)

# --- RUNTIME ---
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=4000) # <--- AUMENTADO para garantir a convergência

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=save_every_epochs, max_keep_ckpts=2),
    logger=dict(type='LoggerHook', interval=1), # Log a cada iteração para ver o loss caindo
    visualization=dict(type='VisualizationHook', enable=True),
)

randomness = dict(seed=0, diff_rank_seed=True)
resume = True

# [ALTERADO] Desligar auto-scale para respeitar nosso LR de 1e-4
auto_scale_lr = dict(enable=False)

custom_hooks = [
    dict(
        type='PretrainVisualizationHook',
        enable=True,
        vis_every_iters=vis_every_iters,
        vis_max_samples=4, # Visualiza o batch todo (se for 4)
    )
]

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)


## for dummy testing
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        scale=image_size,
        interpolation='bicubic',
        backend='pillow'),
    dict(type='PackInputs'),
]