# mae_rangeview_pretrain.py — Pré-treino MAE sobre a RANGE-VIEW (track range-view).
# Cena = imagem de rango (64x512) empilhada em 5 frames, patchificada em 128 tokens
# de dim 1280. O MAEViT4D é reusado: patch_embed = Linear(1280 -> embed_dim).
# Mesmas 8 cenas de treino que o track de vóxels.
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

scene_frames = 5            # frames de range-view empilhados
n_tokens     = 128          # (64/16) * (512/16)
tok_dim      = 16 * 16 * scene_frames   # 1280 = patch_dim (dim de cada token)
embed_dim    = 1024
mask_ratio   = 0.75

train_scenes = [
    '2a81f5233075e987', '2e41fe6faf5cd2ea', '367b072edc9822ea', '394e61f27c2a1700',
    '4014ae5bcda2726f', '41692b0ec7ff4123', '4a2ef30000d19d90', '4b60f9400a30ceaf',
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RangeViewSequenceDataset',
        data_root='/home/lcad/lidar_sweep_viewer/waymo_clean',
        history_len=scene_frames,
        scenes=train_scenes,
        pipeline=[],
    )
)

data_preprocessor = dict(type='BaseDataPreprocessor')

model = dict(
    type='MAE4D',
    backbone=dict(
        type='MAEViT4D',
        history_len=tok_dim,          # = patch_dim: Linear(1280 -> embed_dim)
        embed_dim=embed_dim,
        num_tokens=n_tokens,
        arch='sapiens_0.3b',
        mask_ratio=mask_ratio,
        final_norm=True,
    ),
    neck=dict(
        type='MAEPretrainDecoder',
        num_patches=n_tokens,
        patch_size=1,
        in_chans=tok_dim,             # reconstrói tokens de dim 1280
        embed_dim=embed_dim,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
        init_cfg=None,
    ),
    head=dict(
        type='MAEPretrainHead4D',
        history_len=tok_dim,          # alvo de reconstrução = dim do token
        in_channels=512,
    ),
    init_cfg=None,
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05),
    clip_grad=dict(max_norm=3.0, norm_type=2),
    paramwise_cfg=dict(custom_keys={'bias': dict(decay_mult=0.0),
                                    'norm': dict(decay_mult=0.0)})
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1000)
default_hooks = dict(checkpoint=dict(interval=200, max_keep_ckpts=3),
                     logger=dict(interval=10), runtime_info=None)
randomness = dict(seed=0)
resume = False
work_dir = './work_dirs/mae_rangeview'
env_cfg = dict(cudnn_benchmark=True,
               mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
               dist_cfg=dict(backend='nccl'))
