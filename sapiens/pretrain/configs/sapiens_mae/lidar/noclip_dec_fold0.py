# GENERADO — igual a f1cv_dec_fold0 pero SIN el recorte del objetivo.
# El clip a ±5 desvios del historico (~±2.5 m) truncaba el 32% de los
# valores del futuro. Los modelos aprendian que la respuesta nunca pasa
# de 5: verificado, predicen >5 en solo el 27% de los casos cuando el
# objetivo real lo supera en el 92%. Subprediccion sistematica.
# GENERADO por run_fase1_cv.sh — decoder (dec) del fold 0.
# val RETENIDA del fold 0: ['7e2f727866c69ea0', '82f90331a1dfe968']
# El MAE de este fold se pre-entrena SOLO en sus 8 escenas de train;
# usarlo en otro fold seria FUGA auto-supervisada.
# clean10_gated.py — MOTF gated, horizonte 3s, datos limpios (waymo_clean, 25 escenas)
_base_ = ['../../_base_/default_runtime.py', '_clean10_base.py']

custom_imports = dict(
    imports=[
        'mmpretrain.datasets.trajectory_dataset',
        'mmpretrain.models.backbones.mae_vit_4d',
        'mmpretrain.models.trajectory_pred.trajectory_model_attn',
    ],
    allow_failed_imports=False
)

history_len   = 5
pred_len      = 30
voxel_res     = 2.0
spatial_range = [-10, 10, -10, 10, -2, 4]
num_voxels    = 300

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='TrajectoryDataset',
        data_root='/home/lcad/lidar_sweep_viewer/waymo_clean',
        sequence_len=history_len + pred_len,
        history_len=history_len,
        pred_len=pred_len,
        voxel_res=voxel_res,
        spatial_range=spatial_range,
        augment=True,
        clip_norm=None,
        norm_scale=10.0,   # escala FIJA: el modo historico calibra con 0.5 s y se aplica a 3 s
        scenes=[
            '2a81f5233075e987',
            '2e41fe6faf5cd2ea',
            '367b072edc9822ea',
            '394e61f27c2a1700',
            '4014ae5bcda2726f',
            '41692b0ec7ff4123',
            '4a2ef30000d19d90',
            '4b60f9400a30ceaf',
        ],
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
    use_gate=True,
    gate_init=0.5,   # rompe el candado: arranca usando la escena a medias
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=1e-4)
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100)

default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=2),
    logger=dict(interval=20),
)

work_dir = './work_dirs/noclip/noclip_dec_fold0'
load_from = './work_dirs/f1cv/mae_encoder_fold0.pth'
