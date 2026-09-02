# hist11_base_fold2.py — el baseline cinemático con la historia COMPLETA.
#
# Copia de noclip_base_fold2.py con history_len 5 -> 11. Es el único
# cambio: mismo modelo, mismo pred_len, mismas escenas, mismo protocolo.
#
# POR QUE. WOMD-LiDAR entrega 11 frames de LiDAR por escena = 1,1 s, que es
# exactamente la ventana de historia que define el benchmark de Waymo (ver
# docs/ESTUDIO_WAYFORMER.md). Los configs de Fase 1 usan history_len=5: se
# predicen 3 s de futuro con 0,5 s de pasado, tirando mas de la mitad del
# contexto que ya esta en el disco.
#
# COSTO MEDIDO, no estimado: pasar de 5 a 11 cuesta el 13% de las ventanas
# de entrenamiento (236 -> 206 en el fold 0), porque el train ya toma una
# sola ventana por objeto. No es el 7x que uno supondria.
#
# COMO SE EVALUA. Con --poblacion-hist 11 en los DOS brazos, para que
# comparen el mismo futuro sobre los mismos objetos: un modelo de h=5 usa
# su ventana f0=6 y este usa f0=0, y ambos predicen los frames 11..40.
# Sin eso la comparacion es entre poblaciones distintas (183 ventanas de 29
# objetos contra 24 de 24) y no significa nada.
#
# El encoder MAE NO se toca: BaselineTrajectoryModel no lo usa. Para las
# variantes con escena habria que re-pre-entrenarlo, porque su patch_embed
# es Linear(history_len, 1024).
# Igual a f1cv_base_fold2 pero SIN el recorte del objetivo (clip_norm=None) y con
# escala fija de 10 m, que es lo que usan los experimentos 16-18. Se genera para
# completar la CV de 5 folds: hasta el 30/08 TODOS los resultados de Fase 1 eran
# del fold 0 solo, y este proyecto ya vio dos efectos de un fold evaporarse al
# promediar los cinco (18/07 y 06/08).
# val RETENIDA del fold 2: ['2e41fe6faf5cd2ea', '41692b0ec7ff4123']
_base_ = ['../../_base_/default_runtime.py', '_clean10_base.py']

custom_imports = dict(
    imports=[
        'mmpretrain.datasets.trajectory_dataset',
        'mmpretrain.models.trajectory_pred.baseline_model',
    ],
    allow_failed_imports=False
)

history_len = 11
pred_len    = 30

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
        voxel_res=2.0,
        spatial_range=[-10, 10, -10, 10, -2, 4],
        augment=True,
        clip_norm=None,
        norm_scale=10.0,   # escala FIJA: el modo historico calibra con 0.5 s y se aplica a 3 s
        scenes=[
            '2a81f5233075e987',
            '367b072edc9822ea',
            '394e61f27c2a1700',
            '4014ae5bcda2726f',
            '4a2ef30000d19d90',
            '4b60f9400a30ceaf',
            '7e2f727866c69ea0',
            '82f90331a1dfe968',
        ],
    ),
)

model = dict(
    type='BaselineTrajectoryModel',
    history_len=history_len,
    pred_len=pred_len,
    hidden_dim=512,
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

work_dir   = './work_dirs/hist11/base11_fold2'
