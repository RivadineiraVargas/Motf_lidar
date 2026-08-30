# GENERADO por run_noclip_cv.sh — base del fold 4, protocolo VIGENTE.
# Igual a f1cv_base_fold4 pero SIN el recorte del objetivo (clip_norm=None) y con
# escala fija de 10 m, que es lo que usan los experimentos 16-18. Se genera para
# completar la CV de 5 folds: hasta el 30/08 TODOS los resultados de Fase 1 eran
# del fold 0 solo, y este proyecto ya vio dos efectos de un fold evaporarse al
# promediar los cinco (18/07 y 06/08).
# GENERADO por run_fase1_cv.sh — decoder (base) del fold 4.
# val RETENIDA del fold 4: ['394e61f27c2a1700', '4b60f9400a30ceaf']
# El MAE de este fold se pre-entrena SOLO en sus 8 escenas de train;
# usarlo en otro fold seria FUGA auto-supervisada.
# clean10_baseline.py — Baseline (solo histórico), horizonte 3s, waymo_clean 25 escenas
_base_ = ['../../_base_/default_runtime.py', '_clean10_base.py']

custom_imports = dict(
    imports=[
        'mmpretrain.datasets.trajectory_dataset',
        'mmpretrain.models.trajectory_pred.baseline_model',
    ],
    allow_failed_imports=False
)

history_len = 5
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
            '2e41fe6faf5cd2ea',
            '367b072edc9822ea',
            '4014ae5bcda2726f',
            '41692b0ec7ff4123',
            '4a2ef30000d19d90',
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

work_dir = './work_dirs/noclipcv/noclip_base_fold4'
