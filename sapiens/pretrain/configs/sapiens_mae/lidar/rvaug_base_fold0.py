# GENERADO — identico a rvcv_base_fold0 pero con augment=True.
# Motivo: la comparacion voxel vs range-view del fold 0 quedo confundida
# por la augmentacion (voxeles la tenia, range-view no): baselines 1.663
# vs 2.222 pese a evaluar los MISMOS 51 objetos. Con esto la unica
# diferencia entre pipelines vuelve a ser la representacion de la escena.
# GENERADO — baseline (sin escena) para RANGE-VIEW, fold 0.
# Usa RangeViewTrajectoryDataset, el MISMO dataset que clean10_rv_gated_init,
# para que histórico, futuro y filtrado de tracks sean idénticos y la
# comparación pareada por semilla contra gate0/gated sea válida.
# val retenida del fold 0: ['7e2f727866c69ea0', '82f90331a1dfe968']
# clean10_baseline.py — Baseline (solo histórico), horizonte 3s, waymo_clean 25 escenas
_base_ = ['../../_base_/default_runtime.py', '_clean10_base.py']

custom_imports = dict(
    imports=[
        'mmpretrain.datasets.range_view',
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
        type='RangeViewTrajectoryDataset',
        data_root='/home/lcad/lidar_sweep_viewer/waymo_clean',
        sequence_len=history_len + pred_len,
        history_len=history_len,
        pred_len=pred_len,
        voxel_res=2.0,                       # (ignorado pela range-view; req. p/ __init__)
        spatial_range=[-10, 10, -10, 10, -2, 4],
        augment=True,   # IGUALA el lado de voxeles (clean10_*: augment=True)
        scenes=[
            '2a81f5233075e987', '2e41fe6faf5cd2ea', '367b072edc9822ea', '394e61f27c2a1700',
            '4014ae5bcda2726f', '41692b0ec7ff4123', '4a2ef30000d19d90', '4b60f9400a30ceaf',
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
    checkpoint=dict(interval=50, max_keep_ckpts=2),
    logger=dict(interval=20),
)

work_dir = './work_dirs/rvaug/rvaug_base_fold0'
