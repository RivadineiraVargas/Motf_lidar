# baseline_multiescena.py
# Baseline MLP (só histórico, sem cena) nas mesmas 8 cenas de treino.
# Comparação contra trajectory_attn_multiescena para isolar o aporte da cena.
_base_ = ['../../_base_/default_runtime.py']

custom_imports = dict(
    imports=[
        'mmpretrain.datasets.trajectory_dataset',
        'mmpretrain.models.trajectory_pred.baseline_model'
    ],
    allow_failed_imports=False)

history_len = 5
pred_len = 5

train_scenes = [
    '1a01176ef4830499', '1a1918cd002b6a94', '1a64c26c56412fc5',
    '1b20914ea0bc28bd', '1b88ea97e03be2bf', '1ba702a85f4fa15c',
    '1bb5aa0c95f2ce0e', '58d5f1b9e6a1a2f7',
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='TrajectoryDataset',
        data_root='/home/lcad/lidar_sweep_viewer/waymo_10',
        sequence_len=10,
        history_len=history_len,
        pred_len=pred_len,
        voxel_res=2.0,
        spatial_range=[-10, 10, -10, 10, -2, 4],
        scenes=train_scenes,
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
    optimizer=dict(type='AdamW', lr=1e-3)
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=300)
default_hooks = dict(
    checkpoint=dict(interval=100, max_keep_ckpts=2),
    logger=dict(interval=10),
)
work_dir = './work_dirs/baseline_multiescena'
