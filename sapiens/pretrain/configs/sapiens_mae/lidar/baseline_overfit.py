_base_ = ['../../_base_/default_runtime.py']

custom_imports = dict(
    imports=[
        'mmpretrain.datasets.trajectory_dataset',
        'mmpretrain.models.trajectory_pred.baseline_model'
    ],
    allow_failed_imports=False)

history_len = 5
pred_len = 5

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='TrajectoryDataset',
        data_root='/home/lcad/lidar_sweep_viewer/waymo_10',
        sequence_len=10,
        history_len=history_len,
        pred_len=pred_len,
        voxel_res=2.0,
        spatial_range=[-10,10,-10,10,-2,4]
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
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

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200)
default_hooks = dict(
    checkpoint=dict(interval=50, max_keep_ckpts=2),
    logger=dict(interval=1)
)
work_dir = './work_dirs/baseline_overfit'
