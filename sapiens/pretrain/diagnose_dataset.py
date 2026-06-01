import numpy as np
from mmpretrain.datasets import TrajectoryDataset

ds = TrajectoryDataset(
    data_root='/home/lcad/lidar_sweep_viewer/waymo_10',
    sequence_len=10,
    history_len=5,
    pred_len=5,
    voxel_res=2.0,
    spatial_range=[-10,10,-10,10,-2,4]
)

print(f"Total objetos: {len(ds)}")

all_history = []
all_future = []

for i in range(len(ds)):
    data = ds[i]
    hist = data['obj_history_flat'].numpy()
    fut = data['obj_future_flat'].numpy()
    all_history.append(hist)
    all_future.append(fut)
    print(f"Objeto {i}: historia min={hist.min():.2f}, max={hist.max():.2f}, media={hist.mean():.2f}, std={hist.std():.2f}")
    print(f"        futuro min={fut.min():.2f}, max={fut.max():.2f}, media={fut.mean():.2f}, std={fut.std():.2f}")

all_history = np.concatenate(all_history)
all_future = np.concatenate(all_future)
print("\n--- Estadísticas globales ---")
print(f"Historia: min={all_history.min():.2f}, max={all_history.max():.2f}, media={all_history.mean():.2f}, std={all_history.std():.2f}")
print(f"Futuro:   min={all_future.min():.2f}, max={all_future.max():.2f}, media={all_future.mean():.2f}, std={all_future.std():.2f}")
