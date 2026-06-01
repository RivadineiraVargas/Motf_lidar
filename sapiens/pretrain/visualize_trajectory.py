import torch
import matplotlib.pyplot as plt
from mmpretrain.datasets import TrajectoryDataset
from mmpretrain.models.trajectory_pred import TrajectoryPredictionModel
from mmengine.config import Config
from mmengine.runner import load_checkpoint

cfg = Config.fromfile('configs/sapiens_mae/lidar/trajectory_pred_overfit.py')
model_args = {k: v for k, v in cfg.model.items() if k != 'type'}
model = TrajectoryPredictionModel(**model_args)
load_checkpoint(model, 'work_dirs/trajectory_overfit_v2/epoch_200.pth', map_location='cpu')
model.eval()

dataset = TrajectoryDataset(
    data_root='/home/lcad/lidar_sweep_viewer/waymo_10',
    sequence_len=10,
    history_len=5,
    pred_len=5,
    voxel_res=2.0,
    spatial_range=[-10,10,-10,10,-2,4]
)

data = dataset[0]
inputs = data['inputs'].unsqueeze(0)
obj_history_flat = data['obj_history_flat'].unsqueeze(0)
obj_future_flat = data['obj_future_flat'].unsqueeze(0)

with torch.no_grad():
    pred_flat = model(inputs, obj_history_flat)
    pred = pred_flat.view(-1, 5, 3).squeeze(0)
    target = obj_future_flat.view(-1, 5, 3).squeeze(0)

print("Target(real):", target.numpy())
print("Prediction:", pred.numpy())

plt.figure()
plt.plot(target[:,0].numpy(), target[:,1].numpy(), 'go-', label='Real')
plt.plot(pred[:,0].numpy(), pred[:,1].numpy(), 'ro-', label='Predicho')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Trayectoria predicha vs real (overfit)')
plt.savefig('trajectory_overfit.png')
print("Gráfico guardado como trajectory_overfit.png")
