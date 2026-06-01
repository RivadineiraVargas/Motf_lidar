import sys
sys.path.insert(0, '/home/lcad/lidar_sweep_viewer/sapiens/pretrain')

import torch
from mmpretrain.models.backbones.mae_vit_4d import MAEViT4D

model = MAEViT4D(history_len=5, embed_dim=1024, num_tokens=300)
dummy = torch.randn(1, 300, 5)
out = model(dummy)
print("Backbone funcionando, salida shape:", out[0].shape)
print("Todo OK")
