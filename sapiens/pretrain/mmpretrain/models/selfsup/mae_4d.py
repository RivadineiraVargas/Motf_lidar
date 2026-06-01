from mmpretrain.registry import MODELS
from .mae import MAE
import torch


@MODELS.register_module()
class MAE4D(MAE):
    """MAE adaptado para tokens LiDAR 4D."""

    def forward(self, inputs, data_samples=None, mode='tensor', **kwargs):
        # Desempacotar lista que o mmengine entrega
        if isinstance(inputs, list):
            inputs = torch.stack(inputs, dim=0)
        elif isinstance(inputs, dict):
            inputs = inputs['inputs']

        if mode == 'loss':
            losses_dict = super().loss(inputs, data_samples)
            # mae.py train_step espera tupla (losses_dict, preds, masks)
            vis = getattr(self, '_vis_data', (None, None))
            return losses_dict, vis[0], vis[1]
        else:
            return super().forward(inputs, data_samples, mode=mode, **kwargs)
