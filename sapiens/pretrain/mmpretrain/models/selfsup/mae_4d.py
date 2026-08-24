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
            # Devuelve SOLO el dict de pérdidas. Antes retornaba la tupla
            # (losses_dict, pred, mask) porque el train_step viejo la esperaba;
            # el refactor de e600dea (07/07) lo cambió a recibir solo el dict y
            # tomar la visualización de self._vis_data, que MAE.loss() ya deja
            # puesta. Esta clase quedó con el contrato viejo y rompía con
            # "'tuple' object has no attribute 'items'" en parse_losses.
            return super().loss(inputs, data_samples)
        else:
            return super().forward(inputs, data_samples, mode=mode, **kwargs)
