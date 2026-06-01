# mae_head_4d.py — versão final simplificada
import torch.nn as nn
from mmpretrain.registry import MODELS
from mmengine.model import BaseModule


@MODELS.register_module()
class MAEPretrainHead4D(BaseModule):
    def __init__(self, history_len, in_channels=None, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.history_len = history_len
        # in_channels mantido por compatibilidade mas não usado
        # o decoder já projeta para history_len via in_chans=history_len
        self.mse_loss = nn.MSELoss(reduction='none')

    def loss(self, pred, target, mask):
        """
        pred:   (B, num_voxels, history_len) — saída do decoder
        target: (B, num_voxels, history_len) — tokens originais
        mask:   (B, num_voxels)              — 1 = mascarado
        """
        loss = self.mse_loss(pred, target)   # (B, num_voxels, history_len)
        loss = loss.mean(dim=-1)             # (B, num_voxels)
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        return loss