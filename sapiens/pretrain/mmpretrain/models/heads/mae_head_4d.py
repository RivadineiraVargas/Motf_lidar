# mae_head_4d.py — cabeza de pre-entrenamiento del MAE 4D.
#
# Dos modos:
#   'ocupacion' (histórico) — reconstruye el token crudo con MSE.
#   'centroide' (GeoMAE)    — predice el CENTROIDE de los puntos dentro de cada
#                             vóxel enmascarado, no su ocupación.
#
# El porqué del segundo: reconstruir ocupación no obliga al encoder a codificar
# geometría fina, y nuestros 16 experimentos midieron que las features
# resultantes no sirven para pronóstico de trayectorias. GeoMAE (Tian et al.,
# arXiv:2305.08808, evaluado en Waymo) plantea lo mismo — critica a quienes
# "adoptan MAE directamente y solo predicen coordenadas u ocupación" — y reporta
# +2.7 AP cambiando el objetivo a targets geométricos.
#
# La pérdida toma tres ideas de pointmap_l1_loss.py de Sapiens
# (seg/mmseg/models/losses/), que resuelven problemas que aparecen igual acá:
#   1. máscara de validez: los vóxeles vacíos vienen en NaN y no se penalizan.
#      Sin esto se entrenaría al modelo a predecir el vacío.
#   2. normalización por magnitud media: evita que dominen las zonas de valores
#      grandes.
#   3. L1 en vez de L2: robusta a outliers, habituales en nubes de puntos.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmpretrain.registry import MODELS
from mmengine.model import BaseModule


@MODELS.register_module()
class MAEPretrainHead4D(BaseModule):
    def __init__(self, history_len, in_channels=None, target='ocupacion',
                 normalize=True, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert target in ('ocupacion', 'centroide')
        self.history_len = history_len
        self.target = target
        self.normalize = normalize
        self.mse_loss = nn.MSELoss(reduction='none')

    def loss(self, pred, target, mask, geo=None):
        """pred/target: (B, num_voxels, history_len) · mask: (B, num_voxels), 1=enmascarado
        geo: (B, num_voxels, 3) centroides, con NaN en los vóxeles vacíos."""
        if self.target == 'ocupacion' or geo is None:
            loss = self.mse_loss(pred, target).mean(dim=-1)
            return (loss * mask).sum() / (mask.sum() + 1e-6)

        # --- modo centroide ---
        p = pred[..., :3]                                   # (B, V, 3)
        valido = (~torch.isnan(geo).any(dim=-1)).float()     # vóxeles con puntos
        g = torch.nan_to_num(geo, nan=0.0)
        peso = mask * valido                                 # enmascarado Y con dato

        if self.normalize:
            # magnitud media sobre los vóxeles válidos, por muestra del lote
            def escala(x):
                n = torch.linalg.vector_norm(x, dim=-1) * peso
                return (n.sum(dim=1) / peso.sum(dim=1).clamp(min=1)).view(-1, 1, 1)
            p = p / (escala(p) + 1e-8)
            g = g / (escala(g) + 1e-8)

        loss = F.l1_loss(p, g, reduction='none').mean(dim=-1)
        return (loss * peso).sum() / (peso.sum() + 1e-6)
