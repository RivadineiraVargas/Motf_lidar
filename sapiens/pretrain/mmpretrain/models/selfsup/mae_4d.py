from mmpretrain.registry import MODELS
from .mae import MAE
import torch


@MODELS.register_module()
class MAE4D(MAE):
    """MAE adaptado para tokens LiDAR 4D."""

    def forward(self, inputs, data_samples=None, mode='tensor', geo=None, **kwargs):
        # 'geo' son los centroides por vóxel del modo GeoMAE. Viajan como una
        # clave más del dict del dataset; se guardan acá para que loss() los
        # tome, porque MAE.loss() de la clase base solo pasa (pred, inputs, mask).
        if geo is not None:
            self._geo = geo if not isinstance(geo, list) else torch.stack(geo, 0)
        # Desempacotar lista que o mmengine entrega
        if isinstance(inputs, list):
            inputs = torch.stack(inputs, dim=0)
        elif isinstance(inputs, dict):
            g = inputs.get('geo')
            if g is not None:
                self._geo = g if not isinstance(g, list) else torch.stack(g, 0)
            inputs = inputs['inputs']

        if mode == 'loss':
            # Devuelve SOLO el dict de pérdidas. Antes retornaba la tupla
            # (losses_dict, pred, mask) porque el train_step viejo la esperaba;
            # el refactor de e600dea (07/07) lo cambió a recibir solo el dict.
            return self.loss(inputs, data_samples)
        return super().forward(inputs, data_samples, mode=mode, **kwargs)

    def loss(self, inputs, data_samples=None, **kwargs):
        """Igual que MAE.loss, pero pasando los centroides a la cabeza para el
        objetivo GeoMAE. Con target='ocupacion' la cabeza ignora 'geo' y el
        comportamiento es idéntico al histórico."""
        latent, mask, ids_restore = self.backbone(inputs)
        pred = self.neck(latent, ids_restore)
        geo = getattr(self, '_geo', None)
        if geo is not None:
            geo = geo.to(pred.device).float()
        loss = self.head.loss(pred, inputs, mask, geo=geo)
        self._vis_data = (pred, mask)
        return dict(loss=loss)
