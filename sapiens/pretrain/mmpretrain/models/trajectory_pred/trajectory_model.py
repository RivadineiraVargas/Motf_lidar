import torch
import torch.nn as nn
from mmpretrain.registry import MODELS
from mmengine.model import BaseModule

@MODELS.register_module()
class TrajectoryPredictionModel(BaseModule):
    def __init__(self,
                 encoder,
                 history_len=5,
                 pred_len=5,
                 in_channels=768,
                 hidden_dim=256,
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder = MODELS.build(encoder)
        self.history_len = history_len
        self.pred_len = pred_len
        self.in_channels = in_channels

        # Decoder: MLP que toma la representación global de la escena + historia del objeto
        input_dim = in_channels + history_len * 3
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_len * 3)
        )

    def forward(self, inputs, obj_history_flat, **kwargs):
        # inputs: (batch, num_voxels, history_len)
        # obj_history_flat: (batch, history_len*3)
        latent, _, _ = self.encoder(inputs)          # (batch, num_tokens_keep, in_channels)
        global_feat = latent.mean(dim=1)             # (batch, in_channels)
        combined = torch.cat([global_feat, obj_history_flat], dim=1)  # (batch, in_channels + history_len*3)
        pred_flat = self.decoder(combined)           # (batch, pred_len*3)
        pred = pred_flat.view(-1, self.pred_len, 3)  # (batch, pred_len, 3)
        return pred

    def loss(self, inputs, obj_history_flat, obj_future_flat, **kwargs):
        pred = self.forward(inputs, obj_history_flat)
        obj_future = obj_future_flat.view(-1, self.pred_len, 3)
        loss = nn.functional.mse_loss(pred, obj_future)
        return dict(loss=loss)

    def train_step(self, data, optim_wrapper):
        """Implements the standard training step.

        Args:
            data (dict): Data sampled from the dataset, containing keys:
                'inputs', 'obj_history_flat', 'obj_future_flat'.
            optim_wrapper (OptimWrapper): Wrapper that handles optimizer and
                gradient clipping.

        Returns:
            dict: Logs containing the loss.
        """
        # Forward and compute loss
        losses = self.loss(**data)
        loss = losses['loss']

        # Backprop and update parameters
        optim_wrapper.update_params(loss)

        # Return logs
        return {'loss': loss.item()}
