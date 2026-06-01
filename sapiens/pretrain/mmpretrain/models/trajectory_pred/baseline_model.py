import torch
import torch.nn as nn
from mmpretrain.registry import MODELS
from mmengine.model import BaseModel

@MODELS.register_module()
class BaselineTrajectoryModel(BaseModel):
    def __init__(self, history_len=5, pred_len=5, hidden_dim=512, **kwargs):
        super().__init__(**kwargs)
        self.history_len = history_len
        self.pred_len = pred_len
        input_dim = history_len * 3
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_len * 3)
        )

    def forward(self, obj_history_flat, obj_future_flat=None, mode='loss', **kwargs):
        pred_flat = self.decoder(obj_history_flat)
        if mode == 'loss':
            if obj_future_flat is None:
                raise ValueError("For loss mode, obj_future_flat must be provided")
            loss = nn.functional.mse_loss(pred_flat, obj_future_flat)
            return dict(loss=loss)
        elif mode == 'predict':
            return pred_flat
        else:
            return pred_flat
