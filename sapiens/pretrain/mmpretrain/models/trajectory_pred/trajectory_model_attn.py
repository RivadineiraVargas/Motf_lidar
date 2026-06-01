import torch
import torch.nn as nn
from mmpretrain.registry import MODELS
from mmengine.model import BaseModel


@MODELS.register_module()
class TrajectoryModelWithAttention(BaseModel):
    def __init__(self,
                 encoder,
                 history_len=5,
                 pred_len=5,
                 embed_dim=1024,
                 num_heads=8,
                 hidden_dim=512,
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder = MODELS.build(encoder)
        self.history_len = history_len
        self.pred_len = pred_len
        self.embed_dim = embed_dim

        self.history_proj = nn.Linear(history_len * 3, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        input_dim = embed_dim + history_len * 3
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_len * 3)
        )

    def _encode_scene(self, inputs):
        """
        Extrai tokens da cena SEM mascaramento.
        Independente do modo training/eval do encoder.
        inputs: (B, num_voxels, history_len)
        retorna: (B, num_voxels, embed_dim)
        """
        # Garante pos_embed correto para o tamanho atual
        self.encoder._ensure_pos_embed(inputs.size(1), inputs.device)

        x = self.encoder.patch_embed(inputs)
        x = x + self.encoder.pos_embed[:, :x.size(1), :]

        for blk in self.encoder.layers:
            x = blk(x)

        return self.encoder.norm1(x)   # (B, num_voxels, embed_dim)

    def forward(self, inputs, obj_history_flat, mode='loss',
                obj_future_flat=None, **kwargs):
        # inputs:          (B, num_voxels, history_len)
        # obj_history_flat: (B, history_len * 3)

        # 1. Cena completa via encoder (sem mascaramento)
        latent = self._encode_scene(inputs)               # (B, num_voxels, embed_dim)

        # 2. Projetar história do objeto → query
        query = self.history_proj(obj_history_flat).unsqueeze(1)  # (B, 1, embed_dim)

        # 3. Cross-attention: objeto atende à cena completa
        attn_out, _ = self.cross_attn(query, latent, latent)      # (B, 1, embed_dim)
        attn_out = attn_out.squeeze(1)                             # (B, embed_dim)

        # 4. Concatenar com história original e decodificar
        combined = torch.cat([attn_out, obj_history_flat], dim=1) # (B, embed_dim + history_len*3)
        pred_flat = self.decoder(combined)                         # (B, pred_len*3)

        if mode == 'loss':
            if obj_future_flat is None:
                raise ValueError("obj_future_flat obrigatório no modo 'loss'")
            loss = nn.functional.mse_loss(pred_flat, obj_future_flat)
            return dict(loss=loss)
        else:
            return pred_flat