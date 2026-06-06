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
                 scene_dim=64,
                 freeze_encoder=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder = MODELS.build(encoder)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.history_len = history_len
        self.pred_len = pred_len
        self.embed_dim = embed_dim

        self.history_proj = nn.Linear(history_len * 3, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Rama de escena: normalizar + proyectar a poucas dims para que NÃO afogue
        # o histórico (15 dims) na concatenação. Ver diagnóstico waymo_10 (1 cena).
        self.scene_norm = nn.LayerNorm(embed_dim)
        self.scene_proj = nn.Linear(embed_dim, scene_dim)
        # Gate aprendível iniciado em 0 -> tanh(0)=0 -> modelo arranca ignorando a
        # cena (comporta-se como o baseline) e só "abre" a rama se ela ajudar.
        # Garante que nunca pode ser pior que o baseline.
        self.scene_gate = nn.Parameter(torch.zeros(1))

        input_dim = scene_dim + history_len * 3
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

        # 4. Normalizar, projetar a poucas dims e aplicar gate aprendível
        scene_feat = self.scene_proj(self.scene_norm(attn_out))   # (B, scene_dim)
        scene_feat = torch.tanh(self.scene_gate) * scene_feat     # gate -> arranca em 0

        # 5. Concatenar com história original e decodificar
        combined = torch.cat([scene_feat, obj_history_flat], dim=1)  # (B, scene_dim + history_len*3)
        pred_flat = self.decoder(combined)                           # (B, pred_len*3)

        if mode == 'loss':
            if obj_future_flat is None:
                raise ValueError("obj_future_flat obrigatório no modo 'loss'")
            loss = nn.functional.mse_loss(pred_flat, obj_future_flat)
            return dict(loss=loss)
        else:
            return pred_flat