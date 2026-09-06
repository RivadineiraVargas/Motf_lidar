# mae_vit_4d.py — versão corrigida

import torch
import torch.nn as nn
from mmpretrain.registry import MODELS
from mmpretrain.models.selfsup.mae import MAEViT

@MODELS.register_module()
class MAEViT4D(MAEViT):
    def __init__(self,
                 history_len=100,
                 embed_dim=1024,
                 num_tokens=None,
                 **kwargs):
        super().__init__(**kwargs)

        if hasattr(self, 'pos_embed'):
            del self.pos_embed

        self.history_len = history_len
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        # Projeção linear: history_len → embed_dim
        self.patch_embed = nn.Linear(history_len, embed_dim)

        # pos_embed fixo se num_tokens conhecido; senão criado no primeiro forward
        if num_tokens is not None:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_tokens, embed_dim)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

    def init_weights(self):
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

    def _ensure_pos_embed(self, num_tokens: int, device: torch.device):
        """Crea el pos_embed la primera vez, si num_tokens no se conocía en __init__.

        NO REEMPLAZA UNO EXISTENTE. Antes lo hacía —bastaba con que el tamaño no
        coincidiera— y eso tenía dos consecuencias, las dos silenciosas:

          1. El pos_embed ENTRENADO se tiraba y se reemplazaba por ruido
             (trunc_normal). El entrenamiento seguía y daba numeros plausibles.
          2. El Parameter nuevo nacía con requires_grad=True, asi que el freeze de
             `freeze_encoder` —que se aplica UNA vez en __init__— ya no lo
             alcanzaba: el encoder "congelado" pasaba a tener un tensor entrenable.

        Medido: con pos_embed (1,128,1024) congelado y un forward de 660 tokens,
        quedaba (1,660,1024) con requires_grad=True y 1 parámetro entrenable, sin
        lanzar nada. Cualquier caída de ADE se habría atribuido a la
        representación en vez de al pos_embed perdido.

        El desajuste solo puede venir de un config mal armado —el dataset produce
        N tokens y el modelo declara otro—, y ahí lo correcto es frenar.
        """
        if self.pos_embed is None:
            # Registrar como Parameter corretamente — NÃO usar .to() depois
            self.pos_embed = nn.Parameter(
                nn.init.trunc_normal_(
                    torch.zeros(1, num_tokens, self.embed_dim, device=device),
                    std=0.02
                )
            )
        elif self.pos_embed.shape[1] != num_tokens:
            raise ValueError(
                f'pos_embed declarado para {self.pos_embed.shape[1]} tokens pero '
                f'el dataset entregó {num_tokens}. El modelo y el dataset tienen '
                f'que declarar la MISMA tokenización: revisar num_tokens del '
                f'modelo contra az_stride/range_w/patch del dataset (range-view) '
                f'o voxel_res/spatial_range (vóxeles). Reemplazar el pos_embed '
                f'acá destruiría el que vino del checkpoint y lo descongelaría.')

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(
            x, dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward(self, x):
        # x: (B, num_voxels, history_len)
        batch_size, num_tokens, _ = x.shape

        self._ensure_pos_embed(num_tokens, x.device)

        if self.training:
            x_embed = self.patch_embed(x)          # (B, num_tokens, embed_dim)
            x_masked, mask, ids_restore, ids_keep = self.random_masking(
                x_embed, self.mask_ratio
            )
            pos_keep = torch.gather(
                self.pos_embed.expand(batch_size, -1, -1),
                dim=1,
                index=ids_keep.unsqueeze(-1).expand(-1, -1, self.embed_dim)
            )
            x_masked = x_masked + pos_keep
        else:
            x_masked = self.patch_embed(x) + self.pos_embed
            mask = torch.zeros(batch_size, num_tokens, device=x.device)
            ids_restore = torch.arange(num_tokens, device=x.device) \
                               .unsqueeze(0).expand(batch_size, -1)

        for blk in self.layers:
            x_masked = blk(x_masked)

        x_masked = self.norm1(x_masked)

        return x_masked, mask, ids_restore
