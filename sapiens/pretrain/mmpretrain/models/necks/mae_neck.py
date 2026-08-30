# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS
from ..backbones.vision_transformer import TransformerEncoderLayer
from ..utils import build_2d_sincos_position_embedding


@MODELS.register_module()
class MAEPretrainDecoder(BaseModule):
    """Decoder for MAE Pre-training.

    Some of the code is borrowed from `https://github.com/facebookresearch/mae`. # noqa

    Args:
        num_patches (int): The number of total patches. Defaults to 196.
        patch_size (int): Image patch size. Defaults to 16.
        in_chans (int): The channel of input image. Defaults to 3.
        embed_dim (int): Encoder's embedding dimension. Defaults to 1024.
        decoder_embed_dim (int): Decoder's embedding dimension.
            Defaults to 512.
        decoder_depth (int): The depth of decoder. Defaults to 8.
        decoder_num_heads (int): Number of attention heads of decoder.
            Defaults to 16.
        mlp_ratio (int): Ratio of mlp hidden dim to decoder's embedding dim.
            Defaults to 4.
        norm_cfg (dict): Normalization layer. Defaults to LayerNorm.
        init_cfg (Union[List[dict], dict], optional): Initialization config
            dict. Defaults to None.

    Example:
        >>> from mmpretrain.models import MAEPretrainDecoder
        >>> import torch
        >>> self = MAEPretrainDecoder()
        >>> self.eval()
        >>> inputs = torch.rand(1, 50, 1024)
        >>> ids_restore = torch.arange(0, 196).unsqueeze(0)
        >>> level_outputs = self.forward(inputs, ids_restore)
        >>> print(tuple(level_outputs.shape))
        (1, 196, 768)
    """

    def __init__(self,
                 num_patches: int = 196,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 embed_dim: int = 1024,
                 decoder_embed_dim: int = 512,
                 decoder_depth: int = 8,
                 decoder_num_heads: int = 16,
                 mlp_ratio: int = 4,
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 predict_feature_dim: Optional[float] = None,
                 patch_resolution: Optional[Sequence[int]] = None,
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_patches = num_patches
        # Grid real de patches (h, w) para pos embed rectangular. Si no se da,
        # se asume cuadrado (sqrt) -> retrocompatible con configs previas.
        self.patch_resolution = patch_resolution

        # used to convert the dim of features from encoder to the dim
        # compatible with that of decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim)) ## learnable mask token. initialized as normal_

        # create new position embedding, different from that in encoder
        # and is not learnable
        # requires_grad SE DECIDE ACÁ, no en init_weights(). mmengine arma el
        # optimizador (Runner.train -> build_optim_wrapper) ANTES de llamar a
        # _init_model_weights(), y DefaultOptimWrapperConstructor.add_params
        # descarta con `continue` todo parámetro con requires_grad=False. Poner
        # requires_grad_(True) dentro de init_weights es un NO-OP: el tensor nunca
        # entra al optimizador. (Verificado el 30/08 tras que la revisión lo
        # cazara: mi primer arreglo del hallazgo 2 tenía justamente ese defecto.)
        #
        # Si el sincos 2D va a aplicar, el embedding es FIJO (requires_grad=False,
        # como en el MAE original). Si la grilla no es 2D —los 300 tokens de
        # vóxeles no forman un cuadrado— no hay sincos que poner, así que tiene
        # que ser APRENDIBLE; si no, se queda en ceros y todo token enmascarado
        # entra al decoder como `mask_token + 0`, indistinguible de los demás.
        grid_ini = patch_resolution if patch_resolution is not None \
            else int(self.num_patches**.5)
        if isinstance(grid_ini, int):
            n_sincos = grid_ini * grid_ini + 1
        else:
            n_sincos = grid_ini[0] * grid_ini[1] + 1
        self._pos_embed_aprendible = (n_sincos != self.num_patches + 1)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
            requires_grad=self._pos_embed_aprendible)
        
        self.decoder_blocks = nn.ModuleList([
            TransformerEncoderLayer(
                decoder_embed_dim,
                decoder_num_heads,
                int(mlp_ratio * decoder_embed_dim),
                qkv_bias=True,
                norm_cfg=norm_cfg) for _ in range(decoder_depth)
        ])

        self.decoder_norm_name, decoder_norm = build_norm_layer(
            norm_cfg, decoder_embed_dim, postfix=1)
        self.add_module(self.decoder_norm_name, decoder_norm)

        # Used to map features to pixels
        if predict_feature_dim is None:
            predict_feature_dim = patch_size**2 * in_chans
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, predict_feature_dim, bias=True)

    def init_weights(self) -> None:
        """Initialize position embedding and mask token of MAE decoder."""
        super().init_weights()

        grid = self.patch_resolution if self.patch_resolution is not None \
            else int(self.num_patches**.5)
        decoder_pos_embed = build_2d_sincos_position_embedding(
            grid,
            self.decoder_pos_embed.shape[-1],
            cls_token=True)
        # Solo si la grilla es REALMENTE 2D y cuadra con el tensor. El MAE 4D de
        # vóxeles (Fase 1) tiene 300 tokens que no forman un cuadrado: el sincos
        # 2D da 17*17+1=290 contra 301 y rompía. Esta inicialización se agregó en
        # e600dea (07/07) para el MAE rectangular de range-view; el encoder 4D se
        # entrenó el 15/06, ANTES, o sea sin ella. Saltearla acá restaura ese
        # comportamiento — necesario para que los encoders por fold de la CV sean
        # comparables con el original — y conserva el fix donde sí aplica.
        if decoder_pos_embed.shape == self.decoder_pos_embed.shape:
            self.decoder_pos_embed.data.copy_(decoder_pos_embed.float())
        else:
            # ARREGLO 30/08 (hallazgo 2 de la revisión). El Parameter se declara
            # arriba con requires_grad=False, que es lo correcto CUANDO se llena
            # con sincos fijo. Al caer acá no se llenaba con nada: se quedaba en
            # torch.zeros y NO era aprendible, pese a que este mismo log decía que
            # sí. Todo token enmascarado entraba al decoder como `mask_token + 0`,
            # indistinguible de cualquier otro, así que el decoder no podía saber
            # qué vóxel estaba reconstruyendo. Es la misma familia del bug
            # decoder_pos_embed documentado del MAE del colega.
            # Ahora se vuelve aprendible de verdad, con la inicialización estándar
            # de MAE para embeddings de posición aprendidos (normal truncada 0.02).
            # requires_grad ya quedó en True desde __init__ (ver allí el porqué).
            assert self.decoder_pos_embed.requires_grad, \
                'el pos-embed del decoder debería ser aprendible en esta rama'
            torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=.02)
            from mmengine.logging import MMLogger
            MMLogger.get_current_instance().info(
                f'[mae_neck] pos-embed 2D {tuple(decoder_pos_embed.shape)} no '
                f'coincide con {tuple(self.decoder_pos_embed.shape)} (grilla no '
                f'cuadrada, p.ej. vóxeles 4D): APRENDIBLE, init trunc_normal(.02). '
                f'OJO: los encoders anteriores al 30/08 se entrenaron con este '
                f'tensor en ceros y NO son comparables.')

        torch.nn.init.normal_(self.mask_token, std=.02)

    @property
    def decoder_norm(self):
        """The normalization layer of decoder."""
        return getattr(self, self.decoder_norm_name)

    def forward(self, x: torch.Tensor,
                ids_restore: torch.Tensor) -> torch.Tensor:
        """The forward function.

        The process computes the visible patches' features vectors and the mask
        tokens to output feature vectors, which will be used for
        reconstruction.

        Args:
            x (torch.Tensor): hidden features, which is of shape
                    B x (L * mask_ratio) x C.
            ids_restore (torch.Tensor): ids to restore original image.

        Returns:
            torch.Tensor: The reconstructed feature vectors, which is of
            shape B x (num_patches) x C.
        """
        # embed tokens

        x = self.decoder_embed(x) ## B x (L * (1-mask_ratio) + 1) x C -> B x (L * (1-mask_ratio) + 1) x Decoder_dim

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1) ## B x (L * mask_ratio) x Decoder_dim
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1) ## B x L x Decoder_dim
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1) ## B x (L + 1) x Decoder_dim

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x


@MODELS.register_module()
class ClsBatchNormNeck(BaseModule):
    """Normalize cls token across batch before head.

    This module is proposed by MAE, when running linear probing.

    Args:
        input_features (int): The dimension of features.
        affine (bool): a boolean value that when set to ``True``, this module
            has learnable affine parameters. Defaults to False.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-6.
        init_cfg (Dict or List[Dict], optional): Config dict for weight
            initialization. Defaults to None.
    """

    def __init__(self,
                 input_features: int,
                 affine: bool = False,
                 eps: float = 1e-6,
                 init_cfg: Optional[Union[dict, List[dict]]] = None) -> None:
        super().__init__(init_cfg)
        self.bn = nn.BatchNorm1d(input_features, affine=affine, eps=eps)

    def forward(
            self,
            inputs: Tuple[List[torch.Tensor]]) -> Tuple[List[torch.Tensor]]:
        """The forward function."""
        # Only apply batch norm to cls_token
        inputs = [self.bn(input_) for input_ in inputs]
        return tuple(inputs)
