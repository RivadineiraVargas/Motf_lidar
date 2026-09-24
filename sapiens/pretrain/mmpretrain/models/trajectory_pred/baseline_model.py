import torch
import torch.nn as nn
from mmpretrain.registry import MODELS
from mmengine.model import BaseModel


@MODELS.register_module()
class BaselineTrajectoryModel(BaseModel):
    """Baseline puramente cinemático: predice el futuro solo desde la historia
    del objeto, sin ver la escena LiDAR. Es el control contra el que se mide si
    la escena aporta algo.

    `num_modes=K` le da las mismas K hipótesis que al modelo con escena
    (`TrajectoryModelWithAttention`), con la misma pérdida winner-takes-all. Sin
    esto la comparación k=6 contra k=1 solo se podría hacer en la arquitectura
    grande, que tarda 8× más por corrida.

    num_modes=1 (el default) reproduce EXACTAMENTE el comportamiento anterior:
    mismo módulo, mismos nombres de parámetro, misma pérdida, misma salida. Es lo
    que mantiene válidos los checkpoints de los experimentos 15-22.
    """

    def __init__(self, history_len=5, pred_len=5, hidden_dim=512,
                 num_modes=1, cls_weight=1.0, pad_dim=0, **kwargs):
        super().__init__(**kwargs)
        assert num_modes >= 1, 'num_modes debe ser >= 1'
        assert pad_dim >= 0, 'pad_dim debe ser >= 0'
        self.history_len = history_len
        self.pred_len = pred_len
        self.num_modes = num_modes
        self.cls_weight = cls_weight
        # pad_dim=N pega N columnas de CEROS a la entrada. No agrega ninguna
        # informacion — el objetivo es EXACTAMENTE el contrario: reproducir la
        # geometria de la primera capa de TrajectoryModelWithAttention, que
        # recibe scene_dim + history_len*3 y cuyas scene_dim columnas valen cero
        # cuando el gate esta congelado en 0 (`gate0`).
        #
        # POR QUE IMPORTA: nn.Linear inicializa con cota 1/sqrt(in_features). Con
        # in=15 la desviacion de los pesos sobre la historia es 0,1485; con in=79
        # es 0,0645 — 2,3x mas chica, sobre las MISMAS entradas. Es decir, `gate0`
        # y este baseline no se diferencian en informacion ni en capacidad (el MLP
        # es identico), sino en la ESCALA DE INICIALIZACION. Sin este control, el
        # -0,217 que `gate0` le saca al baseline en 5/5 folds se lee como "la
        # arquitectura aporta" cuando podria ser solo eso.
        #
        # pad_dim=0 (el default) deja input_dim, los nombres de los parametros y
        # la salida EXACTAMENTE como estaban: los checkpoints de los exps. 15-30
        # siguen cargando sin cambios.
        self.pad_dim = pad_dim
        input_dim = history_len * 3 + pad_dim
        # UN SOLO Sequential, con la capa de salida adentro. Partirlo en tronco +
        # cabeza renombraría los parámetros (`decoder.6.*` -> `reg_head.*`) y los
        # checkpoints viejos cargarían con la capa de salida ALEATORIA sin avisar:
        # medido en el modelo con escena, el ADE pasó de 2,84 a 22,14. El forward
        # usa decoder[:-1] para el rasgo intermedio y decoder[-1] para la salida.
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modes * pred_len * 3)
        )
        self.mode_head = nn.Linear(hidden_dim, num_modes) if num_modes > 1 else None

    def forward(self, obj_history_flat, obj_future_flat=None, mode='loss', **kwargs):
        if self.pad_dim:
            ceros = obj_history_flat.new_zeros(obj_history_flat.size(0), self.pad_dim)
            # El orden importa: en TrajectoryModelWithAttention la escena va
            # PRIMERO (`cat([scene_feat, obj_history_flat])`). Se replica aca para
            # que las columnas de historia caigan en las mismas posiciones.
            obj_history_flat = torch.cat([ceros, obj_history_flat], dim=1)
        rasgo = self.decoder[:-1](obj_history_flat)
        out = self.decoder[-1](rasgo)
        logits = self.mode_head(rasgo) if self.mode_head is not None else None

        B, K, D = out.size(0), self.num_modes, self.pred_len * 3
        modos = out.view(B, K, D)

        if mode == 'loss':
            if obj_future_flat is None:
                raise ValueError("For loss mode, obj_future_flat must be provided")
            return self._perdida_wta(modos, logits, obj_future_flat)
        if mode == 'predict_multi':
            probs = (torch.softmax(logits, dim=1) if logits is not None
                     else out.new_ones(B, 1))
            return modos, probs
        # 'predict' -> el modo más probable. Con K=1 es el único, así que devuelve
        # lo mismo de siempre y con la misma forma (B, pred_len*3).
        if logits is None:
            return modos.squeeze(1)
        k = logits.argmax(dim=1)
        return modos.gather(1, k.view(B, 1, 1).expand(B, 1, D)).squeeze(1)

    def _perdida_wta(self, modos, logits, objetivo):
        """Winner-takes-all: solo el modo más cercano al futuro real recibe
        gradiente de regresión. Ver la explicación larga en
        trajectory_model_attn.py — la lógica es la misma."""
        B, K, D = modos.shape
        if K == 1:
            return dict(loss=nn.functional.mse_loss(modos.squeeze(1), objetivo))

        dif = (modos - objetivo.unsqueeze(1)).view(B, K, self.pred_len, 3)
        dist = dif.norm(dim=-1).mean(dim=-1)                  # (B, K)
        ganador = dist.argmin(dim=1)

        mejor = modos.gather(1, ganador.view(B, 1, 1).expand(B, 1, D)).squeeze(1)
        reg = nn.functional.mse_loss(mejor, objetivo)
        cls = nn.functional.cross_entropy(logits, ganador.detach())
        # Los nombres NO llevan "loss": mmengine hace
        # loss = sum(v for k, v in log_vars if 'loss' in k) en parse_losses, así
        # que llamarlas loss_reg/loss_cls duplicaría el valor registrado.
        return dict(loss=reg + self.cls_weight * cls,
                    wta_reg=reg.detach(), wta_cls=cls.detach())
