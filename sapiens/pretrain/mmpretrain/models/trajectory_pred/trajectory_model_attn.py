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
                 finetune_blocks=0,
                 use_gate=True,
                 gate_init=0.0,
                 freeze_gate=False,
                 predict_uncertainty=False,
                 num_modes=1,
                 cls_weight=1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder = MODELS.build(encoder)
        # finetune_blocks=N descongela los ÚLTIMOS N bloques del encoder, dejando
        # el resto congelado. JointMotion (Wagner 2024) no congela nada — usa el
        # pre-entrenamiento como INICIALIZACIÓN, no como extractor fijo — pero
        # descongelar los 302M enteros da OOM en 8 GB con lote 16, y bajar el lote
        # a 4 degrada el modelo por sí solo (ADE 4.84 -> 8.29, medido el 28/08).
        # Descongelar solo la cola entra en memoria con lote 16 y permite comparar
        # contra todo lo medido cambiando UNA sola variable.
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            if finetune_blocks > 0:
                capas = getattr(self.encoder, 'layers', None)
                if capas is None:
                    raise AttributeError('el encoder no expone .layers; no se '
                                         'puede descongelar por bloques')
                for capa in capas[-finetune_blocks:]:
                    for p in capa.parameters():
                        p.requires_grad = True
        self.finetune_blocks = finetune_blocks
        self.history_len = history_len
        self.pred_len = pred_len
        self.embed_dim = embed_dim
        # use_gate=False -> rama de escena SIEMPRE activa (sin tanh(gate)).
        # Test limpio (Opción C): da gradiente completo a scene_proj/cross_attn
        # para que aprendan, rompiendo el candado del gate iniciado en 0.
        self.use_gate = use_gate

        self.history_proj = nn.Linear(history_len * 3, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Rama de escena: normalizar + proyectar a poucas dims para que NÃO afogue
        # o histórico (15 dims) na concatenação. Ver diagnóstico waymo_10 (1 cena).
        self.scene_norm = nn.LayerNorm(embed_dim)
        self.scene_proj = nn.Linear(embed_dim, scene_dim)
        # Gate aprendível. gate_init = valor INICIAL de tanh(scene_gate):
        #   0.0  -> arranca ignorando la escena (candado de gradiente: nunca abre).
        #   ~0.5 -> arranca usando la escena a medias, dándole gradiente real a la
        #           rama para que aprenda; luego el gate sube/baja según ayude o no.
        # Rompe el candado manteniendo la decisión aprendible. atanh(v) tal que
        # tanh(scene_gate_init) = gate_init.
        gate_init = float(max(min(gate_init, 0.99), -0.99))
        self.scene_gate = nn.Parameter(torch.atanh(torch.tensor([gate_init])))
        # freeze_gate=True -> el gate NO se aprende, queda clavado en gate_init.
        # Con gate_init=0.0 esto da el CONTROL DE ARQUITECTURA: la rama de escena
        # aporta exactamente 0, pero el modelo conserva cross_attn, scene_proj,
        # scene_norm y el mismo decoder. Comparar contra el baseline MLP mide
        # CAPACIDAD; comparar el gated contra este mide ESCENA. En la Fase 2 la
        # ausencia de este control confundió ambas cosas durante 14 experimentos
        # (ver docs/EXPERIMENTOS_DECODER.md, exp. 14).
        self.freeze_gate = freeze_gate
        if freeze_gate:
            self.scene_gate.requires_grad_(False)

        # predict_uncertainty=True -> el decoder predice media Y log-varianza por
        # cada coordenada (incerteza aleatoria). El PDF la pide: covarianza por pose
        # para el Behavior Selector. Salida doble: pred_len*3 (mu) + pred_len*3 (log_var).
        self.predict_uncertainty = predict_uncertainty

        # PREDICCIÓN MULTIMODAL (Wayformer/MTR). num_modes=K hace que el decoder
        # emita K trayectorias completas más un logit por modo. El futuro es
        # genuinamente multimodal —el auto dobla o sigue— y un modelo de k=1
        # aprende el PROMEDIO de los futuros posibles, que no es ninguno de ellos.
        #
        # num_modes=1 reproduce EXACTAMENTE el comportamiento anterior: no se crea
        # la cabeza de clasificación, la pérdida es la misma MSE (o NLL) y la
        # salida tiene la misma forma. Es lo que permite comparar contra los
        # experimentos 15-22 sin cambiar nada más.
        #
        # El plan de Claudine lo pide en ese orden (Sec. 10, resumido en
        # docs/ESTUDIO_WAYFORMER.md): "k=1 al inicio; k>1 con winner-takes-all
        # después".
        assert num_modes >= 1, 'num_modes debe ser >= 1'
        self.num_modes = num_modes
        self.cls_weight = cls_weight
        out_dim = num_modes * pred_len * 3 * (2 if predict_uncertainty else 1)
        # Cabeza de clasificación de modo: SOLO si hay más de uno. Con K=1 su
        # pérdida sería una constante (log 1 = 0) y sus parámetros no recibirían
        # gradiente útil, pero igual cambiarían la inicialización del resto.
        self.mode_head = nn.Linear(hidden_dim, num_modes) if num_modes > 1 else None

        input_dim = scene_dim + history_len * 3
        # El tronco se separa de la última capa para que la cabeza de modo lea el
        # MISMO rasgo que la de regresión (así decide con la misma información).
        # Con K=1 el conjunto tronco+cabeza es idéntico al nn.Sequential anterior:
        # mismas capas, mismos tamaños, mismo orden de creación de parámetros.
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.reg_head = nn.Linear(hidden_dim, out_dim)

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

        # 4. Normalizar, projetar a poucas dims e (opcionalmente) aplicar gate
        scene_feat = self.scene_proj(self.scene_norm(attn_out))   # (B, scene_dim)
        if self.use_gate:
            scene_feat = torch.tanh(self.scene_gate) * scene_feat  # gate -> arranca em 0
        # use_gate=False: escena siempre activa, gradiente completo a la rama

        # 5. Concatenar com história original e decodificar
        combined = torch.cat([scene_feat, obj_history_flat], dim=1)  # (B, scene_dim + history_len*3)
        rasgo = self.decoder(combined)                               # (B, hidden_dim)
        out = self.reg_head(rasgo)
        logits = self.mode_head(rasgo) if self.mode_head is not None else None

        B, K, D = out.size(0), self.num_modes, self.pred_len * 3

        if not self.predict_uncertainty:
            modos = out.view(B, K, D)                                # (B, K, pred_len*3)
            if mode == 'loss':
                if obj_future_flat is None:
                    raise ValueError("obj_future_flat obrigatório no modo 'loss'")
                return self._perdida_wta(modos, logits, obj_future_flat)
            if mode == 'predict_multi':
                # todas las hipótesis + sus probabilidades, para minADE_k
                probs = (torch.softmax(logits, dim=1) if logits is not None
                         else out.new_ones(B, 1))
                return modos, probs
            # 'predict': el modo MÁS PROBABLE. Con K=1 es el único, así que
            # devuelve exactamente lo mismo que antes — es lo que mantiene
            # comparables los experimentos 15-22.
            return self._modo_mas_probable(modos, logits)

        # Con incerteza: la salida trae media Y log-varianza por modo.
        modos = out.view(B, K, 2 * D)
        mu, log_var = modos[..., :D], modos[..., D:]                 # cada (B, K, pred_len*3)
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        if mode == 'loss':
            if obj_future_flat is None:
                raise ValueError("obj_future_flat obrigatório no modo 'loss'")
            return self._perdida_wta(mu, logits, obj_future_flat, log_var=log_var)
        if mode == 'predict_multi':
            probs = (torch.softmax(logits, dim=1) if logits is not None
                     else out.new_ones(B, 1))
            return mu, probs
        if mode == 'uncertainty':
            k = self._indice_mas_probable(logits, B, mu.device)
            idx = k.view(B, 1, 1).expand(B, 1, D)
            return mu.gather(1, idx).squeeze(1), log_var.gather(1, idx).squeeze(1)
        return self._modo_mas_probable(mu, logits)

    # ---- multimodalidad: selección de modo y pérdida winner-takes-all ----

    def _indice_mas_probable(self, logits, B, dev):
        if logits is None:
            return torch.zeros(B, dtype=torch.long, device=dev)
        return logits.argmax(dim=1)

    def _modo_mas_probable(self, modos, logits):
        """(B, K, D) -> (B, D) tomando el modo de mayor logit."""
        B, _, D = modos.shape
        k = self._indice_mas_probable(logits, B, modos.device)
        return modos.gather(1, k.view(B, 1, 1).expand(B, 1, D)).squeeze(1)

    def _perdida_wta(self, modos, logits, objetivo, log_var=None):
        """Winner-takes-all (Wayformer, MTR): solo el modo MÁS CERCANO al futuro
        real recibe gradiente de regresión; los otros quedan libres para
        especializarse en futuros distintos. Sin esto, K modos entrenados todos
        contra el mismo objetivo colapsan a K copias del promedio, que es
        justamente lo que la multimodalidad viene a evitar.

        El ganador se elige por distancia L2 media sobre los waypoints (la misma
        cantidad que mide el ADE). Como `norm_scale` es una escala FIJA en metros,
        el orden en espacio normalizado es el mismo que en metros: elegir acá
        equivale a elegir por ADE real.

        La cabeza de modo se entrena con entropía cruzada contra el ganador,
        DETACHADO: la clasificación aprende a predecir qué modo será el mejor, sin
        arrastrar a la regresión hacia el modo que hoy es más fácil de clasificar.
        """
        B, K, D = modos.shape
        obj = objetivo.unsqueeze(1)                                  # (B, 1, D)

        if K == 1:
            # Camino idéntico al anterior: sin selección ni clasificación.
            if log_var is None:
                return dict(loss=nn.functional.mse_loss(modos.squeeze(1), objetivo))
            nll = 0.5 * (log_var.squeeze(1)
                         + (objetivo - modos.squeeze(1)) ** 2 * torch.exp(-log_var.squeeze(1)))
            return dict(loss=nll.mean())

        # distancia por waypoint: (B, K, pred_len, 3) -> L2 en XYZ -> media en t
        dif = (modos - obj).view(B, K, self.pred_len, 3)
        dist = dif.norm(dim=-1).mean(dim=-1)                         # (B, K)
        ganador = dist.argmin(dim=1)                                 # (B,)

        idx = ganador.view(B, 1, 1).expand(B, 1, D)
        mejor = modos.gather(1, idx).squeeze(1)                      # (B, D)
        if log_var is None:
            reg = nn.functional.mse_loss(mejor, objetivo)
        else:
            lv = log_var.gather(1, idx).squeeze(1)
            reg = (0.5 * (lv + (objetivo - mejor) ** 2 * torch.exp(-lv))).mean()

        cls = nn.functional.cross_entropy(logits, ganador.detach())
        # OJO con los nombres: mmengine hace
        #     loss = sum(v for k, v in log_vars if 'loss' in k)
        # (BaseModel.parse_losses), así que CUALQUIER clave que contenga "loss"
        # se suma al total. Llamarlas `loss_reg`/`loss_cls` duplicaría el valor
        # registrado —serían reg+cls sumados dos veces— y, aunque estén
        # detachadas y no toquen el gradiente, este proyecto LEE las curvas de
        # pérdida para diagnosticar (así se encontró el problema de `gate_init` y
        # el sobreajuste del experimento 22). Una curva al doble sería peor que
        # no tenerla. Por eso van sin "loss" en el nombre.
        return dict(loss=reg + self.cls_weight * cls,
                    wta_reg=reg.detach(), wta_cls=cls.detach())