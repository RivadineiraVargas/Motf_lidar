"""
range_view.py — Datasets para el track RANGE-VIEW del MOTF.

La range-view (64 beams x 2650 cols) es la representación nativa del LiDAR y, por
ser uma imagem 2D, é a que melhor reaproveita o Sapiens (modelo de imagens). Aqui
ela é "patchificada" em tokens (num_patches, patch_dim) — mesmo formato que o
MAEViT4D já consome (Linear(patch_dim -> embed_dim)). Assim o encoder é reusado
sem mudanças; só muda a tokenização (aqui, em vez de vóxels).

  RangeViewSequenceDataset    -> pré-treino MAE (cena, sem trajetória)
  RangeViewTrajectoryDataset  -> fine-tuning (cena + trajetória do objeto)
"""
import os
import numpy as np
import torch
from .trajectory_dataset import TrajectoryDataset
from .base_dataset import BaseDataset
from mmpretrain.registry import DATASETS

# Parámetros de tokenización de la range-view
RANGE_W   = 512    # columnas tras downsample+crop (de 2650)
AZ_STRIDE = 5      # 2650 -> ~530 (stride), luego crop a 512
W_NATIVE  = 2650   # columnas nativas (360° de azimut)
PATCH     = 16     # parche 16x16
MAX_RANGE = 75.0   # normalización del canal de rango

# CUANTA RESOLUCION SE TIRA HOY, medido. La range-view nativa es 64 x 2650 =
# 169.600 pixeles a 0,136 grados por columna. Con AZ_STRIDE=5 la resolucion
# efectiva pasa a 0,679 grados/columna —5x peor— y el crop a 512 de 530 deja el
# 97 % del azimut. Con parches de 16x16 quedan 4 x 32 = 128 tokens: MENOS que los
# 300 de la representacion en voxeles.
#
# Lo que eso cuesta: un auto de 4,5 m a 33 m (la distancia mediana de los objetos)
# subtiende 7,8 grados = 57,5 columnas nativas, que con el stride quedan en 11,5.
#
# Y lo que la range-view SI tiene y los voxeles no: cubre 348 grados y todo el
# alcance del sensor, asi que el objeto esta dentro en el 96,6 % de las ventanas
# (medido sobre el fold 0) contra el 11,0 % de la caja de voxeles de +-10 m — el
# defecto del experimento 28. Ese problema aqui no existe.
#
# Sin diezmar y con los mismos parches: 4 x 165 = 660 tokens, solo 2,2x los 300
# actuales. Es el mejor rendimiento por token de las opciones evaluadas.
#
# Los tres se pueden pasar por config al dataset. Los DEFAULT reproducen
# exactamente el comportamiento anterior: los experimentos que usaron range-view
# tienen que seguir dando lo mismo.


def _geom(az_stride=None, range_w=None, patch=None):
    """Resuelve la tokenizacion efectiva. Centraliza el default para que ninguna
    funcion se olvide de aplicarlo — que es como divergirian entre si."""
    s = AZ_STRIDE if az_stride is None else int(az_stride)
    p = PATCH if patch is None else int(patch)
    disp = (W_NATIVE + s - 1) // s                  # columnas que quedan tras el stride
    # range_w=None con stride distinto del default: usar TODAS las columnas que
    # queden tras el stride, en vez de heredar el 512 pensado para stride 5.
    if range_w is not None:
        w = int(range_w)
        if w > disp:
            # El slice [:, :w] NO rellena: devolveria menos columnas de las pedidas
            # y el ancho real quedaria por debajo del que num_tokens() le reporto
            # al modelo. Eso desemboca en el desajuste que _ensure_pos_embed
            # rechaza, pero con un mensaje que apunta al lugar equivocado.
            raise ValueError(
                f'range_w={w} pero con az_stride={s} solo hay {disp} columnas '
                f'disponibles de las {W_NATIVE} nativas.')
    elif s == AZ_STRIDE:
        w = RANGE_W
    else:
        w = disp
    # Recortar a multiplo del parche SIEMPRE, tambien con los defaults: si no, un
    # patch distinto del default revienta el reshape de load_range_stack.
    w = (w // p) * p
    if w == 0:
        raise ValueError(f'patch={p} es mayor que las {disp} columnas disponibles '
                         f'con az_stride={s}.')
    return s, w, p


def load_range_stack(range_dir, history_len, az_shift=0, t0=0,
                     az_stride=None, range_w=None, patch=None):
    """Carrega history_len frames de range-view e devolve tokens (num_patches, patch_dim).
    Canal usado: rango (normalizado). Frames empilhados como canais (tempo).

    az_shift: roll de columnas nativas (augmentación). Roda a cena +az_shift cols
    em torno do yaw = rotação de θ = -az_shift·(2π/W_NATIVE). A trajetória DEVE
    rodar por θ de forma consistente (ver __getitem__). Sinal validado por IoU."""
    S, RW, P = _geom(az_stride, range_w, patch)
    chans = []
    # t0: frame inicial de la ventana. Debe coincidir con el 't_start' de la
    # trayectoria, si no la escena y el histórico quedan desalineados.
    for t in range(t0, t0 + history_len):
        ri = np.load(os.path.join(range_dir, f'{t}.npy'))   # (64, 2650, 2)
        if az_shift:
            ri = np.roll(ri, az_shift, axis=1)              # roll nativo (360° wrap)
        rng = ri[:, ::S, 0][:, :RW].astype(np.float32)      # (64, RW)
        rng[rng < 0] = 0.0                                  # no-return -> 0
        rng /= MAX_RANGE
        chans.append(rng)
    img = np.stack(chans, axis=-1)                          # (64, RW, history_len)
    H, W, C = img.shape
    gh, gw = H // P, W // P
    patches = (img.reshape(gh, P, gw, P, C)
                  .transpose(0, 2, 1, 3, 4)
                  .reshape(gh * gw, P * P * C))             # (gh*gw, P*P*history_len)
    return patches.astype(np.float32)


def num_tokens(az_stride=None, range_w=None, patch=None):
    S, RW, P = _geom(az_stride, range_w, patch)
    return (64 // P) * (RW // P)                             # 128 con los defaults


def patch_dim(history_len, patch=None):
    P = PATCH if patch is None else int(patch)
    return P * P * history_len                               # 16*16*hl con el default


def load_range_sweep(npy_path, az_stride=None, range_w=None, patch=None):
    """UN solo sweep (1 frame) -> tokens (num_patches, P*P). Canal: rango
    normalizado. Para la validación del encoder estilo Claudine (overfit en N
    sweeps). Con los defaults da (128, 256), como antes."""
    S, RW, P = _geom(az_stride, range_w, patch)
    ri = np.load(npy_path)                                   # (64, 2650, 2)
    rng = ri[:, ::S, 0][:, :RW].astype(np.float32)
    rng[rng < 0] = 0.0
    rng /= MAX_RANGE
    H, W = rng.shape
    gh, gw = H // P, W // P
    patches = (rng.reshape(gh, P, gw, P)
                  .transpose(0, 2, 1, 3)
                  .reshape(gh * gw, P * P))
    return patches.astype(np.float32)


def unpatchify(patches, az_stride=None, range_w=None, patch=None):
    """tokens -> imagen (64, RW). Inverso exacto de load_range_sweep con LOS
    MISMOS parametros: llamarlo con otros reconstruye una imagen distinta sin
    avisar, o revienta el reshape. Con los defaults: (128,256) -> (64,512)."""
    _, RW, P = _geom(az_stride, range_w, patch)
    gh, gw = 64 // P, RW // P
    if patches.shape[0] != gh * gw:
        raise ValueError(
            f'unpatchify recibio {patches.shape[0]} tokens pero la geometria '
            f'(az_stride/range_w/patch) da {gh * gw}. Pasar los MISMOS parametros '
            f'con los que se tokenizo.')
    img = (patches.reshape(gh, gw, P, P)
                  .transpose(0, 2, 1, 3)
                  .reshape(64, RW))
    return img


@DATASETS.register_module()
class RangeSweepDataset(BaseDataset):
    """Pré-treino MAE por SWEEP individual (cada frame = 1 amostra). Permite os
    testes de overfit em 10 / 100 / 1000 sweeps do plano de Claudine."""

    def __init__(self, data_root, pipeline=[], ann_file='', max_sweeps=0,
                 scenes=None, az_stride=None, range_w=None, patch=None, **kwargs):
        self.max_sweeps = max_sweeps
        self.scenes = set(scenes) if scenes is not None else None
        # Misma tokenizacion parametrizable que los otros dos datasets. OJO al
        # visualizar: unpatchify() hay que llamarlo con ESTOS mismos valores.
        self.az_stride, self.range_w, self.patch = az_stride, range_w, patch
        super().__init__(data_root=data_root, pipeline=pipeline, ann_file=ann_file, **kwargs)
        self.data_list = self.load_data_list()

    def load_data_list(self):
        root = os.path.join(self.data_root, 'range_files')
        if not os.path.isdir(root):
            return []
        frames = []
        for s in sorted(os.listdir(root)):
            if self.scenes is not None and s not in self.scenes:
                continue
            d = os.path.join(root, s)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.endswith('.npy'):
                    frames.append(os.path.join(d, f))
        frames.sort()
        if self.max_sweeps:
            frames = frames[:self.max_sweeps]
        return [{'npy': f} for f in frames]

    def __getitem__(self, idx):
        tok = load_range_sweep(self.data_list[idx]['npy'],
                               az_stride=self.az_stride, range_w=self.range_w,
                               patch=self.patch)
        return {'inputs': torch.from_numpy(tok).float()}


@DATASETS.register_module()
class RangeViewSequenceDataset(BaseDataset):
    """Pré-treino MAE: por cena, tokens da range-view (sem trajetória)."""

    def __init__(self, data_root, pipeline=[], ann_file='', history_len=5,
                 scenes=None, az_stride=None, range_w=None, patch=None, **kwargs):
        self.history_len = history_len
        self.scenes = set(scenes) if scenes is not None else None
        # Misma tokenizacion parametrizable que RangeViewTrajectoryDataset. SIN
        # esto no se podia pre-entrenar el MAE a resolucion nativa: el encoder
        # habria visto 128 tokens y el decoder le daria 660, que es justo el
        # desajuste que _ensure_pos_embed ahora rechaza.
        self.az_stride, self.range_w, self.patch = az_stride, range_w, patch
        super().__init__(data_root=data_root, pipeline=pipeline, ann_file=ann_file, **kwargs)
        self.data_list = self.load_data_list()

    def load_data_list(self):
        root = os.path.join(self.data_root, 'range_files')
        if not os.path.isdir(root):
            return []
        scenes = sorted(d for d in os.listdir(root)
                        if os.path.isdir(os.path.join(root, d)))
        if self.scenes is not None:
            scenes = [s for s in scenes if s in self.scenes]
        out = []
        for s in scenes:
            d = os.path.join(root, s)
            n = len([f for f in os.listdir(d) if f.endswith('.npy')])
            if n >= self.history_len:
                out.append({'range_dir': d, 'scene_name': s})
        return out

    def __getitem__(self, idx):
        tokens = load_range_stack(self.data_list[idx]['range_dir'], self.history_len,
                                  az_stride=self.az_stride, range_w=self.range_w,
                                  patch=self.patch)
        return {'inputs': torch.from_numpy(tokens).float()}


@DATASETS.register_module()
class RangeViewTrajectoryDataset(TrajectoryDataset):
    """Fine-tuning: mesma lógica de trajetória do TrajectoryDataset, mas a cena é
    a range-view (tokens) em vez de vóxels."""

    def __init__(self, *a, az_stride=None, range_w=None, patch=None, **kw):
        # Los tres controlan la tokenizacion; None = el default historico (128
        # tokens). El config del MODELO debe declarar num_tokens acorde: pasarle
        # az_stride=1 al dataset y dejar num_tokens=128 en el modelo no da error,
        # da un pos-embed del tamanyo equivocado.
        self.az_stride, self.range_w, self.patch = az_stride, range_w, patch
        super().__init__(*a, **kw)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        scene = item['scene_name']
        centers = np.array(item['centers'], dtype=np.float64)   # (seq, 3)

        # --- augmentación azimut-shift DISCRETA (0/90/180/270°, como el voxel) ---
        # rotación consistente escena (roll de columnas) + trayectoria (rota XY)
        az_shift = 0
        if self.augment:
            quarter = int(np.random.randint(0, 4))              # 0/1/2/3 -> 0/90/180/270°
            az_shift = quarter * (W_NATIVE // 4)
            theta = -az_shift * 2.0 * np.pi / W_NATIVE          # signo validado por IoU
            c, s = np.cos(theta), np.sin(theta)
            x, y = centers[:, 0].copy(), centers[:, 1].copy()
            centers[:, 0] = x * c - y * s                       # rota XY por theta
            centers[:, 1] = x * s + y * c

        # --- trajetória (idêntico ao TrajectoryDataset) ---
        ref_center = np.array(centers[0])
        relative = np.array([np.array(c) - ref_center for c in centers])
        history_rel = relative[:self.history_len]
        mean_rel = history_rel.mean(axis=0)
        std_rel = np.maximum(history_rel.std(axis=0), 0.5)
        # Mismo recorte que TrajectoryDataset (ver nota allí): ±5 desvíos del
        # histórico ≈ ±2.5 m, mientras que en coordenadas relativas al ego los
        # objetos se desplazan ~40 m en 3 s. clip_norm=None lo desactiva.
        if getattr(self, 'norm_scale', None) is not None:
            std_rel = np.full(3, float(self.norm_scale))
        relative_norm = (relative - mean_rel) / std_rel
        if getattr(self, 'clip_norm', 5.0) is not None:
            relative_norm = np.clip(relative_norm, -self.clip_norm, self.clip_norm)
        obj_history_flat = relative_norm[:self.history_len].reshape(-1).astype(np.float32)
        obj_future_flat = relative_norm[
            self.history_len:self.history_len + self.pred_len].reshape(-1).astype(np.float32)

        # --- cena: range-view em tokens (rodada por az_shift, consistente) ---
        range_dir = os.path.join(self.data_root, 'range_files', scene)
        tokens = load_range_stack(range_dir, self.history_len, az_shift=az_shift,
                                  t0 = item.get('frame0', item.get('t_start', 0)),
                                  az_stride=self.az_stride, range_w=self.range_w,
                                  patch=self.patch)

        return {
            'inputs': torch.from_numpy(tokens).float(),
            'obj_history_flat': torch.tensor(obj_history_flat),
            'obj_future_flat': torch.tensor(obj_future_flat),
            'norm_mean': torch.tensor(mean_rel.astype(np.float32)),
            'norm_std': torch.tensor(std_rel.astype(np.float32)),
            'ref_center': torch.tensor(ref_center.astype(np.float32)),
            'scene_name': scene,
            'object_id': item['object_id'],
        }
