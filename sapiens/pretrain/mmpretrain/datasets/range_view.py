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


def load_range_stack(range_dir, history_len, az_shift=0):
    """Carrega history_len frames de range-view e devolve tokens (num_patches, patch_dim).
    Canal usado: rango (normalizado). Frames empilhados como canais (tempo).

    az_shift: roll de columnas nativas (augmentación). Roda a cena +az_shift cols
    em torno do yaw = rotação de θ = -az_shift·(2π/W_NATIVE). A trajetória DEVE
    rodar por θ de forma consistente (ver __getitem__). Sinal validado por IoU."""
    chans = []
    for t in range(history_len):
        ri = np.load(os.path.join(range_dir, f'{t}.npy'))   # (64, 2650, 2)
        if az_shift:
            ri = np.roll(ri, az_shift, axis=1)              # roll nativo (360° wrap)
        rng = ri[:, ::AZ_STRIDE, 0][:, :RANGE_W].astype(np.float32)  # (64, 512)
        rng[rng < 0] = 0.0                                  # no-return -> 0
        rng /= MAX_RANGE
        chans.append(rng)
    img = np.stack(chans, axis=-1)                          # (64, 512, history_len)
    H, W, C = img.shape
    gh, gw = H // PATCH, W // PATCH                          # 4, 32 -> 128 patches
    patches = (img.reshape(gh, PATCH, gw, PATCH, C)
                  .transpose(0, 2, 1, 3, 4)
                  .reshape(gh * gw, PATCH * PATCH * C))      # (128, 16*16*history_len)
    return patches.astype(np.float32)


def num_tokens():
    return (64 // PATCH) * (RANGE_W // PATCH)                # 128


def patch_dim(history_len):
    return PATCH * PATCH * history_len                       # 16*16*hl


def load_range_sweep(npy_path):
    """UN solo sweep (1 frame) -> tokens (128, 256). Canal: rango normalizado.
    Para la validación del encoder estilo Claudine (overfit en N sweeps)."""
    ri = np.load(npy_path)                                   # (64, 2650, 2)
    rng = ri[:, ::AZ_STRIDE, 0][:, :RANGE_W].astype(np.float32)
    rng[rng < 0] = 0.0
    rng /= MAX_RANGE
    H, W = rng.shape
    gh, gw = H // PATCH, W // PATCH
    patches = (rng.reshape(gh, PATCH, gw, PATCH)
                  .transpose(0, 2, 1, 3)
                  .reshape(gh * gw, PATCH * PATCH))           # (128, 256)
    return patches.astype(np.float32)


def unpatchify(patches):
    """tokens (128, 256) -> imagem (64, 512). Inverso de load_range_sweep (1 canal)."""
    gh, gw = 64 // PATCH, RANGE_W // PATCH
    img = (patches.reshape(gh, gw, PATCH, PATCH)
                  .transpose(0, 2, 1, 3)
                  .reshape(64, RANGE_W))
    return img


@DATASETS.register_module()
class RangeSweepDataset(BaseDataset):
    """Pré-treino MAE por SWEEP individual (cada frame = 1 amostra). Permite os
    testes de overfit em 10 / 100 / 1000 sweeps do plano de Claudine."""

    def __init__(self, data_root, pipeline=[], ann_file='', max_sweeps=0,
                 scenes=None, **kwargs):
        self.max_sweeps = max_sweeps
        self.scenes = set(scenes) if scenes is not None else None
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
        tok = load_range_sweep(self.data_list[idx]['npy'])
        return {'inputs': torch.from_numpy(tok).float()}


@DATASETS.register_module()
class RangeViewSequenceDataset(BaseDataset):
    """Pré-treino MAE: por cena, tokens da range-view (sem trajetória)."""

    def __init__(self, data_root, pipeline=[], ann_file='', history_len=5,
                 scenes=None, **kwargs):
        self.history_len = history_len
        self.scenes = set(scenes) if scenes is not None else None
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
        tokens = load_range_stack(self.data_list[idx]['range_dir'], self.history_len)
        return {'inputs': torch.from_numpy(tokens).float()}


@DATASETS.register_module()
class RangeViewTrajectoryDataset(TrajectoryDataset):
    """Fine-tuning: mesma lógica de trajetória do TrajectoryDataset, mas a cena é
    a range-view (tokens) em vez de vóxels."""

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
        relative_norm = np.clip((relative - mean_rel) / std_rel, -5.0, 5.0)
        obj_history_flat = relative_norm[:self.history_len].reshape(-1).astype(np.float32)
        obj_future_flat = relative_norm[
            self.history_len:self.history_len + self.pred_len].reshape(-1).astype(np.float32)

        # --- cena: range-view em tokens (rodada por az_shift, consistente) ---
        range_dir = os.path.join(self.data_root, 'range_files', scene)
        tokens = load_range_stack(range_dir, self.history_len, az_shift=az_shift)

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
