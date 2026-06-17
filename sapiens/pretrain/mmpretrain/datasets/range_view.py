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
PATCH     = 16     # parche 16x16
MAX_RANGE = 75.0   # normalización del canal de rango


def load_range_stack(range_dir, history_len):
    """Carrega history_len frames de range-view e devolve tokens (num_patches, patch_dim).
    Canal usado: rango (normalizado). Frames empilhados como canais (tempo)."""
    chans = []
    for t in range(history_len):
        ri = np.load(os.path.join(range_dir, f'{t}.npy'))   # (64, 2650, 2)
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
        centers = item['centers']

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

        # --- cena: range-view em tokens ---
        range_dir = os.path.join(self.data_root, 'range_files', scene)
        tokens = load_range_stack(range_dir, self.history_len)

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
