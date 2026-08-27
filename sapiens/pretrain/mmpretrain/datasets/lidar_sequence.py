# lidar_sequence.py — versão corrigida
import os
import numpy as np
import torch
from .base_dataset import BaseDataset
from mmpretrain.registry import DATASETS


@DATASETS.register_module()
class LidarSequenceDataset(BaseDataset):

    def __init__(self,
                 data_root,
                 pipeline=[],
                 ann_file='',
                 sequence_len=10,
                 history_len=5,          # corrigido: default 5, não 100
                 max_windows=1,          # >1 = varias ventanas por escena
                 geo_target=False,       # True = objetivo centroide (GeoMAE)
                 voxel_res=0.5,
                 spatial_range=[-40, 40, -40, 40, -2, 4],
                 mask_ratio=0.75,
                 scenes=None,
                 **kwargs):
        self.sequence_len = sequence_len
        self.history_len = history_len
        self.max_windows = max_windows
        self.geo_target = geo_target
        self.voxel_res = voxel_res
        self.spatial_range = spatial_range
        self.mask_ratio = mask_ratio
        # Restringe a una whitelist de escenas (protocolo 10/100/1000 de Claudine)
        self.scenes = set(scenes) if scenes is not None else None

        self.grid_x = int((spatial_range[1] - spatial_range[0]) / voxel_res)
        self.grid_y = int((spatial_range[3] - spatial_range[2]) / voxel_res)
        self.grid_z = int((spatial_range[5] - spatial_range[4]) / voxel_res)
        self.num_voxels = self.grid_x * self.grid_y * self.grid_z

        super().__init__(
            data_root=data_root,
            pipeline=pipeline,
            ann_file=ann_file,
            **kwargs
        )
        # BaseDataset não sabe carregar bin_files — forçar carga manual uma única vez
        if not hasattr(self, 'data_list') or len(self.data_list) == 0:
            self.data_list = self.load_data_list()

    def load_data_list(self):
        data_list = []
        bin_dir = os.path.join(self.data_root, 'bin_files')
        if not os.path.isdir(bin_dir):
            raise FileNotFoundError(f"Diretório não encontrado: {bin_dir}")

        scenes = sorted([
            d for d in os.listdir(bin_dir)
            if os.path.isdir(os.path.join(bin_dir, d))
        ])
        if self.scenes is not None:
            scenes = [s for s in scenes if s in self.scenes]

        for scene in scenes:
            scene_path = os.path.join(bin_dir, scene)
            bin_files = sorted([
                f for f in os.listdir(scene_path)
                if f.endswith('.bin')
            ])
            # Una VENTANA por posición de arranque, no una por escena.
            # Antes esto devolvía 1 ítem por escena: con 8 escenas de train, el
            # MAE se pre-entrenaba con 8 muestras (verificado el 26/08). Con 11
            # barridos y history_len=5 entran ~7 ventanas por escena.
            n_win = len(bin_files) - self.history_len + 1
            for t0 in range(max(0, min(n_win, self.max_windows))):
                data_list.append({
                    'scene_path': scene_path,
                    'bin_files': bin_files,
                    'scene_name': scene,
                    't0': t0,
                })

        return data_list

    def load_bin(self, path):
        return np.fromfile(path, dtype=np.float32).reshape(-1, 4)

    def point_cloud_to_voxel_grid(self, points):
        """Voxelização vetorizada — sem loop Python."""
        grid = np.zeros(
            (self.grid_x, self.grid_y, self.grid_z), dtype=np.float32
        )
        mask = (
            (points[:, 0] >= self.spatial_range[0]) &
            (points[:, 0] <  self.spatial_range[1]) &
            (points[:, 1] >= self.spatial_range[2]) &
            (points[:, 1] <  self.spatial_range[3]) &
            (points[:, 2] >= self.spatial_range[4]) &
            (points[:, 2] <  self.spatial_range[5])
        )
        pts = points[mask]
        if len(pts) == 0:
            return grid

        ix = ((pts[:, 0] - self.spatial_range[0]) / self.voxel_res).astype(np.int32)
        iy = ((pts[:, 1] - self.spatial_range[2]) / self.voxel_res).astype(np.int32)
        iz = ((pts[:, 2] - self.spatial_range[4]) / self.voxel_res).astype(np.int32)

        # Clipar para evitar out-of-bounds por erros de ponto flutuante
        ix = np.clip(ix, 0, self.grid_x - 1)
        iy = np.clip(iy, 0, self.grid_y - 1)
        iz = np.clip(iz, 0, self.grid_z - 1)

        # Indexação vetorizada — muito mais rápido que loop
        grid[ix, iy, iz] = 1.0
        return grid

    def __getitem__(self, idx):
        item = self.data_list[idx]

        voxel_sequences = []
        t0 = item.get('t0', 0)
        for bin_file in item['bin_files'][t0:t0 + self.history_len]:
            points = self.load_bin(os.path.join(item['scene_path'], bin_file))
            grid = self.point_cloud_to_voxel_grid(points)
            voxel_sequences.append(grid)

        # (history_len, X, Y, Z) → (num_voxels, history_len)
        history = np.stack(voxel_sequences, axis=0)
        tokens = history.reshape(self.history_len, -1).T

        out = {'inputs': torch.from_numpy(tokens).float()}
        if self.geo_target:
            # Objetivo estilo GeoMAE (Tian et al. 2023): en vez de reconstruir la
            # ocupación cruda, predecir el CENTROIDE de los puntos dentro de cada
            # vóxel — desplazamiento respecto del centro del vóxel, en [-1,1].
            # Reconstruir ocupación no obliga al encoder a codificar geometría
            # fina; el centroide sí. Se calcula sobre el ÚLTIMO barrido de la
            # ventana, que es el que el decoder de trayectorias usará como
            # "presente".
            pts = self.load_bin(os.path.join(
                item['scene_path'], item['bin_files'][t0 + self.history_len - 1]))
            out['geo'] = torch.from_numpy(self.voxel_centroids(pts)).float()
        return out

    def voxel_centroids(self, points):
        """(num_voxels, 3): centroide de los puntos de cada vóxel, relativo al
        centro del vóxel y normalizado a [-1, 1]. Los vóxeles vacíos quedan en
        NaN para que la pérdida los enmascare (misma idea que el valid_mask de
        pointmap_l1_loss.py de Sapiens)."""
        sr, vr = self.spatial_range, self.voxel_res
        out = np.full((self.grid_x * self.grid_y * self.grid_z, 3), np.nan, np.float32)
        m = ((points[:, 0] >= sr[0]) & (points[:, 0] < sr[1]) &
             (points[:, 1] >= sr[2]) & (points[:, 1] < sr[3]) &
             (points[:, 2] >= sr[4]) & (points[:, 2] < sr[5]))
        p = points[m][:, :3]
        if len(p) == 0:
            return out
        ix = np.clip(((p[:, 0]-sr[0])/vr).astype(int), 0, self.grid_x-1)
        iy = np.clip(((p[:, 1]-sr[2])/vr).astype(int), 0, self.grid_y-1)
        iz = np.clip(((p[:, 2]-sr[4])/vr).astype(int), 0, self.grid_z-1)
        lin = (ix*self.grid_y + iy)*self.grid_z + iz
        centro = np.stack([sr[0]+(ix+.5)*vr, sr[2]+(iy+.5)*vr, sr[4]+(iz+.5)*vr], 1)
        rel = (p - centro) / (vr/2.0)                      # -> [-1,1]
        suma = np.zeros_like(out); cnt = np.zeros(len(out), np.float32)
        np.add.at(suma, lin, rel); np.add.at(cnt, lin, 1.0)
        ocup = cnt > 0
        out[ocup] = suma[ocup] / cnt[ocup, None]
        return out