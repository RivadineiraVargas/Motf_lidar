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
                 voxel_res=0.5,
                 spatial_range=[-40, 40, -40, 40, -2, 4],
                 mask_ratio=0.75,
                 **kwargs):
        self.sequence_len = sequence_len
        self.history_len = history_len
        self.voxel_res = voxel_res
        self.spatial_range = spatial_range
        self.mask_ratio = mask_ratio

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

        for scene in scenes:
            scene_path = os.path.join(bin_dir, scene)
            bin_files = sorted([
                f for f in os.listdir(scene_path)
                if f.endswith('.bin')
            ])
            if len(bin_files) >= self.sequence_len:
                data_list.append({
                    'scene_path': scene_path,
                    'bin_files': bin_files,
                    'scene_name': scene,
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
        for bin_file in item['bin_files'][:self.history_len]:
            points = self.load_bin(os.path.join(item['scene_path'], bin_file))
            grid = self.point_cloud_to_voxel_grid(points)
            voxel_sequences.append(grid)

        # (history_len, X, Y, Z) → (num_voxels, history_len)
        history = np.stack(voxel_sequences, axis=0)
        tokens = history.reshape(self.history_len, -1).T

        return {'inputs': torch.from_numpy(tokens).float()}