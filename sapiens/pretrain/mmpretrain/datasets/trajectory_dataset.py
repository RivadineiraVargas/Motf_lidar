# trajectory_dataset.py — versão corrigida
import os
import numpy as np
import torch
from .base_dataset import BaseDataset
from mmpretrain.registry import DATASETS


@DATASETS.register_module()
class TrajectoryDataset(BaseDataset):

    def __init__(self,
                 data_root,
                 pipeline=[],
                 ann_file='',
                 sequence_len=10,
                 history_len=5,
                 pred_len=5,
                 voxel_res=0.5,
                 spatial_range=[-40, 40, -40, 40, -2, 4],
                 max_jump=5.0,
                 scenes=None,
                 **kwargs):
        self.sequence_len = sequence_len
        self.history_len = history_len
        self.pred_len = pred_len
        self.voxel_res = voxel_res
        self.spatial_range = spatial_range
        # Lista branca de cenas a incluir (p/ split train/val). None = todas.
        self.scenes = set(scenes) if scenes is not None else None
        # Salto máximo plausível (m) entre frames consecutivos (~0.1s no Waymo).
        # Descarta tracks corrompidos pelo bug de associação: os bbox usam índice
        # por frame (não track ID persistente), então quando um objeto some os
        # índices deslizam e o "mesmo" id salta para outro carro a dezenas de m.
        self.max_jump = max_jump

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
        # Carregar uma única vez — BaseDataset não conhece o formato bin/bbox
        self.data_list = self.load_data_list()

    def load_pose(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
        matrix = [
            list(map(float, l.strip().split()))
            for l in lines
            if len(l.strip().split()) == 4
        ]
        return np.array(matrix) if len(matrix) == 4 else None

    def parse_bbox_file(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
        vertices = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 3:
                vertices.append([float(p) for p in parts])
        if len(vertices) == 8:
            return np.mean(vertices, axis=0)
        return None

    def load_data_list(self):
        bin_root  = os.path.join(self.data_root, 'bin_files')
        bbox_root = os.path.join(self.data_root, 'objs_bbox')
        pose_root = os.path.join(self.data_root, 'poses')

        if not all(os.path.isdir(p) for p in [bin_root, bbox_root, pose_root]):
            return []

        scenes = sorted([
            d for d in os.listdir(bin_root)
            if os.path.isdir(os.path.join(bin_root, d))
        ])
        # Filtrar pela lista branca (split train/val), se fornecida
        if self.scenes is not None:
            scenes = [s for s in scenes if s in self.scenes]

        data_list = []
        for scene in scenes:
            scene_bbox = os.path.join(bbox_root, scene)
            scene_pose = os.path.join(pose_root, scene)
            if not os.path.isdir(scene_bbox) or not os.path.isdir(scene_pose):
                continue

            frame_dirs = sorted([
                d for d in os.listdir(scene_bbox)
                if os.path.isdir(os.path.join(scene_bbox, d)) and d.isdigit()
            ])
            if len(frame_dirs) < self.sequence_len:
                continue

            object_tracks = {}
            for frame_dir in frame_dirs:
                frame_path = os.path.join(scene_bbox, frame_dir)
                pose_path  = os.path.join(scene_pose, frame_dir + '.txt')
                if not os.path.exists(pose_path):
                    continue
                pose = self.load_pose(pose_path)
                if pose is None:
                    continue

                for obj_file in sorted(os.listdir(frame_path)):
                    if not obj_file.endswith('.txt'):
                        continue
                    obj_id = obj_file.replace('.txt', '')
                    center_global = self.parse_bbox_file(
                        os.path.join(frame_path, obj_file)
                    )
                    if center_global is None:
                        continue

                    center_hom    = np.append(center_global, 1.0)
                    center_sensor = (np.linalg.inv(pose) @ center_hom)[:3]

                    object_tracks.setdefault(obj_id, []).append(
                        (int(frame_dir), center_sensor, np.array(center_global))
                    )

            n_dropped = 0
            for obj_id, track in object_tracks.items():
                track.sort(key=lambda x: x[0])
                centers   = [c for _, c, _ in track]
                globals_  = [g for _, _, g in track]
                if len(centers) < self.sequence_len:
                    continue

                # Filtro de consistência: salto global implausível => track corrompido
                seq_g = globals_[:self.sequence_len]
                jumps = [np.linalg.norm(seq_g[k + 1] - seq_g[k])
                         for k in range(self.sequence_len - 1)]
                if max(jumps) > self.max_jump:
                    n_dropped += 1
                    continue

                data_list.append({
                    'scene_name': scene,
                    'object_id':  obj_id,
                    'centers':    centers[:self.sequence_len],
                })

            if n_dropped:
                print(f'[TrajectoryDataset] cena {scene}: {n_dropped} tracks '
                      f'descartados por salto > {self.max_jump}m (bug de associação)')

        return data_list

    def load_bin(self, path):
        return np.fromfile(path, dtype=np.float32).reshape(-1, 4)

    def point_cloud_to_voxel_grid(self, points):
        """Voxelização vetorizada."""
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

        ix = np.clip(
            ((pts[:, 0] - self.spatial_range[0]) / self.voxel_res).astype(np.int32),
            0, self.grid_x - 1
        )
        iy = np.clip(
            ((pts[:, 1] - self.spatial_range[2]) / self.voxel_res).astype(np.int32),
            0, self.grid_y - 1
        )
        iz = np.clip(
            ((pts[:, 2] - self.spatial_range[4]) / self.voxel_res).astype(np.int32),
            0, self.grid_z - 1
        )
        grid[ix, iy, iz] = 1.0
        return grid

    def __getitem__(self, idx):
        item   = self.data_list[idx]
        scene  = item['scene_name']
        centers = item['centers']

        # Deslocamentos relativos ao primeiro frame
        ref_center = np.array(centers[0])
        relative   = np.array([np.array(c) - ref_center for c in centers])

        # Normalizar só com o histórico (sem data leakage).
        # std mínimo 0.5m para evitar escala explosiva em objetos quase estáticos.
        # Clip a [-5, 5] para descartar tracks fora do range do sensor.
        history_rel = relative[:self.history_len]
        mean_rel    = history_rel.mean(axis=0)
        std_rel     = np.maximum(history_rel.std(axis=0), 0.5)
        relative_norm = np.clip((relative - mean_rel) / std_rel, -5.0, 5.0)

        obj_history_flat = relative_norm[:self.history_len].reshape(-1).astype(np.float32)
        obj_future_flat  = relative_norm[
            self.history_len:self.history_len + self.pred_len
        ].reshape(-1).astype(np.float32)

        # Tokens de cena
        scene_bin = os.path.join(self.data_root, 'bin_files', scene)
        voxel_sequences = []
        for i in range(self.history_len):
            points = self.load_bin(os.path.join(scene_bin, f"{i}.bin"))
            grid   = self.point_cloud_to_voxel_grid(points)
            voxel_sequences.append(grid)

        history = np.stack(voxel_sequences, axis=0)
        tokens  = history.reshape(self.history_len, -1).T  # (num_voxels, history_len)

        return {
            'inputs':          torch.from_numpy(tokens).float(),
            'obj_history_flat': torch.tensor(obj_history_flat),
            'obj_future_flat':  torch.tensor(obj_future_flat),
            'norm_mean':        torch.tensor(mean_rel.astype(np.float32)),
            'norm_std':         torch.tensor(std_rel.astype(np.float32)),
            'ref_center':       torch.tensor(ref_center.astype(np.float32)),
            'scene_name':       item['scene_name'],
            'object_id':        item['object_id'],
        }