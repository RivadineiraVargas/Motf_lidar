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
                 augment=False,
                 eval_windows=1,
                 clip_norm=5.0,
                 norm_scale=None,
                 **kwargs):
        self.clip_norm = clip_norm
        self.norm_scale = norm_scale
        self.eval_windows = eval_windows   # antes de super(): full_init() ya llama load_data_list
        self.sequence_len = sequence_len
        self.history_len = history_len
        self.pred_len = pred_len
        self.voxel_res = voxel_res
        self.spatial_range = spatial_range
        # Salto máximo plausível (m) entre frames consecutivos (~0.1s no Waymo).
        # Descarta tracks corrompidos pelo bug de associação: os bbox usam índice
        # por frame (não track ID persistente), então quando um objeto some os
        # índices deslizam e o "mesmo" id salta para outro carro a dezenas de m.
        self.max_jump = max_jump
        self.scenes   = set(scenes) if scenes is not None else None
        self.augment  = augment

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
            n_desalineados = 0
            for obj_id, track in object_tracks.items():
                track.sort(key=lambda x: x[0])
                frames    = [f for f, _, _ in track]
                centers   = [c for _, c, _ in track]
                globals_  = [g for _, _, g in track]
                if len(centers) < self.sequence_len:
                    continue

                # ARREGLO 30/08 (hallazgo 6). `centers` se indexa por POSICIÓN EN
                # EL TRACK: solo contiene los frames donde ESTE objeto fue
                # etiquetado. Antes ese mismo índice se usaba como número de frame
                # ABSOLUTO para cargar los .bin de la escena, así que un objeto que
                # aparecía recién en el frame 6 recibía la escena de los bins 0..4.
                # Medido sobre las escenas de validación del fold 0: el 43% de los
                # objetos veía la escena de OTRO momento, con desfases de hasta 6
                # frames sobre 11 disponibles.
                # Ahora cada ventana lleva `frame0`, el frame absoluto real donde
                # empieza, y __getitem__ carga la escena desde ahí.
                n_lidar = getattr(self, 'n_lidar_frames', 11)
                ventanas = 0
                for k in range(len(centers) - self.sequence_len + 1):
                    if ventanas >= self.eval_windows:
                        break
                    f0 = frames[k]
                    # La escena necesita history_len sweeps consecutivos desde f0.
                    if f0 + self.history_len > n_lidar:
                        break                      # ya no entra ninguna ventana más
                    # El tramo con LiDAR debe ser contiguo en frames absolutos: si
                    # el track tiene huecos, la "historia" abarcaría más tiempo del
                    # que dice y no alinearía con los sweeps.
                    tramo = frames[k:k + self.history_len]
                    if tramo != list(range(f0, f0 + self.history_len)):
                        n_desalineados += 1
                        continue
                    # Filtro de consistencia POR VENTANA (antes solo miraba la
                    # primera): un track que se corrompe más adelante pasaba el
                    # filtro y sus frames rotos entraban en las ventanas extra.
                    seq_g = globals_[k:k + self.sequence_len]
                    jumps = [np.linalg.norm(seq_g[j + 1] - seq_g[j])
                             for j in range(len(seq_g) - 1)]
                    if jumps and max(jumps) > self.max_jump:
                        n_dropped += 1
                        continue
                    data_list.append({
                        'scene_name': scene,
                        'object_id':  obj_id,
                        't_start':    k,            # índice dentro del track
                        'frame0':     f0,           # frame ABSOLUTO de la escena
                        'centers':    centers[k:k + self.sequence_len],
                    })
                    ventanas += 1

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

    def _augment(self, relative, voxel_sequences):
        """Rotación aleatoria 0/90/180/270° + flip opcional en plano XY.
        Aplicada consistentemente a trayectoria y vóxeles.
        """
        k    = np.random.randint(0, 4)   # número de rotaciones de 90°
        flip = bool(np.random.randint(0, 2))

        # Rotar trayectoria XY, dejar Z intacto
        rel = relative.copy()
        if flip:
            rel[:, 0] = -rel[:, 0]
        angles = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
        cos_a, sin_a = np.cos(angles[k]), np.sin(angles[k])
        x_new = rel[:, 0] * cos_a - rel[:, 1] * sin_a
        y_new = rel[:, 0] * sin_a + rel[:, 1] * cos_a
        rel[:, 0] = x_new
        rel[:, 1] = y_new

        # Rotar grilla de vóxeles (grid_x, grid_y, grid_z) en plano XY
        aug_voxels = []
        for grid in voxel_sequences:
            g = grid.copy()
            if flip:
                g = g[::-1, :, :].copy()
            if k > 0:
                g = np.rot90(g, k=k, axes=(0, 1)).copy()
            aug_voxels.append(g)

        return rel, aug_voxels

    def __getitem__(self, idx):
        item   = self.data_list[idx]
        scene  = item['scene_name']
        centers = item['centers']

        # Deslocamentos relativos ao primeiro frame
        ref_center = np.array(centers[0])
        relative   = np.array([np.array(c) - ref_center for c in centers])

        # Tokens de cena (cargar antes de augmentación para rotar consistentemente)
        scene_bin = os.path.join(self.data_root, 'bin_files', scene)
        voxel_sequences = []
        # frame0 es el frame ABSOLUTO donde arranca la ventana: es lo que alinea
        # la escena con la trayectoria del objeto (ver hallazgo 6 en load_data_list).
        # El fallback a t_start solo existe para items viejos serializados.
        t0 = item.get('frame0', item.get('t_start', 0))
        for i in range(t0, t0 + self.history_len):
            points = self.load_bin(os.path.join(scene_bin, f"{i}.bin"))
            grid   = self.point_cloud_to_voxel_grid(points)
            voxel_sequences.append(grid)

        # Augmentación: rotar trayectoria + vóxeles consistentemente
        if self.augment:
            relative, voxel_sequences = self._augment(relative, voxel_sequences)

        # Normalizar só com o histórico (sem data leakage).
        # std mínimo 0.5m para evitar escala explosiva em objetos quase estáticos.
        # Normalización con media/desvío del HISTÓRICO (5 puntos, ~0.5 s).
        # OJO: como std_rel tiene piso 0.5, el clip a ±5 equivale a ±2.5 m desde
        # el centro del histórico — pero los objetos se desplazan ~8.8 m en 3 s.
        # Resultado: el 32% de los valores del FUTURO se recortan (el histórico,
        # 0%). Eso trunca el objetivo justo en los objetos que más se mueven y
        # vuelve OPTIMISTAS los ADE absolutos. Las comparaciones entre modelos no
        # se ven afectadas (todos comparten el mismo objetivo recortado).
        # clip_norm=None desactiva el recorte: se usa en EVALUACIÓN para medir el
        # error real contra la trayectoria completa.
        history_rel = relative[:self.history_len]
        mean_rel    = history_rel.mean(axis=0)
        # norm_scale: escala FIJA en metros para normalizar. El modo histórico
        # (norm_scale=None) calcula el desvío con los 5 puntos del histórico
        # (~0.5 s) y lo aplica también al futuro (3 s, decenas de metros): salen
        # valores de hasta 28 y el entrenamiento se vuelve inestable (pérdida que
        # oscila 36 -> 11 -> 17) y 11x más lento. El clip a ±5 tapaba eso
        # truncando el objetivo. Una escala fija acota el rango de salida sin
        # truncar nada.
        if self.norm_scale is not None:
            std_rel = np.full(3, float(self.norm_scale))
        else:
            std_rel = np.maximum(history_rel.std(axis=0), 0.5)
        relative_norm = (relative - mean_rel) / std_rel
        if self.clip_norm is not None:
            relative_norm = np.clip(relative_norm, -self.clip_norm, self.clip_norm)

        obj_history_flat = relative_norm[:self.history_len].reshape(-1).astype(np.float32)
        obj_future_flat  = relative_norm[
            self.history_len:self.history_len + self.pred_len
        ].reshape(-1).astype(np.float32)

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