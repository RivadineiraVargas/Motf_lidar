"""
visualize_bev_trajectories.py
Visualização Bird's Eye View profissional:
  - Mapa de densidade LiDAR (estilo autonomous driving)
  - Bounding boxes de todos os objetos
  - Trajetórias reais (verde) vs preditas (vermelho) de todos os objetos
  - Gera imagem PNG + vídeo MP4

Uso:
    cd /home/lcad/lidar_sweep_viewer/sapiens/pretrain
    conda activate sapiens_final
    python visualize_bev_trajectories.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from matplotlib.animation import FuncAnimation
import torch

# ── Configuração ──────────────────────────────────────────────────────────────
WAYMO_ROOT   = '/home/lcad/lidar_sweep_viewer/waymo_10'
SCENE_ID     = '58d5f1b9e6a1a2f7'
CHECKPOINT   = 'work_dirs/baseline_overfit_norm/epoch_200.pth'
HISTORY_LEN  = 5
PRED_LEN     = 5
VOXEL_RES    = 2.0
SPATIAL_RANGE = [-10, 10, -10, 10, -2, 4]

# Área de visualização em coordenadas do sensor (metros)
VIEW_RANGE   = 30   # ±30m ao redor do ego-veículo

OUT_IMAGE    = 'bev_trajectories.png'
OUT_VIDEO    = 'bev_trajectories.mp4'


# ── Funções auxiliares ────────────────────────────────────────────────────────

def load_pose(frame_idx):
    path = os.path.join(WAYMO_ROOT, 'poses', SCENE_ID, f'{frame_idx}.txt')
    with open(path) as f:
        rows = [list(map(float, l.split())) for l in f if len(l.split()) == 4]
    return np.array(rows) if len(rows) == 4 else np.eye(4)


def load_lidar(frame_idx):
    path = os.path.join(WAYMO_ROOT, 'bin_files', SCENE_ID, f'{frame_idx}.bin')
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return pts[:, :3]   # x, y, z


def global_to_sensor(pts_global, pose):
    """Transforma pontos de coordenadas globais para sensor frame."""
    inv_pose = np.linalg.inv(pose)
    hom = np.hstack([pts_global, np.ones((len(pts_global), 1))])
    return (inv_pose @ hom.T).T[:, :3]


def load_bbox_sensor(frame_idx, obj_id, pose):
    """Carrega 8 vértices da bbox e transforma para sensor frame."""
    path = os.path.join(WAYMO_ROOT, 'objs_bbox', SCENE_ID,
                        str(frame_idx), f'{obj_id}.txt')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        verts = [list(map(float, l.split())) for l in f if len(l.split()) == 3]
    if len(verts) != 8:
        return None
    verts_global = np.array(verts)
    return global_to_sensor(verts_global, pose)   # (8, 3)


def bbox_to_bev_corners(verts_sensor):
    """
    Projeta bbox 3D para BEV — retorna 4 cantos (x, y) da base inferior.
    Os 4 primeiros vértices são a face inferior da bbox do Waymo.
    """
    base = verts_sensor[:4, :2]   # (4, 2) — face inferior
    return base


def render_lidar_density(pts_sensor, ax, view_range=VIEW_RANGE):
    """
    Renderiza LiDAR como mapa de densidade 2D estilo professional BEV.
    Fundo preto, pontos mais densos = mais brilhantes.
    """
    # Filtrar por range
    mask = (
        (np.abs(pts_sensor[:, 0]) < view_range) &
        (np.abs(pts_sensor[:, 1]) < view_range) &
        (pts_sensor[:, 2] > -3) & (pts_sensor[:, 2] < 5)
    )
    pts = pts_sensor[mask]
    if len(pts) == 0:
        return

    # Altura → cor (baixo=azul, alto=amarelo, estilo KITTI)
    z_norm = np.clip((pts[:, 2] + 2) / 5.0, 0, 1)

    # Tamanho do ponto proporcional à densidade visual
    ax.scatter(
        pts[:, 0], pts[:, 1],
        c=z_norm,
        cmap='plasma',
        s=0.3,
        alpha=0.6,
        linewidths=0,
        rasterized=True,
    )


def draw_bbox_bev(ax, corners, color='#00FFFF', linewidth=1.5):
    """Desenha bounding box BEV como quadrilátero."""
    # Fechar o polígono
    poly = plt.Polygon(
        corners,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        linestyle='-',
        zorder=3,
    )
    ax.add_patch(poly)
    # Seta indicando frente do objeto (aresta 0→1)
    mid = (corners[0] + corners[1]) / 2
    direction = corners[1] - corners[0]
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm * 1.5
        ax.annotate('', xy=mid + direction, xytext=mid,
                    arrowprops=dict(arrowstyle='->', color=color,
                                   lw=1.5), zorder=4)


def load_all_object_tracks():
    """Carrega trajetórias de todos os objetos na cena."""
    bbox_root = os.path.join(WAYMO_ROOT, 'objs_bbox', SCENE_ID)
    pose_root = os.path.join(WAYMO_ROOT, 'poses', SCENE_ID)

    frame_dirs = sorted([
        d for d in os.listdir(bbox_root)
        if os.path.isdir(os.path.join(bbox_root, d)) and d.isdigit()
    ], key=int)

    tracks = {}   # obj_id → [(frame_idx, center_sensor)]
    poses  = {}   # frame_idx → pose matrix

    for frame_dir in frame_dirs:
        frame_idx = int(frame_dir)
        pose = load_pose(frame_idx)
        poses[frame_idx] = pose

        frame_path = os.path.join(bbox_root, frame_dir)
        for obj_file in sorted(os.listdir(frame_path)):
            if not obj_file.endswith('.txt'):
                continue
            obj_id = obj_file.replace('.txt', '')
            path = os.path.join(frame_path, obj_file)
            with open(path) as f:
                verts = [list(map(float, l.split()))
                         for l in f if len(l.split()) == 3]
            if len(verts) != 8:
                continue
            center_global = np.mean(verts, axis=0)
            center_hom = np.append(center_global, 1.0)
            center_sensor = (np.linalg.inv(pose) @ center_hom)[:3]
            tracks.setdefault(obj_id, []).append((frame_idx, center_sensor))

    # Ordenar por frame e filtrar objetos com frames suficientes
    valid_tracks = {}
    for obj_id, track in tracks.items():
        track.sort(key=lambda x: x[0])
        if len(track) >= HISTORY_LEN + PRED_LEN:
            valid_tracks[obj_id] = track

    return valid_tracks, poses


def predict_trajectories(valid_tracks):
    """Usa o modelo baseline para predizer trajetórias."""
    from mmpretrain.models.trajectory_pred.baseline_model import BaselineTrajectoryModel
    from mmengine.runner import load_checkpoint

    model = BaselineTrajectoryModel(history_len=HISTORY_LEN, pred_len=PRED_LEN)
    load_checkpoint(model, CHECKPOINT, map_location='cpu')
    model.eval()

    predictions = {}   # obj_id → {'history', 'future_gt', 'future_pred'}

    for obj_id, track in valid_tracks.items():
        centers = np.array([c for _, c in track[:HISTORY_LEN + PRED_LEN]])

        # Normalizar usando só o histórico (sem data leakage)
        ref = centers[0]
        relative = centers - ref
        hist_rel  = relative[:HISTORY_LEN]
        mean_rel  = hist_rel.mean(axis=0)
        std_rel   = hist_rel.std(axis=0) + 1e-6
        relative_norm = (relative - mean_rel) / std_rel

        hist_norm   = relative_norm[:HISTORY_LEN]
        future_norm = relative_norm[HISTORY_LEN:HISTORY_LEN + PRED_LEN]

        hist_flat = torch.tensor(hist_norm.reshape(-1), dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred_flat = model(hist_flat, mode='predict')

        pred_norm = pred_flat.squeeze(0).view(PRED_LEN, 3).numpy()

        # Desnormalizar
        hist_world   = hist_norm   * std_rel + mean_rel + ref
        future_world = future_norm * std_rel + mean_rel + ref
        pred_world   = pred_norm   * std_rel + mean_rel + ref

        predictions[obj_id] = {
            'history':    hist_world,
            'future_gt':  future_world,
            'future_pred': pred_world,
        }

    return predictions


# ── Figura principal — imagem estática ───────────────────────────────────────

def render_static(valid_tracks, predictions, poses):
    """Gera imagem BEV completa com todos os objetos e trajetórias."""
    frame_idx = HISTORY_LEN - 1   # frame atual = último frame do histórico

    # Carregar LiDAR no sensor frame
    pose = poses.get(frame_idx, np.eye(4))
    pts_sensor = load_lidar(frame_idx)

    fig, ax = plt.subplots(figsize=(12, 12), facecolor='#0a0a0a')
    ax.set_facecolor('#0a0a0a')

    # 1. Renderizar LiDAR
    render_lidar_density(pts_sensor, ax, VIEW_RANGE)

    # 2. Bounding boxes no frame atual
    bbox_root = os.path.join(WAYMO_ROOT, 'objs_bbox', SCENE_ID, str(frame_idx))
    for obj_file in os.listdir(bbox_root):
        if not obj_file.endswith('.txt'):
            continue
        obj_id = obj_file.replace('.txt', '')
        verts_sensor = load_bbox_sensor(frame_idx, obj_id, pose)
        if verts_sensor is None:
            continue
        corners = bbox_to_bev_corners(verts_sensor)
        # Objetos com predição em ciano, outros em cinza
        color = '#00FFFF' if obj_id in predictions else '#555555'
        draw_bbox_bev(ax, corners, color=color)

    # 3. Trajetórias
    for obj_id, pred in predictions.items():
        hist  = pred['history']
        gt    = pred['future_gt']
        preds = pred['future_pred']

        # Histórico — linha tracejada branca
        ax.plot(hist[:, 0], hist[:, 1],
                '--', color='#AAAAAA', linewidth=1.5,
                alpha=0.7, zorder=5)

        # Futuro real — verde
        pts_gt = np.vstack([hist[-1:], gt])
        ax.plot(pts_gt[:, 0], pts_gt[:, 1],
                '-o', color='#00FF88', linewidth=2.5,
                markersize=5, zorder=6, label='Real' if obj_id == list(predictions.keys())[0] else '')

        # Futuro predito — vermelho/laranja
        pts_pred = np.vstack([hist[-1:], preds])
        ax.plot(pts_pred[:, 0], pts_pred[:, 1],
                '-o', color='#FF4444', linewidth=2.5,
                markersize=5, zorder=6, label='Predito' if obj_id == list(predictions.keys())[0] else '')

    # Ego-veículo no centro
    ax.scatter([0], [0], c='white', s=80, marker='*', zorder=10, label='Ego')
    ego_box = plt.Rectangle((-2, -1), 4, 2, color='white',
                              fill=False, linewidth=2, zorder=9)
    ax.add_patch(ego_box)

    # Estilo
    ax.set_xlim(-VIEW_RANGE, VIEW_RANGE)
    ax.set_ylim(-VIEW_RANGE, VIEW_RANGE)
    ax.set_xlabel('X (m)', color='#AAAAAA', fontsize=11)
    ax.set_ylabel('Y (m)', color='#AAAAAA', fontsize=11)
    ax.tick_params(colors='#AAAAAA')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')

    legend = ax.legend(
        loc='upper right', framealpha=0.3,
        facecolor='#1a1a1a', edgecolor='#444444',
        labelcolor='white', fontsize=10
    )

    ax.set_title(
        f'MOTF — Bird\'s Eye View | Cena: {SCENE_ID[:8]}... | '
        f'Frame {frame_idx} | {len(predictions)} objetos',
        color='white', fontsize=13, pad=12
    )

    # Grid sutil
    ax.grid(True, linestyle='--', alpha=0.15, color='#AAAAAA')

    plt.tight_layout()
    plt.savefig(OUT_IMAGE, dpi=150, bbox_inches='tight',
                facecolor='#0a0a0a')
    print(f'✅ Imagem salva: {OUT_IMAGE}')
    plt.close()


# ── Vídeo — evolução frame a frame ───────────────────────────────────────────

def render_video(valid_tracks, predictions, poses):
    """Gera vídeo mostrando evolução temporal das trajetórias."""
    total_frames = HISTORY_LEN + PRED_LEN   # 10 frames

    fig, ax = plt.subplots(figsize=(10, 10), facecolor='#0a0a0a')

    def draw_frame(t):
        ax.clear()
        ax.set_facecolor('#0a0a0a')

        # Frame real de LiDAR disponível
        lidar_frame = min(t, HISTORY_LEN - 1)
        pose = poses.get(lidar_frame, np.eye(4))
        pts_sensor = load_lidar(lidar_frame)
        render_lidar_density(pts_sensor, ax, VIEW_RANGE)

        # Bboxes do frame atual (se disponível)
        bbox_frame_dir = os.path.join(
            WAYMO_ROOT, 'objs_bbox', SCENE_ID, str(lidar_frame)
        )
        if os.path.isdir(bbox_frame_dir):
            for obj_file in os.listdir(bbox_frame_dir):
                if not obj_file.endswith('.txt'):
                    continue
                obj_id = obj_file.replace('.txt', '')
                verts_sensor = load_bbox_sensor(lidar_frame, obj_id, pose)
                if verts_sensor is None:
                    continue
                corners = bbox_to_bev_corners(verts_sensor)
                color = '#00FFFF' if obj_id in predictions else '#555555'
                draw_bbox_bev(ax, corners, color=color)

        # Trajetórias até o frame atual
        for obj_id, pred in predictions.items():
            hist  = pred['history']
            gt    = pred['future_gt']
            preds = pred['future_pred']

            if t < HISTORY_LEN:
                # Mostrando histórico
                ax.plot(hist[:t+1, 0], hist[:t+1, 1],
                        '--', color='#AAAAAA', linewidth=1.5, alpha=0.7)
            else:
                # Mostrando futuro
                future_t = t - HISTORY_LEN + 1
                ax.plot(hist[:, 0], hist[:, 1],
                        '--', color='#AAAAAA', linewidth=1.2, alpha=0.5)

                pts_gt = np.vstack([hist[-1:], gt[:future_t]])
                ax.plot(pts_gt[:, 0], pts_gt[:, 1],
                        '-o', color='#00FF88', linewidth=2.5, markersize=5,
                        label='Real' if obj_id == list(predictions.keys())[0] else '')

                pts_pred = np.vstack([hist[-1:], preds[:future_t]])
                ax.plot(pts_pred[:, 0], pts_pred[:, 1],
                        '-o', color='#FF4444', linewidth=2.5, markersize=5,
                        label='Predito' if obj_id == list(predictions.keys())[0] else '')

        # Ego-veículo
        ax.scatter([0], [0], c='white', s=80, marker='*', zorder=10)
        ego_box = plt.Rectangle((-2, -1), 4, 2, color='white',
                                  fill=False, linewidth=2, zorder=9)
        ax.add_patch(ego_box)

        # Estilo
        ax.set_xlim(-VIEW_RANGE, VIEW_RANGE)
        ax.set_ylim(-VIEW_RANGE, VIEW_RANGE)
        ax.set_xlabel('X (m)', color='#AAAAAA')
        ax.set_ylabel('Y (m)', color='#AAAAAA')
        ax.tick_params(colors='#AAAAAA')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')
        ax.grid(True, linestyle='--', alpha=0.15, color='#AAAAAA')

        phase = 'Histórico' if t < HISTORY_LEN else 'Predição'
        ax.set_title(
            f'MOTF — {phase} | Frame {t+1}/{total_frames}',
            color='white', fontsize=13
        )

        if t >= HISTORY_LEN:
            ax.legend(loc='upper right', framealpha=0.3,
                     facecolor='#1a1a1a', edgecolor='#444444',
                     labelcolor='white', fontsize=9)

    ani = FuncAnimation(
        fig, draw_frame,
        frames=total_frames,
        interval=800,
        repeat=False,
    )

    ani.save(OUT_VIDEO, writer='ffmpeg', fps=2, dpi=120,
             savefig_kwargs={'facecolor': '#0a0a0a'})
    print(f'✅ Vídeo salvo: {OUT_VIDEO}')
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('🔍 Carregando trajetórias...')
    valid_tracks, poses = load_all_object_tracks()
    print(f'   {len(valid_tracks)} objetos com trajetória completa')

    print('🤖 Executando predições...')
    predictions = predict_trajectories(valid_tracks)
    print(f'   {len(predictions)} objetos preditos')

    print('🖼️  Gerando imagem estática...')
    render_static(valid_tracks, predictions, poses)

    print('🎬 Gerando vídeo...')
    render_video(valid_tracks, predictions, poses)

    print('\n✅ Pronto!')
    print(f'   Imagem: {OUT_IMAGE}')
    print(f'   Vídeo:  {OUT_VIDEO}')
