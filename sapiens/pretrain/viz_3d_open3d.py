"""
viz_3d_open3d.py  —  PASO 2 de la visualización 3D (estilo show_point_cloud)
Corre en lidar_mae (tem open3d). Lê o .npz do passo 1 + as nuvens de pontos e
bboxes do waymo_10 e monta a cena 3D:
  - nuvem de pontos LiDAR real (colorida por altura)
  - carros como bounding boxes 3D wireframe (ciano = tem predição)
  - trajetória histórica (branca), futura real (verde) e predita (vermelha)

Uso:
    conda activate lidar_mae
    python viz_3d_open3d.py attn                 # render PNG offscreen (default)
    python viz_3d_open3d.py attn --interactive   # abre janela 3D interativa
"""
import os
import sys
import numpy as np
import open3d as o3d

WAYMO_ROOT = '/home/lcad/lidar_sweep_viewer/waymo_10'

MODEL_TYPE   = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else 'attn'
INTERACTIVE  = '--interactive' in sys.argv
NPZ_PATH     = f'viz_predictions_{MODEL_TYPE}.npz'
OUT_PNG      = f'viz_3d_{MODEL_TYPE}.png'

# arestas de um cubo a partir de 8 vértices (mesma convenção do show_point_cloud)
BOX_LINES = [
    [0, 1], [1, 2], [2, 3], [3, 0],   # base inferior
    [4, 5], [5, 6], [6, 7], [7, 4],   # base superior
    [0, 4], [1, 5], [2, 6], [3, 7],   # verticais
]


def load_pose(scene, frame):
    path = os.path.join(WAYMO_ROOT, 'poses', scene, f'{frame}.txt')
    with open(path) as f:
        rows = [list(map(float, l.split())) for l in f if len(l.split()) == 4]
    return np.array(rows) if len(rows) == 4 else np.eye(4)


def load_lidar(scene, frame):
    path = os.path.join(WAYMO_ROOT, 'bin_files', scene, f'{frame}.bin')
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]


def global_to_sensor(verts_global, pose):
    inv = np.linalg.inv(pose)
    hom = np.hstack([verts_global, np.ones((len(verts_global), 1))])
    return (inv @ hom.T).T[:, :3]


def load_bbox_sensor(scene, frame, obj_id, pose):
    path = os.path.join(WAYMO_ROOT, 'objs_bbox', scene, str(frame), f'{obj_id}.txt')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        verts = [list(map(float, l.split())) for l in f if len(l.split()) == 3]
    if len(verts) != 8:
        return None
    return global_to_sensor(np.array(verts), pose)


def make_box(verts, color):
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(verts)
    ls.lines = o3d.utility.Vector2iVector(BOX_LINES)
    ls.colors = o3d.utility.Vector3dVector([color] * len(BOX_LINES))
    return ls


def make_polyline(pts, color):
    """Linha 3D conectando uma sequência de pontos."""
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(pts) - 1)])
    ls.colors = o3d.utility.Vector3dVector([color] * (len(pts) - 1))
    return ls


def make_spheres(pts, color, radius=0.25):
    """Marcadores esféricos nos pontos da trajetória."""
    meshes = []
    for p in pts:
        s = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=6)
        s.translate(p)
        s.paint_uniform_color(color)
        meshes.append(s)
    return meshes


def height_colormap(z):
    """Cor por altura estilo KITTI (baixo escuro -> alto claro)."""
    zn = np.clip((z - z.min()) / (np.ptp(z) + 1e-6), 0, 1)
    # plasma-ish: roxo -> laranja -> amarelo
    r = np.clip(1.5 * zn, 0, 1)
    g = np.clip(1.5 * zn - 0.3, 0, 1)
    b = np.clip(0.6 - zn, 0, 1)
    return np.stack([r, g, b], axis=1)


def main():
    data = np.load(NPZ_PATH, allow_pickle=True)
    scene = str(data['scene_id'])
    frame = int(data['display_frame'])
    obj_ids = data['obj_ids']
    history = data['history']
    future_gt = data['future_gt']
    future_pred = data['future_pred']
    pred_set = set(obj_ids.tolist())
    print(f'Cena {scene} | frame {frame} | {len(obj_ids)} objetos preditos | modelo {MODEL_TYPE}')

    geoms = []

    # 1. Nuvem de pontos
    pts = load_lidar(scene, frame)
    rng = np.linalg.norm(pts[:, :2], axis=1) < 60
    pts = pts[rng]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(height_colormap(pts[:, 2]))
    geoms.append(pcd)

    # 2. Bounding boxes 3D de TODOS os objetos no frame (ciano = tem predição)
    pose = load_pose(scene, frame)
    bbox_dir = os.path.join(WAYMO_ROOT, 'objs_bbox', scene, str(frame))
    n_boxes = 0
    if os.path.isdir(bbox_dir):
        for f in sorted(os.listdir(bbox_dir)):
            if not f.endswith('.txt'):
                continue
            oid = f[:-4]
            verts = load_bbox_sensor(scene, frame, oid, pose)
            if verts is None:
                continue
            color = [0.0, 1.0, 1.0] if oid in pred_set else [0.4, 0.4, 0.4]
            geoms.append(make_box(verts, color))
            n_boxes += 1

    # 3. Trajetórias: histórico (branco), real (verde), predita (vermelha)
    for i in range(len(obj_ids)):
        h, gt, pr = history[i], future_gt[i], future_pred[i]
        geoms.append(make_polyline(h, [0.7, 0.7, 0.7]))            # histórico
        geoms.append(make_polyline(np.vstack([h[-1:], gt]), [0.0, 1.0, 0.2]))   # real
        geoms.append(make_polyline(np.vstack([h[-1:], pr]), [1.0, 0.1, 0.1]))   # predito
        geoms += make_spheres(gt, [0.0, 1.0, 0.2])
        geoms += make_spheres(pr, [1.0, 0.1, 0.1])

    print(f'{n_boxes} bounding boxes 3D desenhadas')

    if INTERACTIVE:
        o3d.visualization.draw_geometries(
            geoms, window_name=f'MOTF 3D — {MODEL_TYPE}',
            width=1280, height=800)
        return

    # Centro da ação = média dos pontos das trajetórias (não o ego)
    all_traj = np.concatenate([history.reshape(-1, 3),
                               future_gt.reshape(-1, 3),
                               future_pred.reshape(-1, 3)], axis=0)
    lookat = all_traj.mean(axis=0)

    # Render offscreen -> PNG (perspectiva 3D inclinada ~45°, aproximada)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1600, height=1200)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.03, 0.03, 0.06])
    opt.point_size = 2.2
    ctr = vis.get_view_control()
    ctr.set_front([0.45, -0.45, -0.77])   # perspectiva 3D inclinada
    ctr.set_up([0.0, 0.0, 1.0])
    ctr.set_lookat(lookat)
    ctr.set_zoom(0.18)                     # mais perto que antes (0.35)
    vis.poll_events(); vis.update_renderer()
    vis.capture_screen_image(OUT_PNG, do_render=True)
    vis.destroy_window()
    print(f'✅ Imagem 3D salva: {OUT_PNG}')


if __name__ == '__main__':
    main()
