"""
viz_dashboard_cpp_style.py
Replica o estilo do show_point_cloud.cpp (BEV cenital: pontos brancos sobre
preto, estilo OpenCV) MAS sobrepõe as predições do nosso modelo:
  - nuvem LiDAR em BEV (branco)
  - bounding boxes dos carros (ciano)
  - trajetória histórica (cinza), futura real (verde) e predita (vermelha)

Lê o viz_predictions_<tipo>.npz exportado por export_predictions_npz.py.
Roda em sapiens_final (tem cv2). Gera PNG por frame + (opcional) zoom.

Uso:
    conda activate sapiens_final
    python viz_dashboard_cpp_style.py attn        # default
    python viz_dashboard_cpp_style.py baseline
"""
import os
import sys
import numpy as np
import cv2

WAYMO_ROOT = '/home/lcad/lidar_sweep_viewer/waymo_10'
# Mesmos parâmetros do show_point_cloud.cpp
METERS = 51.2
SIZE   = 1024
SCALE  = SIZE / (2 * METERS)

MODEL_TYPE = sys.argv[1] if len(sys.argv) > 1 else 'attn'
NPZ_PATH   = f'viz_predictions_{MODEL_TYPE}.npz'
OUT_FULL   = f'dashboard_{MODEL_TYPE}_full.png'
OUT_ZOOM   = f'dashboard_{MODEL_TYPE}_zoom.png'


def to_px(x, y):
    """Sensor (x,y) -> pixel (col,row), igual ao calculate_birdview do .cpp."""
    px = int((x + METERS) * SCALE)
    py = int((y + METERS) * SCALE)
    return px, SIZE - 1 - py


def load_pose(scene, frame):
    path = os.path.join(WAYMO_ROOT, 'poses', scene, f'{frame}.txt')
    with open(path) as f:
        rows = [list(map(float, l.split())) for l in f if len(l.split()) == 4]
    return np.array(rows) if len(rows) == 4 else np.eye(4)


def load_lidar(scene, frame):
    path = os.path.join(WAYMO_ROOT, 'bin_files', scene, f'{frame}.bin')
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]


def load_bbox_base(scene, frame, obj_id, pose):
    """Retorna os 4 cantos da base da bbox em sensor frame (x,y)."""
    path = os.path.join(WAYMO_ROOT, 'objs_bbox', scene, str(frame), f'{obj_id}.txt')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        verts = [list(map(float, l.split())) for l in f if len(l.split()) == 3]
    if len(verts) != 8:
        return None
    verts = np.array(verts)
    inv = np.linalg.inv(pose)
    hom = np.hstack([verts, np.ones((8, 1))])
    sensor = (inv @ hom.T).T[:, :3]
    return sensor[:4, :2]   # base inferior (x,y)


def draw_polyline(img, pts_xy, color, thickness=2, dot=True):
    px = [to_px(p[0], p[1]) for p in pts_xy]
    for a, b in zip(px[:-1], px[1:]):
        cv2.line(img, a, b, color, thickness, cv2.LINE_AA)
    if dot:
        for p in px:
            cv2.circle(img, p, 3, color, -1, cv2.LINE_AA)


def main():
    data = np.load(NPZ_PATH, allow_pickle=True)
    scene = str(data['scene_id'])
    frame = int(data['display_frame'])
    obj_ids = data['obj_ids']
    history, future_gt, future_pred = data['history'], data['future_gt'], data['future_pred']
    pred_set = set(obj_ids.tolist())
    print(f'Cena {scene} | frame {frame} | {len(obj_ids)} objetos | modelo {MODEL_TYPE}')

    # ── BEV: nuvem de pontos (branco sobre preto, estilo .cpp) ──
    img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    pts = load_lidar(scene, frame)
    for x, y in pts[:, :2]:
        if -METERS < x < METERS and -METERS < y < METERS:
            px, py = to_px(x, y)
            img[py, px] = (255, 255, 255)

    # ── Bounding boxes (ciano = tem predição, cinza = resto) ──
    pose = load_pose(scene, frame)
    bbox_dir = os.path.join(WAYMO_ROOT, 'objs_bbox', scene, str(frame))
    if os.path.isdir(bbox_dir):
        for f in sorted(os.listdir(bbox_dir)):
            if not f.endswith('.txt'):
                continue
            oid = f[:-4]
            base = load_bbox_base(scene, frame, oid, pose)
            if base is None:
                continue
            poly = np.array([to_px(x, y) for x, y in base], dtype=np.int32)
            color = (255, 255, 0) if oid in pred_set else (90, 90, 90)
            cv2.polylines(img, [poly], True, color, 2, cv2.LINE_AA)

    # ── Trajetórias: histórico cinza, real verde, predita vermelha ──
    for i in range(len(obj_ids)):
        h, gt, pr = history[i], future_gt[i], future_pred[i]
        draw_polyline(img, h, (160, 160, 160), 1, dot=False)
        draw_polyline(img, np.vstack([h[-1:], gt]), (0, 255, 0), 2)     # real (BGR verde)
        draw_polyline(img, np.vstack([h[-1:], pr]), (0, 0, 255), 2)     # predito (BGR vermelho)

    # Ego no centro
    cx, cy = to_px(0, 0)
    cv2.drawMarker(img, (cx, cy), (255, 255, 255), cv2.MARKER_STAR, 16, 2)

    # Legenda
    cv2.putText(img, f'MOTF {MODEL_TYPE} | verde=real  vermelho=predito  ciano=bbox',
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(OUT_FULL, img)
    print(f'✅ {OUT_FULL}')

    # ── Zoom: recorte centrado na média dos objetos ──
    all_xy = np.concatenate([history.reshape(-1, 3)[:, :2],
                             future_gt.reshape(-1, 3)[:, :2],
                             future_pred.reshape(-1, 3)[:, :2]], axis=0)
    mcx, mcy = to_px(all_xy[:, 0].mean(), all_xy[:, 1].mean())
    half = 320
    x0, y0 = max(0, mcx - half), max(0, mcy - half)
    x1, y1 = min(SIZE, mcx + half), min(SIZE, mcy + half)
    crop = img[y0:y1, x0:x1]
    crop = cv2.resize(crop, (900, 900), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(OUT_ZOOM, crop)
    print(f'✅ {OUT_ZOOM}')


if __name__ == '__main__':
    main()
