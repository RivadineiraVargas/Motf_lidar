"""
export_decoder_mini_global.py — Lleva las predicciones del decoder mini al
"simulador": (1) exporta predictions_global.txt para el viewer C++
(show_point_cloud, dashboard rangeview+BEV, tecla 't'); (2) genera un video
GIF por escena con range-view arriba y BEV abajo (puntos LiDAR + GT verde +
pred rojo), un frame por sweep t=0..10.

Formato txt (igual a export_predictions_global.py):
    <scene> <obj_id> <kind> <t> <x> <y> <z>   kind: 0=hist 1=GT 2=pred (GLOBAL)

Uso:
  conda run -n sapiens_gpu python export_decoder_mini_global.py \
      [--scenes 2a81f5233075e987 82f90331a1dfe968] [--out work_dirs/decoder_mini]
"""
import argparse, glob, math, os
import numpy as np
import torch
import cv2
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpretrain.registry import MODELS
import train_decoder_mini as tdm
from train_decoder_mini import (ROOT, CFG, CKPT, MAXR, SCALE, WP_STEP, N_WP,
                                MiniWayformerDecoder, MiniBaseline,
                                build_sample, encode_sweeps, center_of)

BEV_M = 75.0        # semiancho del BEV (m)
BEV_PX = 900        # tamaño del BEV en px

# --- Proyección de bboxes a la range view (método de Gabriel, adaptado) ---
# CALIBRADO contra los puntos LiDAR de los .bin (ver docs/CHECKLIST_CLAUDINE.md):
#   - azimut ESPEJADO (la fórmula comentada de Gabriel): u = W*(-yaw+pi)/2pi
#   - inclinaciones por fila NO uniformes, estimadas de los datos y guardadas
#     en waymo_clean/beam_inclinations.npy (fila 0 = +0.9deg ... fila 63 = -14.8deg)
#   - error mediano de proyección validado: 0.75-1.5 m en 3 escenas
# CALIBRABLES por si hace falta re-ajustar:
LIDAR_Z_OFFSET = 2.0      # altura del sensor sobre el frame de la pose (m)
AZIMUT_OFFSET_PX = 0      # corrimiento horizontal (px sobre 2650)
BEAM_INCL = np.load(f'{ROOT}/beam_inclinations.npy')   # (64,) rad, decreciente
BEAM_INCL_MIN = math.radians(-17.6)   # fallback lineal si no hay tabla
BEAM_INCL_MAX = math.radians(2.4)


def proyectar_a_rangeview(verts_sensor, u_size, v_size):
    """Proyección yaw->u (espejada) y pitch->fila por tabla de beams."""
    filas = v_size / 64.0            # la franja puede venir re-escalada de 64
    pts = []
    for x, y, z in verts_sensor:
        yaw = -np.arctan2(y, x)      # ESPEJO (convención de columnas WOMD)
        u = (int(u_size * ((yaw + np.pi) / (2.0 * np.pi))) + AZIMUT_OFFSET_PX) % u_size
        zz = z - LIDAR_Z_OFFSET
        pitch = np.arctan2(zz, np.sqrt(x * x + y * y))
        fila = int(np.abs(pitch - BEAM_INCL).argmin())
        v = int((fila + 0.5) * filas)
        if 0 <= u < u_size and 0 <= v < v_size:
            pts.append((u, v))
    return pts


def dibujar_bbox_rangeview(rv_color, verts_sensor):
    """Wireframe verde (12 aristas) como draw_bounding_box3d_range_view."""
    cx, cy = verts_sensor[0][:2]
    if np.linalg.norm([cx, cy]) < 7.0:           # ignorar el ego
        return
    u_size = rv_color.shape[1]
    p = proyectar_a_rangeview(verts_sensor, u_size, rv_color.shape[0])
    if len(p) < 8:                                # proyección incompleta
        return

    def linea(a, b):
        # evitar aristas que cruzan la costura del azimut (+-180 grados):
        # dibujarian una linea a lo ancho de toda la imagen (bug de Gabriel)
        if abs(a[0] - b[0]) > u_size / 2:
            return
        cv2.line(rv_color, a, b, (0, 255, 0), 2)

    for i in range(4):
        linea(p[i], p[(i + 1) % 4])
        linea(p[i + 4], p[((i + 1) % 4) + 4])
        linea(p[i], p[i + 4])


def load_pose(scene, t):
    return np.loadtxt(f'{ROOT}/poses/{scene}/{t}.txt')


def bev_xy(p):
    s = BEV_PX / (2 * BEV_M)
    # clamp: coords enormes (objetos/preds fuera de rango) desbordan los int
    # de OpenCV y dibujan bandas sólidas
    px = max(-BEV_PX, min(2 * BEV_PX, (p[0] + BEV_M) * s))
    py = max(-BEV_PX, min(2 * BEV_PX, (BEV_M - p[1]) * s))
    return int(px), int(py)


def draw_poly(img, pts, color, thick=2):
    pts = [bev_xy(p) for p in pts]
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(img, a, b, color, thick)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', nargs='+',
                    default=['2a81f5233075e987', '82f90331a1dfe968'])
    ap.add_argument('--out', default='work_dirs/decoder_mini')
    ap.add_argument('--txt', default='/home/lcad/lidar_sweep_viewer/predictions_global.txt')
    ap.add_argument('--strip', choices=['compacta', 'rayas'], default='compacta')
    ap.add_argument('--sin-gif', action='store_true',
                    help='solo exporta el txt (rapido, p/ viewer C++)')
    # --- modelo (default = el pipeline viejo: encoder 10sw + decoder 8s) ---
    ap.add_argument('--enc-cfg', default=CFG,
                    help='config del MAE (p.ej. .../config_rangeview_rect_fold0.py)')
    ap.add_argument('--enc-ckpt', default=CKPT,
                    help='checkpoint del MAE (p.ej. work_dirs/rv_rect_fold0/epoch_1000.pth)')
    ap.add_argument('--dec', default=None,
                    help='checkpoint del decoder wayformer (default: <out>/decoder_mini.pth)')
    ap.add_argument('--dec-baseline', default=None,
                    help='checkpoint del baseline sin escena (default: <out>/decoder_mini_baseline.pth)')
    ap.add_argument('--n-wp', type=int, default=None,
                    help='waypoints del horizonte: 2/6/10/16 = 1s/3s/5s/8s. '
                         'DEBE coincidir con el horizonte del decoder entrenado.')
    args = ap.parse_args()
    dev = 'cuda'

    # El horizonte se fija ANTES de construir samples y modelos: build_sample y
    # head_traj leen el global N_WP del modulo (mismo mecanismo que train_decoder).
    if args.n_wp is not None:
        tdm.N_WP = args.n_wp
    print(f'[cfg] encoder {args.enc_ckpt}')
    print(f'[cfg] horizonte {tdm.N_WP} wp = {tdm.N_WP * 0.5:.1f}s')

    init_default_scope('mmpretrain')
    cfg = Config.fromfile(args.enc_cfg)
    mae = MODELS.build({**cfg.model, 'data_preprocessor': cfg.data_preprocessor})
    mae.load_state_dict(torch.load(args.enc_ckpt, map_location='cpu').get('state_dict'),
                        strict=False)
    encoder = mae.backbone.to(dev)
    encoder.eval()
    model = MiniWayformerDecoder().to(dev)
    # strict=False: checkpoints previos a t_emb (que inicia en 0 = sin efecto)
    dec_path = args.dec or f'{args.out}/decoder_mini.pth'
    print(f'[cfg] decoder {dec_path}')
    model.load_state_dict(torch.load(dec_path, map_location=dev), strict=False)
    model.eval()
    base = None
    bpath = args.dec_baseline or f'{args.out}/decoder_mini_baseline.pth'
    if os.path.exists(bpath):
        base = MiniBaseline().to(dev)
        base.load_state_dict(torch.load(bpath, map_location=dev), strict=False)
        base.eval()

    lines = []
    for scene in args.scenes:
        lat = encode_sweeps(encoder, scene, range(11), dev)
        frames = []
        for t in range(11):
            s = build_sample(scene, t)
            if s['n'] == 0:
                continue
            with torch.no_grad():
                traj, vlog = model(lat[t], (s['feat'] / SCALE).to(dev), s['n'])
                pred_b = None
                if base is not None:
                    traj_b, _ = base(lat[t], (s['feat'] / SCALE).to(dev), s['n'])
                    pred_b = (s['cv'] + traj_b[0, :s['n']].cpu() * SCALE).numpy()
            pred = (s['cv'] + traj[0, :s['n']].cpu() * SCALE).numpy()  # cv + residuo
            validez = (torch.sigmoid(vlog[0, :s['n']]) > 0.5).cpu().numpy()
            pose = load_pose(scene, t)                            # ego -> global
            to_glob = lambda p2: (pose @ np.array([p2[0], p2[1], 0, 1]))[:3]

            # --- txt para el viewer C++, SOLO desde t=10 ---
            # El viewer conecta todos los puntos de un track ordenados por
            # waypoint; exportar los 11 sweeps superponia 11 trayectorias por
            # objeto y dibujaba una "mancha" en vez de una linea.
            if t == 10:
                for i, tid in enumerate(s['ids']):
                    c = s['cur'][i].numpy(); m = s['wpm'][i].numpy() > 0
                    # hist desde labels pasados (t-5k), si existen
                    for k in range(2, 0, -1):
                        fh = f'{ROOT}/objs_bbox/{scene}/{t - k * WP_STEP}/{tid}.txt'
                        if t - k * WP_STEP >= 0 and os.path.exists(fh):
                            g = center_of(fh)
                            lines.append(f'{scene} {tid} 0 {k} {g[0]:.4f} {g[1]:.4f} {g[2]:.4f}')
                    for k in np.where(m)[0]:
                        g = to_glob(c + s['gt'][i].numpy()[k])
                        lines.append(f'{scene} {tid} 1 {k} {g[0]:.4f} {g[1]:.4f} {g[2]:.4f}')
                        g = to_glob(c + pred[i][k])
                        lines.append(f'{scene} {tid} 2 {k} {g[0]:.4f} {g[1]:.4f} {g[2]:.4f}')

            # --- frame del video: rangeview arriba + BEV abajo ---
            if args.sin_gif:
                continue
            r = np.load(f'{ROOT}/range_files/{scene}/{t}.npy')[..., 0]
            u = np.clip(255 * (1 - r / MAXR), 0, 255).astype(np.uint8)
            u[r < 0] = 0
            # base = formato COLEGA (show_rangeview_and_birdview.py): 64 beams
            # -> 1024 filas nearest (16 filas/beam). Dos presentaciones:
            #   rayas    = proporcional 900x348 nearest (fiel al colega)
            #   compacta = 900x256 INTER_AREA (mas legible, default)
            rv = cv2.resize(u, (2650, 1024), interpolation=cv2.INTER_NEAREST)
            rv = cv2.cvtColor(rv, cv2.COLOR_GRAY2BGR)
            # bboxes proyectados a la range view (método Gabriel), en
            # resolución nativa 2650x1024, ANTES del resize de la franja
            inv = np.linalg.inv(pose)
            for fb in sorted(glob.glob(f'{ROOT}/objs_bbox/{scene}/{t}/*.txt')):
                verts = np.loadtxt(fb)                            # (8,3) global
                vh = np.hstack([verts, np.ones((8, 1))])
                verts_sensor = (inv @ vh.T).T[:, :3]
                dibujar_bbox_rangeview(rv, verts_sensor)
            if args.strip == 'rayas':
                h = round(1024 * BEV_PX / 2650)
                rv = cv2.resize(rv, (BEV_PX, h), interpolation=cv2.INTER_NEAREST)
            else:
                rv = cv2.resize(rv, (BEV_PX, 256), interpolation=cv2.INTER_AREA)

            bev = np.zeros((BEV_PX, BEV_PX, 3), np.uint8)
            pts = np.fromfile(f'{ROOT}/bin_files/{scene}/{t}.bin',
                              dtype=np.float32).reshape(-1, 4)[:, :2]
            keep = (np.abs(pts[:, 0]) < BEV_M) & (np.abs(pts[:, 1]) < BEV_M)
            for p in pts[keep][::4]:
                bev[bev_xy(p)[1], bev_xy(p)[0]] = (255, 255, 255)
            for i in range(s['n']):
                c = s['cur'][i].numpy(); m = s['wpm'][i].numpy() > 0
                gt_abs = c + s['gt'][i].numpy()
                pr_abs = c + pred[i]
                # errores explícitos: segmento amarillo pred_k -> GT_k
                for k in np.where(m)[0]:
                    cv2.line(bev, bev_xy(pr_abs[k]), bev_xy(gt_abs[k]),
                             (0, 255, 255), 1)
                if pred_b is not None:                            # baseline naranja
                    draw_poly(bev, [c] + list(c + pred_b[i][m]), (0, 140, 255), 1)
                draw_poly(bev, [c] + list(gt_abs[m]), (0, 255, 0))
                draw_poly(bev, [c] + list(pr_abs[m]), (0, 0, 255))
                # validez predicha: cian lleno = valida; magenta anillo = invalida
                if validez[i]:
                    cv2.circle(bev, bev_xy(c), 4, (255, 255, 0), -1)
                else:
                    cv2.circle(bev, bev_xy(c), 6, (255, 0, 255), 2)
            cv2.putText(bev, f'{scene[:8]} t={t} GT verde/Wayformer rojo/'
                        f'baseline naranja/error amarillo/valida cian',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            frames.append(np.vstack([rv, bev]))

        if args.sin_gif:
            continue
        gif = f'{args.out}/sim_{scene[:8]}.gif'
        from PIL import Image
        pil = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).quantize(64)
               for f in frames]
        # disposal=2 + optimize=False: la paleta optimizada de PIL corrompe
        # los últimos frames (bandas de color falsas)
        pil[0].save(gif, save_all=True, duration=500, loop=0,
                    append_images=pil[1:], disposal=2, optimize=False)
        cv2.imwrite(f'{args.out}/sim_{scene[:8]}_t10.png', frames[-1])
        print(f'[OK] {gif} ({len(frames)} frames) + _t10.png')

    with open(args.txt, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'[OK] {args.txt} ({len(lines)} puntos) — viewer C++: '
          f'./show_point_cloud --input waymo_clean_view  (tecla t)')


if __name__ == '__main__':
    main()
