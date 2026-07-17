"""
rebuild_grid_bins.py — Reconstruye bin_files en formato GRILLA COMPLETA
(contrato del viewer C++: 64x2650 puntos EN ORDEN, 4 floats [x,y,z,rango],
rango=-1 en pixeles sin retorno) a partir de los range_files/*.npy de
waymo_clean, que conservan la grilla entera.

Motivo: la extracción de waymo_clean descartó los sin-retorno (~153k pts
sueltos), rompiendo el reshape(64,2650) con el que show_point_cloud.cpp
construye la vista superior (range image). waymo_10 sí cumple el contrato.

Geometría (calibrada contra los puntos reales, ver beam_inclinations.npy):
  yaw(col)  = pi - 2*pi*(col+0.5)/2650      (azimut ESPEJADO, convención WOMD)
  pitch(row)= beam_inclinations[row]         (no uniforme, +0.9..-14.8 grados)
  z_offset  = 2.0 m (sensor sobre el frame de la pose)

Salida en un root NUEVO (no toca los datos originales):
  waymo_clean_view/bin_files/<scene>/<t>.bin  + symlinks objs_bbox/poses

Uso:  python utilities/rebuild_grid_bins.py [--validar]
"""
import argparse, os
import numpy as np

ROOT = '/home/lcad/lidar_sweep_viewer/waymo_clean'
OUT = '/home/lcad/lidar_sweep_viewer/waymo_clean_view'
W, H = 2650, 64
Z_OFF = 2.0
INCL = np.load(f'{ROOT}/beam_inclinations.npy')          # (64,) rad


def grid_xyz(rng):
    """(64,2650) rangos -> (64*2650, 4) [x,y,z,rango] en frame del sensor."""
    col = np.arange(W)
    yaw = np.pi - 2 * np.pi * (col + 0.5) / W            # (2650,)
    pitch = INCL[:, None]                                 # (64,1)
    hdist = rng * np.cos(pitch)
    x = hdist * np.cos(yaw)[None, :]
    y = hdist * np.sin(yaw)[None, :]
    z = Z_OFF + rng * np.sin(pitch)
    pts = np.stack([x, y, z, rng], axis=-1).reshape(-1, 4).astype(np.float32)
    # sin retorno: rango -1 y xyz lejos de la ventana del BEV (como waymo_10)
    invalid = pts[:, 3] <= 0
    pts[invalid, :3] = 1e6
    pts[invalid, 3] = -1.0
    return pts


def validar(scene, t):
    """Proyecta cada punto ORIGINAL a su celda (fila,col) y compara con el
    xyz reconstruido de esa celda — test directo del mapeo de grilla."""
    rng = np.load(f'{ROOT}/range_files/{scene}/{t}.npy')[..., 0]
    rec = grid_xyz(rng)[:, :3].reshape(H, W, 3)
    orig = np.fromfile(f'{ROOT}/bin_files/{scene}/{t}.bin',
                       np.float32).reshape(-1, 4)[:, :3]
    np.random.seed(0)
    s = orig[np.random.choice(len(orig), 5000, replace=False)]
    x, y, z = s.T
    yaw = np.arctan2(y, x)
    col = ((np.pi - yaw) * W / (2 * np.pi) - 0.5).round().astype(int) % W
    zz = z - Z_OFF
    pitch = np.arctan2(zz, np.hypot(x, y))
    row = np.abs(pitch[:, None] - INCL[None, :]).argmin(1)
    cel = rec[row, col]
    valid = np.load(f'{ROOT}/range_files/{scene}/{t}.npy')[row, col, 0] > 0
    d = np.linalg.norm(cel[valid] - s[valid], axis=1)
    print(f'  {scene[:8]} t={t}: {valid.mean()*100:.0f}% de puntos caen en '
          f'celda valida; err xyz mediana {np.median(d):.3f} m, '
          f'p90 {np.percentile(d, 90):.3f} m')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--validar', action='store_true')
    args = ap.parse_args()

    scenes = sorted(os.listdir(f'{ROOT}/range_files'))
    if args.validar:
        print('validación reconstrucción vs puntos originales:')
        for sc, t in [(scenes[0], 10), (scenes[9], 5), (scenes[12], 3)]:
            validar(sc, t)
        return

    os.makedirs(f'{OUT}/bin_files', exist_ok=True)
    for link in ('objs_bbox', 'poses'):
        dst = f'{OUT}/{link}'
        if not os.path.islink(dst) and not os.path.exists(dst):
            os.symlink(f'{ROOT}/{link}', dst)
    n = 0
    for sc in scenes:
        os.makedirs(f'{OUT}/bin_files/{sc}', exist_ok=True)
        for f in sorted(os.listdir(f'{ROOT}/range_files/{sc}')):
            t = f[:-4]
            rng = np.load(f'{ROOT}/range_files/{sc}/{f}')[..., 0]
            grid_xyz(rng).tofile(f'{OUT}/bin_files/{sc}/{t}.bin')
            n += 1
    print(f'[OK] {n} bins de grilla completa en {OUT}/bin_files '
          f'(+ symlinks objs_bbox/poses)')


if __name__ == '__main__':
    main()
