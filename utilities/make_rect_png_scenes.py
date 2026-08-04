"""
make_rect_png_scenes.py — Genera range-view PNGs (formato rect colega) para
una LISTA de escenas dada, con sweeps seleccionables. Uso general (a
diferencia de make_rect_png_100.py que tiene el split hardcodeado).

Pensado para re-pre-entrenar el encoder MAE SOLO en las escenas de train de
un fold (sin fuga de las escenas retenidas).

Uso:
  python utilities/make_rect_png_scenes.py \
      --out waymo_clean/range_png_rect/fold0_train \
      --sweeps 0 2 4 6 8 \
      --scenes 2e41fe6f 367b072e ...   (ids completos)
"""
import argparse
import os

import cv2
import numpy as np

SRC = '/home/lcad/lidar_sweep_viewer/waymo_clean/range_files'
MAXR = 75.0


def conv(src, dst):
    r = np.load(src)[..., 0]
    u = np.clip(255 * (1 - r / MAXR), 0, 255).astype(np.uint8)
    u[r <= 0] = 0
    img = cv2.resize(u, (2650, 1024), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(dst, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--scenes', nargs='+', required=True)
    ap.add_argument('--sweeps', nargs='+', type=int, default=[0, 2, 4, 6, 8])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    n = 0
    for s in args.scenes:
        for t in args.sweeps:
            src = os.path.join(SRC, s, f'{t}.npy')
            if not os.path.exists(src):
                print(f'  [aviso] falta {src}, salteado')
                continue
            conv(src, os.path.join(args.out, f'sweep_{n:03d}_{s[:8]}_{t}.png'))
            n += 1
    print(f'Generadas {n} PNG en {args.out}')


if __name__ == '__main__':
    main()
