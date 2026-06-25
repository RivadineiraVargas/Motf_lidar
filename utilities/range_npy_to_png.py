"""
range_npy_to_png.py — Convierte nuestras range-images limpias (.npy) a PNG en el
formato del COLEGA (Gabriel): 2650x1024, gris invertido (cerca=brillante), beams
escalados 64->1024 (nearest, da las "rayas"). Para el pipeline MAE de imágenes.

Uso:
  conda activate sapiens_gpu
  python utilities/range_npy_to_png.py --src waymo_clean/range_files \
      --out waymo_clean/range_png/train --max 10
"""
import os, glob, argparse
import numpy as np
import cv2

MAXR = 75.0   # rango máximo (m) para normalizar a 8-bit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='waymo_clean/range_files')
    ap.add_argument('--out', default='waymo_clean/range_png/train')
    ap.add_argument('--max', type=int, default=0, help='nº de sweeps (0=todos)')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    npys = sorted(glob.glob(os.path.join(args.src, '*', '*.npy')))
    if args.max:
        npys = npys[:args.max]

    for i, p in enumerate(npys):
        r = np.load(p)[..., 0]                                    # 64x2650 rango
        u = np.clip(255 * (1 - r / MAXR), 0, 255).astype(np.uint8)  # cerca=brillante
        u[r <= 0] = 0                                            # sin retorno = negro
        img = cv2.resize(u, (2650, 1024), interpolation=cv2.INTER_NEAREST)  # 64->1024
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(os.path.join(args.out, f'sweep_{i:03d}.png'), bgr)

    print(f'Generadas {len(npys)} PNG 2650x1024 en {args.out}')


if __name__ == '__main__':
    main()
