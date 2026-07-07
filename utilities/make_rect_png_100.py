"""
make_rect_png_100.py — Genera el set TRAIN de 100 sweeps multi-escena para el
overfit-100 (protocolo 10/100/1000 de Claudine), en PNG rect formato colega
(2650x1024, gris invertido, beams 64->1024 nearest).

Split (reproducible, sin azar):
  - Se EXCLUYE la escena 82f90331a1dfe968 (test "no-visto" del overfit-10).
  - 24 escenas restantes x sweeps {0,2,4,6} = 96
  - + sweep 8 de las primeras 4 escenas (orden alfabetico) = 100
  - Val intra-escena queda libre: sweeps impares (1,3,5,7,9) nunca entrenados.

Uso:
  python utilities/make_rect_png_100.py \
      --src waymo_clean/range_files --out waymo_clean/range_png_rect/train100
"""
import os, argparse
import numpy as np
import cv2

MAXR = 75.0
UNSEEN_SCENE = '82f90331a1dfe968'


def conv(src, dst):
    r = np.load(src)[..., 0]
    u = np.clip(255 * (1 - r / MAXR), 0, 255).astype(np.uint8)
    u[r <= 0] = 0
    img = cv2.resize(u, (2650, 1024), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(dst, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='waymo_clean/range_files')
    ap.add_argument('--out', default='waymo_clean/range_png_rect/train100')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    scenes = sorted(d for d in os.listdir(args.src)
                    if os.path.isdir(os.path.join(args.src, d)) and d != UNSEEN_SCENE)
    assert len(scenes) == 24, f'esperaba 24 escenas de train, hay {len(scenes)}'

    pares = [(s, t) for s in scenes for t in (0, 2, 4, 6)]
    pares += [(s, 8) for s in scenes[:4]]
    assert len(pares) == 100

    for i, (s, t) in enumerate(pares):
        conv(os.path.join(args.src, s, f'{t}.npy'),
             os.path.join(args.out, f'sweep_{i:03d}_{s[:8]}_{t}.png'))
    print(f'Generadas {len(pares)} PNG en {args.out}')


if __name__ == '__main__':
    main()
