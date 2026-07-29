"""
cross_validate_decoder.py — Blinda el resultado "la escena LiDAR ayuda"
(Wayformer 7.19m vs baseline 7.85m en 82f9) con validación cruzada por
escenas + semillas múltiples. Sin esto, esa conclusión sale de UNA sola
medición y no es defendible frente a una pregunta de significancia.

Diseño (deterministic, dos fuentes de varianza separadas):
  - 5 FOLDS: las 25 escenas se parten en 5 grupos de 5 (orden alfabético
    fijo del scene_id -> reproducible). En cada fold, esas 5 son "no
    vistas" y las 20 restantes son train. Mide varianza ENTRE ESCENAS.
  - 3 SEEDS por fold: reinicializa los pesos del decoder (misma partición
    de datos). Mide varianza de INICIALIZACIÓN.
  - 2 arquitecturas (wayformer, baseline) en cada combinación fold x seed,
    para poder comparar de forma PAREADA (misma condición, mismos datos).

Optimización: las features del encoder MAE se cachean UNA vez por escena
(25 escenas) y se reusan en las 5x3x2=30 corridas -> evita recodificar lo
mismo 30 veces (el costo dominante sería si no se cachea).

Uso:
  conda run -n sapiens_gpu python cross_validate_decoder.py \
      --enc work_dirs/rv_rect_overfit100/epoch_3000.pth \
      --epochs 100 --out work_dirs/cv
"""
import argparse
import csv
import os
import statistics as st

import numpy as np
import torch

from train_decoder_mini import ROOT, CKPT, load_frozen_encoder, train_decoder

N_FOLDS = 5
SEEDS = [0, 1, 2]
ARCHS = ['wayformer', 'baseline']


def make_folds():
    scenes = sorted(os.listdir(f'{ROOT}/range_files'))
    assert len(scenes) == 25, f'esperaba 25 escenas, hay {len(scenes)}'
    folds = [scenes[i::N_FOLDS] for i in range(N_FOLDS)]   # 5 grupos de 5
    return scenes, folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--enc', default='work_dirs/rv_rect_overfit100/epoch_3000.pth')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--hist', type=int, default=1,
                    help='sweeps de MEMORIA DE ESCENA (no confundir con la '
                         'historia de trayectoria por objeto, que ya está '
                         'fija en H_PAST=10 dentro de build_sample). El '
                         'experimento a+b+c a blindar usó hist=1.')
    ap.add_argument('--out', default='work_dirs/cv')
    ap.add_argument('--cache', default='work_dirs/mae_feat_cache_100sw')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = 'cuda'

    scenes, folds = make_folds()
    print(f'25 escenas -> {N_FOLDS} folds de {len(folds[0])}: '
          f'{[f[:2] for f in [[s[:8] for s in fo] for fo in folds]]}')

    encoder = load_frozen_encoder(args.enc, dev)

    # precalentar cache: codifica las 25 escenas UNA vez antes del loop
    print('[cache] precalentando features del encoder (una vez por escena)...')
    from train_decoder_mini import encode_sweeps
    for i, sc in enumerate(scenes):
        encode_sweeps(encoder, sc, list(range(11)), dev, cache_dir=args.cache)
        print(f'  [{i+1}/25] {sc[:8]} cacheada')

    rows = []
    csv_path = f'{args.out}/cv_results.csv'
    with open(csv_path, 'w', newline='') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(['fold', 'seed', 'arch', 'ade8', 'ade5', 'fde', 'acc',
                         'train_ade8', 'best_ep', 'held_out_scenes'])

        for fi, held_out in enumerate(folds):
            train_scenes = [s for s in scenes if s not in held_out]
            for seed in SEEDS:
                for arch in ARCHS:
                    run_dir = f'{args.out}/fold{fi}_seed{seed}_{arch}'
                    print(f'\n--- fold {fi} seed {seed} arch {arch} '
                          f'(held-out: {[h[:8] for h in held_out]}) ---')
                    m, ep = train_decoder(
                        train_scenes, held_out, epochs=args.epochs, lr=args.lr,
                        arch=arch, hist=args.hist, out_dir=run_dir, seed=seed,
                        encoder=encoder, cache_dir=args.cache, eval_every=20,
                        save_viz=False, verbose=False, dev=dev)
                    print(f'  -> ADE8 {m["ade8"]:.2f}  ADE5 {m["ade5"]:.2f}  '
                          f'FDE {m["fde"]:.2f}  (mejor ep {ep})')
                    row = [fi, seed, arch, m['ade8'], m['ade5'], m['fde'],
                          m['acc'], m['train_ade8'], ep, ';'.join(held_out)]
                    rows.append(row)
                    writer.writerow(row)
                    fcsv.flush()

    # --- resumen ---
    print('\n' + '=' * 70)
    print('RESUMEN (mean ± std sobre 5 folds x 3 semillas = 15 corridas c/u)')
    print('=' * 70)
    for arch in ARCHS:
        vals = [r[3] for r in rows if r[2] == arch]     # ade8
        print(f'{arch:10s}  ADE8 = {st.mean(vals):.2f} ± {st.stdev(vals):.2f}  '
              f'(min {min(vals):.2f}, max {max(vals):.2f}, n={len(vals)})')

    # comparación PAREADA: mismo fold+seed, wayformer vs baseline
    print('\nComparación pareada (wayformer - baseline, por fold x seed):')
    diffs = []
    for fi in range(N_FOLDS):
        for seed in SEEDS:
            w = next(r[3] for r in rows if r[0] == fi and r[1] == seed and r[2] == 'wayformer')
            b = next(r[3] for r in rows if r[0] == fi and r[1] == seed and r[2] == 'baseline')
            diffs.append(w - b)
    d_mean, d_std = st.mean(diffs), st.stdev(diffs)
    wins = sum(1 for d in diffs if d < 0)
    print(f'  diff ADE8 media: {d_mean:+.2f} ± {d_std:.2f} '
          f'(negativo = wayformer gana)')
    print(f'  wayformer gana en {wins}/{len(diffs)} combinaciones fold x seed')
    # t-test pareado simple (sin scipy): t = mean_diff / (std_diff/sqrt(n))
    n = len(diffs)
    t_stat = d_mean / (d_std / np.sqrt(n)) if d_std > 0 else float('nan')
    print(f'  t pareado = {t_stat:.2f} (n={n}; |t|>~2.1 sugiere significancia '
          f'a p<0.05 con {n-1} grados de libertad, orientativo sin scipy)')

    print(f'\n[OK] resultados completos en {csv_path}')


if __name__ == '__main__':
    main()
