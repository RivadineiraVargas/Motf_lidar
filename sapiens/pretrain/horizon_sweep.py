"""
horizon_sweep.py — Barrido del HORIZONTE de predicción (1s/3s/5s/8s) con
validación cruzada por escenas, para responder: ¿el beneficio de la escena
LiDAR depende del horizonte? En la Fase 1 (pipeline viejo) apareció un
"punto dulce" en 3s (+25%); todos los experimentos del decoder MAE de esta
semana fueron a 8s, donde la escena no ayudó. Este barrido testea si a un
horizonte menor la conclusión cambia.

Horizontes: 1s=2wp, 3s=6wp, 5s=10wp, 8s=16wp (waypoints a 2Hz).

Diseño: por cada horizonte, cross-validation 5 folds x 1 semilla x 2 archs
(wayformer con escena, baseline sin escena). Comparación PAREADA por fold.
Es un barrido de SCREENING (1 semilla) — el objetivo es la TENDENCIA del
beneficio de la escena a lo largo del horizonte, no un p-valor definitivo en
cada punto (el de 8s ya lo tenemos con 15 corridas en cv_results.csv). Si
algún horizonte muestra reversión clara, se profundiza ahí con más semillas.

Las features del encoder NO dependen del horizonte -> el cache se comparte
entre todos los horizontes (se calcula una vez).

Uso:
  conda run -n sapiens_gpu python horizon_sweep.py \
      --enc work_dirs/rv_rect_overfit100/epoch_3000.pth --epochs 100
"""
import argparse
import csv
import os
import statistics as st

import numpy as np

from train_decoder_mini import load_frozen_encoder, train_decoder, encode_sweeps
from cross_validate_decoder import make_folds

HORIZONS = [('1s', 2), ('3s', 6), ('5s', 10), ('8s', 16)]
SEEDS = [0]
ARCHS = ['wayformer', 'baseline']


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['fold'] = int(r['fold']); r['seed'] = int(r['seed'])
        r['ade'] = float(r['ade'])
    return rows


def summarize(csv_path):
    rows = read_csv(csv_path)
    if not rows:
        print('[resumen] sin datos'); return
    print('\n' + '=' * 72)
    print('BARRIDO DE HORIZONTE — beneficio de la escena por horizonte')
    print('=' * 72)
    print(f'{"horiz":6s} {"arquitectura":>17s} {"con escena":>14s} {"baseline":>14s} '
          f'{"diff pareada":>16s} {"señal":>8s}')
    for hlabel, _ in HORIZONS:
        b = {(r['fold'], r['seed']): r['ade'] for r in rows
             if r['arch'] == f'baseline_h{hlabel}'}
        # cualquier arquitectura con escena medida a este horizonte, no solo
        # 'wayformer' (p.ej. wayformer_gated) — si no, se agregan corridas al
        # CSV y el resumen las ignora en silencio.
        archs = sorted({r['arch'].rsplit(f'_h{hlabel}', 1)[0] for r in rows
                        if r['arch'].endswith(f'_h{hlabel}')
                        and not r['arch'].startswith('baseline')})
        for arch in archs:
            w = {(r['fold'], r['seed']): r['ade'] for r in rows
                 if r['arch'] == f'{arch}_h{hlabel}'}
            keys = sorted(set(w) & set(b))
            if not keys:
                continue
            wv = [w[k] for k in keys]; bv = [b[k] for k in keys]
            diffs = [w[k] - b[k] for k in keys]   # <0 => la escena ayuda
            dm = st.mean(diffs)
            ds = st.stdev(diffs) if len(diffs) > 1 else 0.0
            wins = sum(1 for d in diffs if d < 0)
            senal = 'ESCENA' if dm < 0 else 'baseline'
            print(f'{hlabel:6s} {arch:>17s} '
                  f'{st.mean(wv):8.2f}±{(st.stdev(wv) if len(wv)>1 else 0):.2f} '
                  f'{st.mean(bv):8.2f}±{(st.stdev(bv) if len(bv)>1 else 0):.2f} '
                  f'{dm:+9.2f}±{ds:.2f}   {senal:>8s} ({wins}/{len(keys)})')
    print('\ndiff = wayformer - baseline (ADE @ horizonte, metros). '
          'Negativo = la escena AYUDA.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--enc', default='work_dirs/rv_rect_overfit100/epoch_3000.pth')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--hist', type=int, default=1)
    ap.add_argument('--cache', default='work_dirs/mae_feat_cache_100sw')
    ap.add_argument('--out', default='work_dirs/horizon_sweep')
    ap.add_argument('--folds', nargs='+', type=int, default=None,
                    help='subconjunto de folds (default: los 5). OBLIGATORIO '
                         'restringir al fold del encoder si el encoder fue '
                         're-pre-entrenado en las escenas de train de ese fold '
                         '(usarlo en otros folds seria FUGA).')
    ap.add_argument('--seeds', nargs='+', type=int, default=[0])
    ap.add_argument('--archs', nargs='+', default=ARCHS,
                    help='arquitecturas a correr (default: wayformer baseline). '
                         'El baseline no depende del encoder ni de la arch, asi '
                         'que si ya esta medido se saltea solo.')
    ap.add_argument('--horizons', nargs='+', default=None,
                    choices=[h for h, _ in HORIZONS],
                    help='subconjunto de horizontes (default: los 4). Util para '
                         'agregar semillas solo donde hay senal.')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    csv_path = f'{args.out}/horizon_results.csv'
    dev = 'cuda'

    scenes, folds = make_folds()
    # (indice real del fold, escenas retenidas) — el indice se preserva aunque
    # se corra un subconjunto, para que el CSV sea comparable con los barridos
    # previos y con cv_results.csv.
    fold_items = [(i, h) for i, h in enumerate(folds)
                  if args.folds is None or i in args.folds]
    encoder = load_frozen_encoder(args.enc, dev)
    print('[cache] precalentando features (una vez, compartidas entre horizontes)...')
    for i, sc in enumerate(scenes):
        encode_sweeps(encoder, sc, list(range(11)), dev, cache_dir=args.cache)
    print('  cache listo (25 escenas)')

    ya = {(r['fold'], r['seed'], r['arch']) for r in read_csv(csv_path)}
    new_file = not os.path.exists(csv_path)
    horizons = [h for h in HORIZONS
                if args.horizons is None or h[0] in args.horizons]
    total = len(horizons) * len(fold_items) * len(args.seeds) * len(args.archs)
    done = 0
    with open(csv_path, 'a', newline='') as fcsv:
        writer = csv.writer(fcsv)
        if new_file:
            writer.writerow(['fold', 'seed', 'arch', 'horizon_s', 'n_wp',
                             'ade', 'fde', 'acc', 'best_ep'])
        for hlabel, n_wp in horizons:
            for fi, held_out in fold_items:
                train_scenes = [s for s in scenes if s not in held_out]
                for seed in args.seeds:
                    for arch in args.archs:
                        label = f'{arch}_h{hlabel}'
                        done += 1
                        if (fi, seed, label) in ya:
                            print(f'[{done}/{total}] saltado {label} fold{fi} seed{seed}')
                            continue
                        print(f'[{done}/{total}] {label} fold{fi} seed{seed} '
                              f'(horizonte {hlabel}, {n_wp} wp)')
                        m, ep = train_decoder(
                            train_scenes, held_out, epochs=args.epochs,
                            lr=args.lr, arch=arch, hist=args.hist,
                            out_dir=f'{args.out}/{label}_f{fi}s{seed}',
                            seed=seed, encoder=encoder, cache_dir=args.cache,
                            eval_every=20, save_viz=False, verbose=False,
                            dev=dev, n_wp=n_wp)
                        # m['ade8'] es en realidad ADE @ horizonte actual
                        writer.writerow([fi, seed, label, hlabel, n_wp,
                                         m['ade8'], m['fde'], m['acc'], ep])
                        fcsv.flush()
                        # el gate no va al CSV (romperia el esquema de los
                        # barridos ya guardados); queda en el log, que es donde
                        # se lee "cuanto decidio el modelo usar la escena".
                        g, gf = m.get('gate'), m.get('gate_final')
                        print(f'       -> ADE@{hlabel} {m["ade8"]:.2f}  '
                              f'FDE {m["fde"]:.2f}  (ep {ep})'
                              + (f'  gate {g:+.3f} (final {gf:+.3f})'
                                 if g is not None else ''))

    summarize(csv_path)


if __name__ == '__main__':
    main()
