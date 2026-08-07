"""
cross_validate_decoder.py — Blinda con validación cruzada por escenas +
semillas cualquier comparación entre arquitecturas del decoder mini. Nació
para testear "la escena LiDAR ayuda" (Wayformer vs baseline) y se generalizó
para poder agregar arquitecturas nuevas (p.ej. wayformer_pooled) sin
re-correr lo que ya está medido.

Diseño (deterministic, dos fuentes de varianza separadas):
  - 5 FOLDS: las 25 escenas se parten en 5 grupos de 5 (orden alfabético
    fijo del scene_id -> reproducible). En cada fold, esas 5 son "no
    vistas" y las 20 restantes son train. Mide varianza ENTRE ESCENAS.
  - 3 SEEDS por fold: reinicializa los pesos del decoder (misma partición
    de datos). Mide varianza de INICIALIZACIÓN.
  - Arquitecturas seleccionables con --archs: se pueden agregar corridas
    nuevas al mismo cv_results.csv (modo append) sin repetir las que ya
    están, y el resumen final SIEMPRE lee el archivo completo del disco
    (no solo lo corrido en esta invocación) para comparar todo lo disponible.

Optimización: las features del encoder MAE se cachean UNA vez por escena
(25 escenas) y se reusan entre corridas -> evita recodificar lo mismo.

Uso:
  # corrida original (wayformer + baseline):
  conda run -n sapiens_gpu python cross_validate_decoder.py \
      --enc work_dirs/rv_rect_overfit100/epoch_3000.pth --epochs 100

  # agregar una arquitectura nueva al mismo CSV, sin re-correr las otras:
  conda run -n sapiens_gpu python cross_validate_decoder.py \
      --archs wayformer_pooled --epochs 100
"""
import argparse
import csv
import os
import statistics as st
from itertools import combinations

import numpy as np

from train_decoder_mini import ROOT, load_frozen_encoder, train_decoder, encode_sweeps

N_FOLDS = 5
SEEDS = [0, 1, 2]
ALL_ARCHS = ['wayformer', 'baseline', 'wayformer_pooled', 'wayformer_gated']


def make_folds():
    scenes = sorted(os.listdir(f'{ROOT}/range_files'))
    assert len(scenes) == 25, f'esperaba 25 escenas, hay {len(scenes)}'
    folds = [scenes[i::N_FOLDS] for i in range(N_FOLDS)]   # 5 grupos de 5
    return scenes, folds


def read_csv(csv_path):
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline='') as f:
        r = csv.DictReader(f)
        rows = []
        for row in r:
            row['fold'] = int(row['fold']); row['seed'] = int(row['seed'])
            row['ade8'] = float(row['ade8'])
            rows.append(row)
        return rows


def summarize(csv_path):
    rows = read_csv(csv_path)
    if not rows:
        print('[resumen] sin datos aún'); return
    archs = sorted(set(r['arch'] for r in rows))
    n_per_arch = {a: sum(1 for r in rows if r['arch'] == a) for a in archs}

    print('\n' + '=' * 70)
    print(f'RESUMEN acumulado ({csv_path})')
    print('=' * 70)
    for a in archs:
        vals = [r['ade8'] for r in rows if r['arch'] == a]
        print(f'{a:18s}  ADE8 = {st.mean(vals):.2f} ± '
              f'{st.stdev(vals) if len(vals) > 1 else 0:.2f}  '
              f'(min {min(vals):.2f}, max {max(vals):.2f}, n={len(vals)})')

    print('\nComparaciones PAREADAS (mismo fold+seed en ambos archs):')
    for a1, a2 in combinations(archs, 2):
        idx1 = {(r['fold'], r['seed']): r['ade8'] for r in rows if r['arch'] == a1}
        idx2 = {(r['fold'], r['seed']): r['ade8'] for r in rows if r['arch'] == a2}
        keys = sorted(set(idx1) & set(idx2))
        if len(keys) < 2:
            continue
        diffs = [idx1[k] - idx2[k] for k in keys]
        d_mean, d_std = st.mean(diffs), st.stdev(diffs)
        wins1 = sum(1 for d in diffs if d < 0)
        n = len(diffs)
        t_stat = d_mean / (d_std / np.sqrt(n)) if d_std > 0 else float('nan')
        print(f'  {a1} - {a2}: diff medio {d_mean:+.2f} ± {d_std:.2f}  '
              f'({a1} gana en {wins1}/{n})  t={t_stat:.2f} (n={n}, orientativo)')
    print(f'\nn por arquitectura: {n_per_arch}')


def run(archs, epochs, lr, hist, enc_ckpt, cache_dir, csv_path, dev='cuda',
       folds_subset=None, finetune_blocks=0, enc_lr=1e-5, model_arch='wayformer'):
    scenes, folds = make_folds()
    print(f'25 escenas -> {N_FOLDS} folds de {len(folds[0])}')
    print(f'arquitecturas a correr esta invocación: {archs}')
    fold_ids = folds_subset if folds_subset is not None else list(range(N_FOLDS))

    finetuning = finetune_blocks > 0
    encoder = None
    if not finetuning:
        # frozen: se carga UNA vez y se reusa (nunca muta) + cache en disco
        encoder = load_frozen_encoder(enc_ckpt, dev)
        print('[cache] precalentando features del encoder (una vez por escena)...')
        for i, sc in enumerate(scenes):
            encode_sweeps(encoder, sc, list(range(11)), dev, cache_dir=cache_dir)
            print(f'  [{i+1}/25] {sc[:8]} cacheada')
    else:
        # fine-tuning: el encoder MUTA durante cada corrida -> hay que
        # recargarlo desde el checkpoint ANTES de cada fold/seed, si no,
        # una corrida contamina a la siguiente con pesos ya ajustados
        print('[fine-tune] el encoder se recarga fresco antes de cada corrida '
              '(sus pesos cambian durante el entrenamiento)')

    ya = {(r['fold'], r['seed'], r['arch']) for r in read_csv(csv_path)}
    new_file = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as fcsv:
        writer = csv.writer(fcsv)
        if new_file:
            writer.writerow(['fold', 'seed', 'arch', 'ade8', 'ade5', 'fde', 'acc',
                             'train_ade8', 'best_ep', 'held_out_scenes'])
        for fi in fold_ids:
            held_out = folds[fi]
            train_scenes = [s for s in scenes if s not in held_out]
            for seed in SEEDS:
                for arch_label in archs:
                    if (fi, seed, arch_label) in ya:
                        print(f'[saltado] fold {fi} seed {seed} {arch_label} (ya en CSV)')
                        continue
                    run_dir = f'work_dirs/cv/fold{fi}_seed{seed}_{arch_label}'
                    print(f'\n--- fold {fi} seed {seed} arch {arch_label} '
                          f'(held-out: {[h[:8] for h in held_out]}) ---')
                    enc_this_run = (load_frozen_encoder(enc_ckpt, dev)
                                   if finetuning else encoder)
                    m, ep = train_decoder(
                        train_scenes, held_out, epochs=epochs, lr=lr,
                        arch=model_arch, hist=hist, out_dir=run_dir, seed=seed,
                        encoder=enc_this_run, cache_dir=cache_dir, eval_every=20,
                        save_viz=False, verbose=False, dev=dev,
                        finetune_encoder_blocks=finetune_blocks, enc_lr=enc_lr)
                    print(f'  -> ADE8 {m["ade8"]:.2f}  ADE5 {m["ade5"]:.2f}  '
                          f'FDE {m["fde"]:.2f}  (mejor ep {ep})')
                    writer.writerow([fi, seed, arch_label, m['ade8'], m['ade5'],
                                     m['fde'], m['acc'], m['train_ade8'], ep,
                                     ';'.join(held_out)])
                    fcsv.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--enc', default='work_dirs/rv_rect_overfit100/epoch_3000.pth')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--hist', type=int, default=1,
                    help='sweeps de MEMORIA DE ESCENA (no confundir con la '
                         'historia de trayectoria por objeto, ya fija en '
                         'H_PAST=10 dentro de build_sample).')
    ap.add_argument('--archs', nargs='+', choices=ALL_ARCHS,
                    default=['wayformer', 'baseline'])
    ap.add_argument('--out', default='work_dirs/cv')
    ap.add_argument('--cache', default='work_dirs/mae_feat_cache_100sw')
    ap.add_argument('--folds', nargs='+', type=int, default=None,
                    help='restringir a estos folds (default: los 5)')
    ap.add_argument('--finetune-blocks', type=int, default=0,
                    help='descongelar los últimos N bloques del encoder '
                         '(0 = frozen, comportamiento original)')
    ap.add_argument('--enc-lr', type=float, default=1e-5)
    ap.add_argument('--label', default=None,
                    help='etiqueta del arch en el CSV cuando se hace '
                         'fine-tuning (default: "<arch>_ft<N>")')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    csv_path = f'{args.out}/cv_results.csv'

    if args.finetune_blocks > 0:
        model_arch = args.archs[0]           # una sola clase de modelo aquí
        label = args.label or f'{model_arch}_ft{args.finetune_blocks}'
        run([label], args.epochs, args.lr, args.hist, args.enc, args.cache,
           csv_path, folds_subset=args.folds,
           finetune_blocks=args.finetune_blocks, enc_lr=args.enc_lr,
           model_arch=model_arch)
    else:
        run(args.archs, args.epochs, args.lr, args.hist, args.enc, args.cache,
           csv_path, folds_subset=args.folds)
    summarize(csv_path)


if __name__ == '__main__':
    main()
