# curva_mae_voxel.py — ¿en qué época está el óptimo del MAE de vóxeles?
#
# LA PREGUNTA QUE DEJÓ ABIERTA LA ADENDA DEL EXPERIMENTO 23. Los cinco encoders
# de work_dirs/f1cv se entrenaron 1000 épocas y TODA la Fase 1 (experimentos
# 19-22) usó `epoch_1000.pth`. La adenda midió 600/800/1000 y no encontró
# degradación — pero esos son el ÚLTIMO 40 % de la corrida, porque el config
# tiene `checkpoint=dict(interval=200, max_keep_ckpts=3)` y en disco no queda
# nada anterior.
#
# En range-view (curva_overfit10.py) el óptimo estaba en la época 50 de 6000, o
# sea al 0,8 % de la corrida. El equivalente acá sería la época ~8. Si el pico
# fuera igual de temprano, las tres mediciones de la adenda caerían todas
# después de la caída y una meseta baja se vería exactamente así de plana.
# Lo que la adenda probó es "¿se degrada en el último 40 %?" (no). Lo que NO
# probó es "¿es la 1000 la mejor época?".
#
# POR QUE IMPORTA. Si el óptimo está mucho antes de la 1000, los experimentos
# 19-22 midieron con encoders subóptimos — y ese es el conjunto que respondió
# que la escena no aporta (0/5 folds), que es la pregunta central de la tesis.
# No la refutaría por sí solo, pero dejaría la condición SIN PROBAR, que es el
# mismo error del experimento 17.
#
# QUE MIDE. Para cada checkpoint, la pérdida de reconstrucción enmascarada sobre
# las mismas tres poblaciones del experimento 21, con la MISMA implementación
# —este script las importa de diagnostico_encoder_mae.py en vez de reescribirlas,
# para que la curva y la adenda sean comparables tensor a tensor:
#   train_vistas  — las 8 ventanas exactas que el MAE optimizó
#   train_nuevas  — las otras 6 ventanas/escena de las MISMAS escenas
#   val           — las 7 ventanas de cada una de las 2 escenas RETENIDAS
# La columna que decide es `val`: es la única fuera de las escenas de train.
#
# DOS COSAS HEREDADAS QUE NO HAY QUE TOCAR:
#   - modelo.train(), no eval(). MAEViT4D solo enmascara bajo `if self.training`;
#     en eval la pérdida sale 0/1e-6 = 0 para cualquier modelo. `perdidas()` y
#     `verificar_modo_train()` vienen del diagnóstico justamente por esto.
#   - Máscaras pareadas por semilla: la máscara nº k es idéntica en las tres
#     poblaciones Y en todos los checkpoints, así que las diferencias a lo largo
#     de la curva no pueden venir del sorteo.
#
# CONTROL DE SANIDAD. La época 1000 de esta curva debe reproducir el valor de la
# adenda para el mismo fold (fold 0: 0,1939 · 1: 0,2445 · 2: 0,1261 · 3: 0,1632 ·
# 4: 0,2060). El script lo compara e imprime la diferencia. Si no cierra, algo
# cambió entre las dos mediciones y la curva no es comparable con la adenda.
#
# No entrena ni escribe en work_dirs más que su propio CSV.
#
# Uso:  conda activate sapiens_gpu
#       cd sapiens/pretrain && python curva_mae_voxel.py --fold 0 --work-dir work_dirs/f1cv_curva/mae_fold0
import argparse
import csv
import gc
import glob
import os
import re
import statistics as st

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpretrain.registry import MODELS

from diagnostico_encoder_mae import (D, escenas_val, construir_pobl,
                                     verificar_modo_train, perdidas, trivial)

# Valores de la adenda del exp. 23 (población `val`, época 1000) para el control
# de sanidad. Fuente: docs/EXPERIMENTOS_DECODER.md, "Adenda al 23".
#
# MEDIDOS CON 4 MASCARAS, y el doc no lo dice: hay que dejarlo acá. Verificado
# midiendo el mismo checkpoint del fold 0 con las dos densidades —4 da 0,1939,
# que es el valor de la adenda; 8 da 0,1960—. Por eso `--mascaras` tiene default
# 4 y run_curva_mae.sh pasa 4: subirlo a 8 no mide "mejor", mide OTRA cosa y
# rompe la comparabilidad con la adenda, que es el punto del control.
ADENDA_EP1000 = {0: 0.1939, 1: 0.2445, 2: 0.1261, 3: 0.1632, 4: 0.2060}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--work-dir', required=True,
                    help='dir con los epoch_*.pth densos del re-pre-entrenamiento')
    ap.add_argument('--mascaras', type=int, default=4,
                    help='4 es lo que usó la adenda del exp. 23; ver ADENDA_EP1000')
    ap.add_argument('--cada', type=int, default=1,
                    help='evaluar 1 de cada N checkpoints')
    ap.add_argument('--out', default=None)
    ap.add_argument('--forzar', action='store_true',
                    help='sobrescribir un CSV existente aunque tenga MAS puntos '
                         'que la corrida actual (ver el guard de abajo)')
    args = ap.parse_args()

    init_default_scope('mmpretrain')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    F = args.fold
    cfg_path = f'{D}/f1cv_mae_fold{F}.py'
    cfg = Config.fromfile(cfg_path)

    tr = list(cfg.train_dataloader.dataset['scenes'])
    va = escenas_val(cfg_path)
    assert not (set(tr) & set(va)), f'fold {F}: fuga — escena en train y val'

    pobl = {
        'train_vistas': construir_pobl(cfg, tr, 1),
        'train_nuevas': construir_pobl(cfg, tr, 7, t0_desde=1),
        'val':          construir_pobl(cfg, va, 7),
    }
    print(f'######## FOLD {F} — train {len(tr)} escenas · val {va}')
    for k, xs in pobl.items():
        print(f'  {k:13s} {len(xs):3d} ventanas')
    triv = {k: st.mean(trivial(xs)) for k, xs in pobl.items()}
    print(f'  trivial (predecir 0): ' +
          '  '.join(f'{k} {v:.4f}' for k, v in triv.items()))

    cks = sorted(glob.glob(f'{args.work_dir}/epoch_*.pth'),
                 key=lambda p: int(re.search(r'epoch_(\d+)', p).group(1)))
    if not cks:
        raise SystemExit(f'{args.work_dir} no tiene epoch_*.pth. Entrenar primero '
                         f'con run_curva_mae.sh.')
    cks = cks[::args.cada]
    print(f'\n{len(cks)} checkpoints · {args.mascaras} máscaras por muestra '
          f'(semillas 0..{args.mascaras - 1}, pareadas)\n')

    out = args.out or f'{args.work_dir}/curva_mae_voxel.csv'
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)

    # GUARD ANTIBORRADO. El CSV se abre en 'w' al final, así que una corrida con
    # POCOS checkpoints pisa la curva de una corrida con MUCHOS. Y eso no es
    # hipotético: run_curva_mae.sh poda los checkpoints al terminar —deja el mejor
    # y el 1000—, de modo que volver a correr el mismo fold encuentra 2, mide una
    # "curva" de 2 puntos y borra la de 101 sin decir nada. El experimento 26
    # entero vive en estos CSV.
    if os.path.exists(out) and not args.forzar:
        with open(out) as fh:
            previas = max(sum(1 for _ in fh) - 1, 0)
        if previas > len(cks) + 1:      # +1 por la fila 'sin_entrenar'
            raise SystemExit(
                f'{out} ya tiene {previas} puntos y esta corrida solo mediría '
                f'{len(cks) + 1} ({len(cks)} checkpoints en {args.work_dir}).\n'
                f'Sobrescribirlo perdería la curva. Si los checkpoints fueron '
                f'podados, la curva YA está medida y no hay nada que rehacer; '
                f'para re-medir de verdad hay que re-entrenar primero.\n'
                f'Usar --forzar solo si de verdad se quiere pisar el CSV.')

    filas = []

    for etiqueta, ruta in ([('sin_entrenar', None)] +
                           [(re.search(r'epoch_(\d+)', c).group(1), c) for c in cks]):
        modelo = MODELS.build(cfg.model)
        if ruta is None:
            modelo.init_weights()
        else:
            ck = torch.load(ruta, map_location='cpu', weights_only=False)
            pesos = ck.get('state_dict', ck)
            inc = modelo.load_state_dict(pesos, strict=False)
            # Mismo guard que eval_fase1_seeds.py (hallazgo 10): strict=False
            # acepta en silencio un checkpoint que no case en NADA y deja pesos
            # aleatorios. Acá eso se leería como "esta época es malísima".
            cargadas = len(pesos) - len(inc.unexpected_keys)
            if cargadas <= 0:
                raise SystemExit(f'{ruta} no aportó ni un peso a {cfg_path}.')
            if cargadas < 0.5 * len(modelo.state_dict()):
                print(f'  !! AVISO: solo {cargadas}/{len(modelo.state_dict())} '
                      f'tensores cargados en {ruta}')
            del ck, pesos
            gc.collect()
        modelo.to(dev)
        verificar_modo_train(modelo)
        modelo.train()      # imprescindible para que haya máscara

        fila = {'epoca': etiqueta}
        for k, xs in pobl.items():
            v = perdidas(modelo, xs, args.mascaras, dev)
            fila[k] = st.mean(v)
            fila[k + '_sd'] = st.stdev(v) if len(v) > 1 else 0.0
        filas.append(fila)
        print(f'  época {etiqueta:>12s}  vistas {fila["train_vistas"]:.4f}  '
              f'nuevas {fila["train_nuevas"]:.4f}  '
              f'val {fila["val"]:.4f} ± {fila["val_sd"]:.4f}')
        del modelo
        gc.collect()
        torch.cuda.empty_cache()

    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)

    ent = [f for f in filas if f['epoca'] != 'sin_entrenar']
    base = next(f for f in filas if f['epoca'] == 'sin_entrenar')
    mejor = min(ent, key=lambda f: f['val'])
    ultima = max(ent, key=lambda f: int(f['epoca']))
    print(f'\n=== FOLD {F} ===')
    print(f'  sin entrenar      val {base["val"]:.4f}')
    print(f'  MEJOR época {mejor["epoca"]:>5s}  val {mejor["val"]:.4f}')
    print(f'  última época {ultima["epoca"]:>4s}  val {ultima["val"]:.4f}'
          f'   ({(ultima["val"] - mejor["val"]) / mejor["val"] * 100:+.1f} % vs la mejor)')

    # Control de sanidad contra la adenda del exp. 23.
    ref = ADENDA_EP1000.get(F)
    if ref is not None and ultima['epoca'] == '1000':
        d = ultima['val'] - ref
        estado = 'OK' if abs(d) < 0.01 else '!! NO CIERRA'
        print(f'  control vs adenda (ép1000 = {ref:.4f}): {d:+.4f}  {estado}')
        if abs(d) >= 0.01:
            print('     La curva NO es comparable con la adenda: algo cambió entre')
            print('     las dos mediciones (datos, config o semilla). Revisar antes')
            print('     de sacar conclusiones.')
    print(f'\n  CSV: {out}')


if __name__ == '__main__':
    main()
