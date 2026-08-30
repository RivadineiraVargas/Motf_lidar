"""
eval_fase1_seeds.py — evalúa un checkpoint de Fase 1 (vóxeles, 10 escenas) con
las métricas corregidas por la auditoría del 23/08.

Qué corrige respecto de la evaluación original de Fase 1:
  B5 — separa objetos MÓVILES de parados. El 60-75% se desplaza <1 m; para esos
       la velocidad constante ya es casi perfecta y ningún modelo puede mejorarla,
       solo empeorarla. Promediar sobre esa población comprime cualquier efecto
       real hacia cero. La métrica primaria pasa a ser ADE sobre móviles.
  agregación POR ESCENA — las muestras de una misma escena están correlacionadas;
       promediarlas todas juntas como independientes fabrica significancia falsa.

Salida: una fila por (variante, semilla, escena) en CSV, para que el análisis
posterior agregue por escena y compare pareado por semilla.
"""
import argparse, csv, os
import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpretrain.registry import MODELS, DATASETS

MOVING_MIN = 1.0     # m de desplazamiento GT para contar como móvil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--variant', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out', default='work_dirs/fase1_seeds/fase1_results.csv')
    ap.add_argument('--val-scenes', nargs='+', default=None,
                    help='escenas retenidas del fold (default: cfg.val_scenes)')
    ap.add_argument('--fold', type=int, default=-1)
    ap.add_argument('--sin-clip', action='store_true',
                    help='evalúa contra la trayectoria SIN recortar (error real)')
    ap.add_argument('--eval-windows', type=int, default=1,
                    help='ventanas temporales por objeto (B2 de la auditoría). '
                         '1 = comportamiento histórico; 7 = tope que permiten los '
                         '11 sweeps de LiDAR con history_len=5.')
    args = ap.parse_args()
    init_default_scope('mmpretrain')
    cfg = Config.fromfile(args.cfg)
    dev = 'cuda'

    model = MODELS.build(cfg.model)
    sd = torch.load(args.ckpt, map_location='cpu')
    pesos = sd.get('state_dict', sd)
    faltan = model.load_state_dict(pesos, strict=False)
    # ARREGLO 30/08 (hallazgo 10). strict=False acepta en silencio un checkpoint
    # que no case en NADA —prefijos cambiados, arquitectura distinta— y deja el
    # modelo con pesos aleatorios produciendo un ADE plausible. Es exactamente el
    # fallo de c6c9e05, donde dos experimentos corrieron sin encoder y se detectó
    # por casualidad. Acá se exige que algo se haya cargado de verdad.
    cargadas = len(pesos) - len(getattr(faltan, 'unexpected_keys', []))
    if cargadas <= 0:
        raise RuntimeError(
            f'{args.ckpt}: ninguna clave del checkpoint coincidió con el modelo de '
            f'{args.cfg}. Evaluar así mide un modelo ALEATORIO.')
    esperadas = len(model.state_dict())
    if cargadas < 0.5 * esperadas:
        print(f'[eval] AVISO: solo {cargadas}/{esperadas} tensores cargados del '
              f'checkpoint — revisar que config y checkpoint correspondan.')
    model = model.to(dev)
    model.eval()
    # H5 de la auditoría: el valor del gate se publicaba desde logs no versionados.
    # Acá va al CSV. Como se evalúa en época FIJA (la última), el gate del
    # checkpoint ES el convergido — no se repite la trampa de Fase 2, donde el
    # early-stop guardaba un gate casi sin mover.
    gate = (float(torch.tanh(model.scene_gate).item())
            if hasattr(model, 'scene_gate') else float('nan'))

    # dataset de VALIDACION: las 2 escenas retenidas, nunca vistas en train
    dcfg = dict(cfg.train_dataloader.dataset)
    dcfg['scenes'] = list(args.val_scenes or cfg.val_scenes)
    dcfg['augment'] = False
    dcfg['eval_windows'] = args.eval_windows
    if args.sin_clip:
        dcfg['clip_norm'] = None
    ds = DATASETS.build(dcfg)

    is_baseline = 'Baseline' in cfg.model['type']
    per_scene = {}
    pred_len = cfg.pred_len
    for i in range(len(ds)):
        d = ds[i]
        scene = d['scene_name']
        # misma lógica de desnormalización que evaluate_clean10_newmae.py:
        # el dataset normaliza cada trayectoria con su propia media/desvío.
        with torch.no_grad():
            h = d['obj_history_flat'].unsqueeze(0).to(dev)
            # el baseline (BaselineTrajectoryModel) NO recibe la escena
            pred_flat = (model(h, mode='predict') if is_baseline else
                         model(d['inputs'].unsqueeze(0).to(dev), h,
                               mode='predict')).cpu()
        pred = (pred_flat.view(pred_len, 3) * d['norm_std'] + d['norm_mean']).numpy()
        gt = (d['obj_future_flat'].view(pred_len, 3) * d['norm_std'] + d['norm_mean']).numpy()
        err = np.linalg.norm(pred[:, :2] - gt[:, :2], axis=1)          # solo XY
        # desplazamiento REAL del objeto: define si es móvil (B5)
        despl = float(np.linalg.norm(gt[-1, :2] - gt[0, :2]))
        per_scene.setdefault(scene, []).append((float(err.mean()), float(err[-1]), despl))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    new = not os.path.exists(args.out)
    with open(args.out, 'a', newline='') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(['fold', 'variant', 'seed', 'scene', 'n_obj', 'n_moving',
                        'ade_all', 'fde_all', 'ade_moving', 'fde_moving', 'gate'])
        for sc, v in sorted(per_scene.items()):
            a = np.array([x[0] for x in v]); f = np.array([x[1] for x in v])
            mv = np.array([x[2] for x in v]) >= MOVING_MIN
            w.writerow([args.fold, args.variant, args.seed, sc, len(v), int(mv.sum()),
                        f'{a.mean():.5f}', f'{f.mean():.5f}',
                        f'{a[mv].mean():.5f}' if mv.any() else '',
                        f'{f[mv].mean():.5f}' if mv.any() else '',
                        f'{gate:.5f}'])
    print(f'[eval] {args.variant} seed {args.seed}: '
          + ', '.join(f'{sc} n={len(v)}' for sc, v in sorted(per_scene.items())))


if __name__ == '__main__':
    main()
