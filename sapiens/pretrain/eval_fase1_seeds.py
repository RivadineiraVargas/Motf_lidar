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
    ap.add_argument('--poblacion-hist', type=int, default=None,
                    help='Evalúa sobre la población que soportaría una historia '
                         'de N frames, prediciendo el MISMO futuro: se queda con '
                         'las ventanas cuyo futuro arranca en el frame absoluto N '
                         '(frame0 + history_len == N) y cuyo objeto existe también '
                         'con history_len=N. Sirve para comparar modelos de '
                         'distinta historia sin cambiar ni la población ni el '
                         'objetivo: con N=11, un modelo de h=5 usa su ventana '
                         'f0=6 y uno de h=11 su ventana f0=0, y los dos predicen '
                         'los frames 11..40 de los mismos objetos. Sin esto la '
                         'comparación es entre poblaciones distintas.')
    ap.add_argument('--eval-windows', type=int, default=1,
                    help='ventanas temporales por objeto (B2 de la auditoría). '
                         '1 = comportamiento histórico; 7 = tope que permiten los '
                         '11 sweeps de LiDAR con history_len=5.')
    ap.add_argument('--cfg-options', nargs='+', default=None,
                    metavar='CLAVE=VALOR',
                    help='sobreescribe entradas del config, igual que en '
                         'tools/train.py. Ej.: model.num_modes=6. En el flujo '
                         'normal NO hace falta: el config de entrenamiento ya '
                         'declara num_modes y este evaluador lee ESE config, así '
                         'que el K de la evaluación es el mismo con el que se '
                         'entrenó por construcción.')
    args = ap.parse_args()
    init_default_scope('mmpretrain')
    cfg = Config.fromfile(args.cfg)
    if args.cfg_options:
        d = {}
        for kv in args.cfg_options:
            if '=' not in kv:
                raise SystemExit(f'--cfg-options espera CLAVE=VALOR, no {kv!r}')
            k, v = kv.split('=', 1)
            try:
                d[k] = eval(v, {'__builtins__': {}})   # números, listas, True/False
            except Exception:
                d[k] = v                                # strings sueltos
        cfg.merge_from_dict(d)
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

    if args.poblacion_hist is not None:
        N = args.poblacion_hist
        # Población de referencia: los objetos que EXISTEN con historia N. Se
        # midió que el conjunto de h=11 es subconjunto del de h=5 en los 5 folds,
        # así que tomar el de la historia más larga deja los brazos pareados.
        ref = dict(dcfg)
        ref['history_len'] = N
        ref['sequence_len'] = N + cfg.pred_len
        permitidos = {(it['scene_name'], it['object_id'])
                      for it in DATASETS.build(ref).data_list
                      if it['frame0'] + N == N}
        h = ds.history_len
        # OJO: se filtra data_list, NO se usa len(ds). BaseDataset serializa la
        # lista en __init__ y su __len__ sigue contando los ítems originales;
        # este dataset, en cambio, lee data_list en __getitem__. Filtrar y luego
        # iterar con len(ds) leería índices fuera de rango.
        ds.data_list = [it for it in ds.data_list
                        if it['frame0'] + h == N
                        and (it['scene_name'], it['object_id']) in permitidos]
        print(f'[eval] población alineada a historia {N}: {len(ds.data_list)} '
              f'ventanas de {len(permitidos)} objetos (history_len={h}, '
              f'futuro desde el frame {N})')
        if not ds.data_list:
            raise SystemExit(f'[eval] población vacía con --poblacion-hist {N} '
                             f'e history_len={h}: no hay nada que medir.')

    is_baseline = 'Baseline' in cfg.model['type']
    # num_modes vive en el modelo construido, no en el config: así respeta un
    # --cfg-options model.num_modes=6 y no miente si el config no lo declara.
    K = int(getattr(model, 'num_modes', 1))
    es_multimodal = K > 1
    if es_multimodal:
        print(f'[eval] modelo MULTIMODAL con K={K}: se reportan ade_all (modo más '
              f'probable, comparable con los exp. 15-22) y min_ade_k (el mejor de '
              f'los {K}, comparable con la literatura).')
    per_scene = {}
    pred_len = cfg.pred_len
    for i in range(len(ds.data_list)):
        d = ds[i]
        scene = d['scene_name']
        # misma lógica de desnormalización que evaluate_clean10_newmae.py:
        # el dataset normaliza cada trayectoria con su propia media/desvío.
        with torch.no_grad():
            h = d['obj_history_flat'].unsqueeze(0).to(dev)
            escena = None if is_baseline else d['inputs'].unsqueeze(0).to(dev)
            # OJO con el nombre: `args` es el Namespace de argparse y se usa más
            # abajo (args.out). Llamar así a esta tupla lo tapaba y reventaba al
            # escribir el CSV, después de evaluarlo todo.
            entrada = (h,) if is_baseline else (escena, h)
            # 'predict' devuelve el modo MÁS PROBABLE (con K=1, el único). Es lo
            # que mantiene esta métrica comparable con los experimentos 15-22.
            pred_flat = model(*entrada, mode='predict').cpu()
            # 'predict_multi' devuelve las K hipótesis. Solo se pide si el modelo
            # es multimodal: un modelo de K=1 no tiene minADE que reportar, y
            # pedírselo daría exactamente el mismo número dos veces.
            modos = None
            if es_multimodal:
                modos, _ = model(*entrada, mode='predict_multi')
                modos = modos.cpu()

        def a_metros(t):
            return (t.view(-1, pred_len, 3) * d['norm_std'] + d['norm_mean']).numpy()

        pred = a_metros(pred_flat)[0]
        gt = (d['obj_future_flat'].view(pred_len, 3) * d['norm_std'] + d['norm_mean']).numpy()
        err = np.linalg.norm(pred[:, :2] - gt[:, :2], axis=1)          # solo XY
        # desplazamiento REAL del objeto: define si es móvil (B5)
        despl = float(np.linalg.norm(gt[-1, :2] - gt[0, :2]))

        # minADE_k / minFDE_k: el error del MEJOR de los K modos, que es la
        # métrica de WOMD y la que reportan Wayformer, MTR y MotionLM. NO es
        # comparable con el ADE de arriba: con K modos el mínimo siempre es menor
        # o igual, así que un k=6 se ve mejor que un k=1 aunque no haya aprendido
        # nada. Por eso se guardan LAS DOS y el agregador las trata por separado.
        if modos is None:
            min_ade, min_fde = float(err.mean()), float(err[-1])
        else:
            todos = a_metros(modos.squeeze(0))                  # (K, pred_len, 3)
            e = np.linalg.norm(todos[:, :, :2] - gt[None, :, :2], axis=2)  # (K, T)
            ade_k = e.mean(axis=1)
            # El mejor modo para ADE y para FDE puede NO ser el mismo. WOMD define
            # minADE y minFDE como mínimos independientes, así que se toman aparte.
            min_ade, min_fde = float(ade_k.min()), float(e[:, -1].min())

        per_scene.setdefault(scene, []).append(
            (float(err.mean()), float(err[-1]), despl, min_ade, min_fde))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    new = not os.path.exists(args.out)
    with open(args.out, 'a', newline='') as fh:
        w = csv.writer(fh)
        if new:
            # Cuatro columnas NUEVAS al final. Los nombres siguen el patrón
            # {métrica}_{población} que agregar_resultados.py arma en `leer()`
            # (`col = f'{metrica}_{"moving"|"all"}'`), así que basta con agregar
            # 'minade'/'minfde' a --metrica y funciona sin más código. Un CSV de
            # 11 columnas y uno de 15 conviven en la misma corrida del agregador,
            # porque lee con csv.DictReader (por nombre, no por posición).
            w.writerow(['fold', 'variant', 'seed', 'scene', 'n_obj', 'n_moving',
                        'ade_all', 'fde_all', 'ade_moving', 'fde_moving', 'gate',
                        'minade_all', 'minfde_all', 'minade_moving', 'minfde_moving'])
        for sc, v in sorted(per_scene.items()):
            a = np.array([x[0] for x in v]); f = np.array([x[1] for x in v])
            mv = np.array([x[2] for x in v]) >= MOVING_MIN
            ma = np.array([x[3] for x in v]); mf = np.array([x[4] for x in v])
            w.writerow([args.fold, args.variant, args.seed, sc, len(v), int(mv.sum()),
                        f'{a.mean():.5f}', f'{f.mean():.5f}',
                        f'{a[mv].mean():.5f}' if mv.any() else '',
                        f'{f[mv].mean():.5f}' if mv.any() else '',
                        f'{gate:.5f}',
                        # Con K=1 estas son iguales a ade_all/fde_all por
                        # definición (el mínimo sobre un solo modo). Se escriben
                        # igual para que el esquema sea uno solo.
                        f'{ma.mean():.5f}', f'{mf.mean():.5f}',
                        f'{ma[mv].mean():.5f}' if mv.any() else '',
                        f'{mf[mv].mean():.5f}' if mv.any() else ''])
    print(f'[eval] {args.variant} seed {args.seed}: '
          + ', '.join(f'{sc} n={len(v)}' for sc, v in sorted(per_scene.items())))


if __name__ == '__main__':
    main()
