"""
export_fase1_global.py — lleva las predicciones del track FASE 1 al simulador.

POR QUE EXISTE. Habia dos exportadores y ninguno sirve para los experimentos
actuales (regla 3 del proyecto: no mezclar tracks):

  export_decoder_mini_global.py -> track decoder_mini, CONGELADO. Construye
      MiniWayformerDecoder/MiniBaseline, horizonte 16 wp x 0,5 s = 8 s.
  export_predictions_global.py  -> track waymo_10, MUERTO. WAYMO_ROOT=waymo_10,
      PRED_LEN=5, una escena hardcodeada, checkpoints epoch_500 de la era overfit.

Este exporta los modelos de Fase 1: TrajectoryModelWithAttention y
BaselineTrajectoryModel sobre waymo_clean, history_len=5 / pred_len=30 (3 s),
con las escenas de VALIDACION del fold y soporte para pad_dim y num_modes.

EL PELIGRO QUE EVITA. mmengine.runner.load_checkpoint usa strict=False POR
DEFAULT. Si el modelo se construye con la geometria equivocada (pad_dim o
num_modes distintos de los del checkpoint), carga en silencio con capas
aleatorias y dibuja trayectorias plausibles y FALSAS. Ya paso una vez: el ADE
se fue de 2,84 a 22,14 sin ningun aviso (ver el comentario en baseline_model.py).
Aca la carga es ESTRICTA y falla fuerte.

Formato de salida (predictions_global.txt), una linea por punto:
    <scene> <obj_id> <kind> <t> <x> <y> <z>        coords GLOBALES
    kind: 0=historico  1=futuro real  2=prediccion (modo mas probable)
          3.. = modos alternativos, solo con --modos todos

UNA VENTANA POR OBJETO. El dataset genera varias ventanas por objeto (distintos
t_start). Exportarlas todas llena el BEV de lineas superpuestas del MISMO auto.
Por default se exporta UNA (--ventana 0). Esto es lo que pidio Claudine: menos
lineas para poder juzgar alguna.

Uso:
    conda run -n sapiens_gpu python export_fase1_global.py \
        --cfg configs/sapiens_mae/lidar/noclip_base_fold0.py \
        --ckpt work_dirs/initscale/base_pad64_f0s0/epoch_100.pth \
        --pad-dim 64 --fold 0
"""
import argparse, os, re, sys
import numpy as np
import torch


def escenas_val_del_fold(d, fold):
    """Lee las escenas de validacion RETENIDAS del fold, igual que los run_*.sh."""
    t = open(f'{d}/f1cv_mae_fold{fold}.py').read()
    m = re.search(r'val RETENIDA del fold \d+: \[(.*?)\]', t)
    if not m:
        raise SystemExit(f'no encuentro las escenas de validacion del fold {fold}')
    return re.findall(r"'([0-9a-f]{16})'", m.group(1))


def cargar_pose(data_root, scene, frame):
    p = f'{data_root}/poses/{scene}/{frame}.txt'
    if not os.path.exists(p):
        return None
    v = np.loadtxt(p).reshape(-1)
    if v.size != 16:
        return None
    return v.reshape(4, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True, help='config del fold (define el dataset)')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--fold', type=int, default=None,
                    help='si se da, exporta solo las escenas de validacion del fold')
    ap.add_argument('--scenes', nargs='+', default=None,
                    help='escenas explicitas (tienen prioridad sobre --fold)')
    ap.add_argument('--pad-dim', type=int, default=None,
                    help='default: se DEDUCE del checkpoint')
    ap.add_argument('--num-modes', type=int, default=None,
                    help='default: se DEDUCE del checkpoint')
    ap.add_argument('--ventana', type=int, default=0,
                    help='cual ventana de cada objeto exportar (default 0)')
    ap.add_argument('--modos', choices=['mas-probable', 'todos'], default='mas-probable')
    ap.add_argument('--eval-windows', type=int, default=7,
                    help='igual que eval_fase1_seeds.py, para que las ventanas coincidan')
    ap.add_argument('--txt', default='/home/lcad/lidar_sweep_viewer/predictions_global.txt')
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmpretrain.registry import MODELS
    from mmpretrain.datasets import TrajectoryDataset
    init_default_scope('mmpretrain')

    cfg = Config.fromfile(args.cfg)
    sd = torch.load(args.ckpt, map_location='cpu')
    sd = sd.get('state_dict', sd)

    # GEOMETRIA DEDUCIDA DEL CHECKPOINT. Pasar --pad-dim/--num-modes a mano es
    # la forma mas facil de dibujar basura convincente, asi que por default se
    # leen de los tensores mismos:
    #   mode_head.weight  (K, hidden)      -> num_modes; si no esta, K=1
    #   decoder.0.weight  (hidden, in)     -> pad_dim = in - history_len*3
    hl = cfg.train_dataloader.dataset['history_len']
    if args.num_modes is not None:
        nm = args.num_modes
    else:
        nm = sd['mode_head.weight'].shape[0] if 'mode_head.weight' in sd else 1
    if args.pad_dim is not None:
        pd = args.pad_dim
    else:
        w0 = sd.get('decoder.0.weight')
        pd = int(w0.shape[1]) - hl * 3 if w0 is not None else 0
        if pd < 0:
            raise SystemExit(f'decoder.0 espera {w0.shape[1]} entradas pero la '
                             f'historia son {hl*3}: el config no corresponde al ckpt')
    origen = 'deducido del ckpt' if (args.pad_dim is None and args.num_modes is None) else 'parcialmente manual'
    print(f'[geometria] pad_dim={pd}  num_modes={nm}  ({origen})')

    # Solo el baseline tiene pad_dim; en el modelo con escena el ancho extra es
    # scene_dim, que ya viene del config.
    override = {'model.num_modes': nm}
    if cfg.model['type'] == 'BaselineTrajectoryModel':
        override['model.pad_dim'] = pd
    cfg.merge_from_dict(override)

    model = MODELS.build(cfg.model)
    # ESTRICTO a proposito: ver el encabezado. Hay DOS formas de que el
    # checkpoint no corresponda y las dos tienen que fallar fuerte:
    #   (a) desajuste de FORMA -> PyTorch lanza RuntimeError incluso con
    #       strict=False. Se envuelve para que el mensaje diga que revisar.
    #   (b) claves que faltan o sobran -> eso strict=False lo devuelve en
    #       silencio, asi que se revisa a mano.
    PISTA = ('\nEl checkpoint NO corresponde al modelo construido. Revisa '
             '--pad-dim y --num-modes\ncontra como se entreno esa corrida '
             '(el nombre del work-dir suele decirlo).')
    try:
        faltan, sobran = model.load_state_dict(sd, strict=False)
    except RuntimeError as e:
        raise SystemExit(f'{e}{PISTA}')
    if faltan or sobran:
        raise SystemExit(
            f'  faltan en el checkpoint : {list(faltan)[:6]}\n'
            f'  sobran en el checkpoint : {list(sobran)[:6]}{PISTA}')
    model.eval()

    dcfg = cfg.train_dataloader.dataset
    data_root = dcfg['data_root']
    H, P = dcfg['history_len'], dcfg['pred_len']
    con_escena = cfg.model['type'] == 'TrajectoryModelWithAttention'
    K = getattr(model, 'num_modes', 1)
    print(f'[cfg] {cfg.model["type"]}  history={H} pred={P} K={K} '
          f'pad_dim={getattr(model, "pad_dim", 0)}  escena={"si" if con_escena else "no"}')

    D = os.path.dirname(args.cfg)
    escenas = args.scenes
    if not escenas and args.fold is not None:
        escenas = escenas_val_del_fold(D, args.fold)
    if not escenas:
        escenas = list(cfg.get('val_scenes', []) or [])
    if escenas:
        print(f'[escenas] {len(escenas)}: {" ".join(escenas)}')

    # DATASET DE VALIDACION, igual que eval_fase1_seeds.py:104-108. El config
    # trae las escenas de ENTRENAMIENTO del fold y augment=True; las dos cosas
    # hay que darlas vuelta o se exportan predicciones sobre datos rotados de
    # escenas que el modelo si vio.
    kw = {k: v for k, v in dcfg.items() if k != 'type'}
    kw['augment'] = False
    kw['eval_windows'] = args.eval_windows
    if escenas:
        kw['scenes'] = list(escenas)
    ds = TrajectoryDataset(**kw)

    lineas, n_obj, vistos = [], 0, {}
    for i in range(len(ds)):
        meta = ds.data_list[i]
        scene = meta['scene_name']
        if escenas and scene not in escenas:
            continue
        oid = str(meta['object_id'])
        # una ventana por objeto (ver encabezado)
        vistos[(scene, oid)] = vistos.get((scene, oid), -1) + 1
        if vistos[(scene, oid)] != args.ventana:
            continue

        d = ds[i]
        f0 = meta['frame0']
        centers = np.array(meta['centers'])          # marco del SENSOR de cada t
        std = d['norm_std'].numpy()
        mean = d['norm_mean'].numpy()
        ref = d['ref_center'].numpy()

        poses = {t: cargar_pose(data_root, scene, f0 + t) for t in range(H + P)}
        if any(poses[t] is None for t in range(H + P)):
            continue

        with torch.no_grad():
            if args.modos == 'todos' and K > 1:
                if con_escena:
                    modos, probs = model(d['inputs'].unsqueeze(0),
                                         d['obj_history_flat'].unsqueeze(0),
                                         mode='predict_multi')
                else:
                    modos, probs = model(d['obj_history_flat'].unsqueeze(0),
                                         mode='predict_multi')
                hip = modos.squeeze(0).view(K, P, 3).numpy()
                orden = np.argsort(-probs.squeeze(0).numpy())   # mas probable primero
                hip = hip[orden]
            else:
                if con_escena:
                    pf = model(d['inputs'].unsqueeze(0),
                               d['obj_history_flat'].unsqueeze(0), mode='predict')
                else:
                    pf = model(d['obj_history_flat'].unsqueeze(0), mode='predict')
                hip = pf.squeeze(0).view(1, P, 3).numpy()

        def glob(t, c_sensor):
            return (poses[t] @ np.append(c_sensor, 1.0))[:3]

        for t in range(H):                                   # historico
            g = glob(t, centers[t])
            lineas.append(f'{scene} {oid} 0 {t} {g[0]:.4f} {g[1]:.4f} {g[2]:.4f}')
        for k, t in enumerate(range(H, H + P)):              # futuro real
            g = glob(t, centers[t])
            lineas.append(f'{scene} {oid} 1 {t} {g[0]:.4f} {g[1]:.4f} {g[2]:.4f}')
        for m in range(hip.shape[0]):                        # predicciones
            kind = 2 if m == 0 else 2 + m
            for k, t in enumerate(range(H, H + P)):
                g = glob(t, hip[m, k] * std + mean + ref)
                lineas.append(f'{scene} {oid} {kind} {t} {g[0]:.4f} {g[1]:.4f} {g[2]:.4f}')
        n_obj += 1

    with open(args.txt, 'w') as f:
        f.write('\n'.join(lineas) + '\n')
    print(f'OK: {n_obj} objetos, {len(lineas)} puntos -> {args.txt}')
    if args.modos == 'todos' and K > 1:
        print(f'     kind 2 = modo mas probable, kind 3..{1+K} = alternativos')


if __name__ == '__main__':
    main()
