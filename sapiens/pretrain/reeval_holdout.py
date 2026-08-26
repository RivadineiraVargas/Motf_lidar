"""
reeval_holdout.py — re-evalúa los checkpoints ya entrenados sobre TODOS los frames
del held-out, separando objetos móviles de parados.

Responde a dos hallazgos de la auditoría del 23/08:

B2 — el entrenamiento evaluaba SOLO t=10 de cada escena retenida
     (`train_decoder_mini.py`: `build_sample(usc, 10)`), o sea 5 muestras por fold.
     Pero las etiquetas existen para 91 frames y las features del encoder están
     cacheadas para los 11 sweeps, así que t=0..10 es evaluable GRATIS: a 3 s una
     muestra en t necesita etiquetas hasta t+30, disponible para todo t<=10.
     Da ~11x mas datos de test sin tocar la GPU para entrenar.

B5 — el 72-75% de los objetos se desplaza <1 m. Para ellos la velocidad constante
     ya es casi perfecta y ningún modelo puede mejorarla, solo empeorarla:
     promediar sobre esa población COMPRIME cualquier efecto hacia cero.

IMPORTANTE (advertencia de la auditoría): los 11 frames de una escena están
CORRELACIONADOS. Este script emite una fila por (fold, seed, arch, escena) con el
promedio sobre los frames de esa escena; el análisis posterior debe agregar por
ESCENA, nunca tratar las 55 muestras como independientes. Tratarlas como
independientes fabricaría significancia falsa — el error que este proyecto ya
cometió cuatro veces.

LO QUE ESTO NO ARREGLA: el checkpoint que se carga fue elegido por su ADE en
t=10 del propio held-out (hallazgo H1: la selección usa el test). Re-evaluar con
más frames reduce la varianza de la MEDICIÓN, pero no elimina ese sesgo de
selección. Para eso hace falta rediseñar el split (H1), que es otro trabajo.
"""
import argparse, csv, os
import numpy as np
import torch
import train_decoder_mini as tdm
from cross_validate_decoder import make_folds
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpretrain.registry import MODELS

MOVING_MIN = 1.0        # m de desplazamiento GT para contar como móvil

ARCH_CLS = {'wayformer': tdm.MiniWayformerDecoder,
            'baseline': tdm.MiniBaseline,
            'wayformer_gated': tdm.MiniWayformerGated}


def build_model(arch, dev):
    if arch.startswith('gatefix'):
        m = tdm.MiniWayformerGated(gate_init=float(arch[len('gatefix'):]))
    else:
        m = ARCH_CLS[arch]()
    return m.to(dev)


def ckpt_path(run_dir, arch):
    suf = '' if arch == 'wayformer' else f'_{arch}'
    return f'{run_dir}/decoder_mini{suf}.pth'


def eval_scene(model, lat, scene, dev):
    """Promedia sobre TODOS los t evaluables de una escena. Devuelve
    (ade_todos, ade_moviles, n_obj, n_moviles) — el ADE por objeto, no por frame,
    para que un frame con muchos objetos pese lo que corresponde."""
    d_all, d_mov = [], []
    for t in range(11):
        s = tdm.build_sample(scene, t)
        if s['n'] == 0:
            continue
        with torch.no_grad():
            traj, _ = model(lat[t], (s['feat'] / tdm.SCALE).to(dev), s['n'])
        pred = (s['cv'] + traj[0, :s['n']].cpu() * tdm.SCALE).numpy()
        gt = s['gt'][:s['n']].numpy()
        m = s['wpm'][:s['n']].numpy() > 0
        for i in range(s['n']):
            k = np.where(m[i])[0]
            if not len(k):
                continue
            ade = float(np.linalg.norm(pred[i][k] - gt[i][k], axis=-1).mean())
            d_all.append(ade)
            if np.linalg.norm(gt[i][k[-1]]) >= MOVING_MIN:
                d_mov.append(ade)
    return (float(np.mean(d_all)) if d_all else np.nan,
            float(np.mean(d_mov)) if d_mov else np.nan,
            len(d_all), len(d_mov))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--seeds', nargs='+', type=int, default=list(range(8)))
    ap.add_argument('--n-wp', type=int, default=6)
    ap.add_argument('--out', default='work_dirs/reeval/reeval_holdout.csv')
    args = ap.parse_args()
    tdm.N_WP = args.n_wp
    dev = 'cuda'
    init_default_scope('mmpretrain')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    folds = make_folds()[1]
    src = {0: 'work_dirs/horizon_domain',
           **{f: f'work_dirs/horizon_fold{f}' for f in (1, 2, 3, 4)}}

    new = not os.path.exists(args.out)
    with open(args.out, 'a', newline='') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(['fold', 'seed', 'arch', 'scene',
                        'ade_all', 'ade_moving', 'n_obj', 'n_moving'])
        for fi in args.folds:
            # features del encoder de dominio de ESTE fold (usarlo en otro = fuga)
            cfg = Config.fromfile(
                f'configs/sapiens_mae/lidar/config_rangeview_rect_fold{fi}.py')
            mae = MODELS.build({**cfg.model, 'data_preprocessor': cfg.data_preprocessor})
            sd = torch.load(f'work_dirs/rv_rect_fold{fi}/epoch_1000.pth',
                            map_location='cpu').get('state_dict')
            mae.load_state_dict(sd, strict=False)
            enc = mae.backbone.to(dev)
            enc.eval()          # MAEViT.eval() devuelve None: no encadenar
            lat = {sc: tdm.encode_sweeps(enc, sc, range(11), dev,
                                         cache_dir=f'work_dirs/cache_fold{fi}_domain')
                   for sc in folds[fi]}
            del enc; torch.cuda.empty_cache()

            archs = sorted({d.split('_h3s')[0] for d in os.listdir(src[fi])
                            if '_h3s_f' in d})
            for seed in args.seeds:
                for arch in archs:
                    run = f'{src[fi]}/{arch}_h3s_f{fi}s{seed}'
                    p = ckpt_path(run, arch)
                    if not os.path.exists(p):
                        continue
                    model = build_model(arch, dev)
                    model.load_state_dict(torch.load(p, map_location=dev), strict=False)
                    model.eval()
                    for sc in folds[fi]:
                        w.writerow([fi, seed, arch, sc, *[
                            f'{v:.5f}' if isinstance(v, float) else v
                            for v in eval_scene(model, lat[sc], sc, dev)]])
                    fh.flush()
                print(f'[fold {fi}] seed {seed} listo ({len(archs)} arqs)')
            del lat; torch.cuda.empty_cache()
    print(f'\n-> {args.out}')


if __name__ == '__main__':
    main()
