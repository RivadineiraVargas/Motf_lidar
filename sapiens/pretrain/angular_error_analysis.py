"""
angular_error_analysis.py — descompone el error del decoder en DIRECCION y
MAGNITUD, por fold y por semilla.

Motivacion: la CV de 5 folds (exp. 11) mostro que el beneficio de la escena
depende del split (-20% en el fold 0, +40% en el fold 3) pero no decia POR QUE.
El ADE mezcla dos errores distintos: apuntar mal (direccion) y estimar mal
cuanto avanza (magnitud). Separarlos identifica el modo de fallo.

Diseño: por cada fold, se codifican UNA VEZ las 5 escenas retenidas con el
encoder de dominio de ESE fold (nunca las vio) y se reusan las features para las
8 semillas del decoder -> el costo lo domina el encoder, no el barrido.

Metricas (solo objetos MOVILES, |despl GT| >= 1m: en los parados la direccion
no esta definida y son ~72-75% de los objetos):
  - error angular entre el desplazamiento predicho y el real, al ultimo waypoint
  - fraccion de errores gruesos (>45 grados) = fallos de direccion
  - sesgo de magnitud = |pred| - |gt|

Salida: work_dirs/angular/angular_results.csv (una fila por fold/semilla/arq).
"""
import argparse, csv, os
import numpy as np
import torch
import train_decoder_mini as tdm
from cross_validate_decoder import make_folds
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpretrain.registry import MODELS

MOVING_MIN = 1.0        # m: umbral para considerar que el objeto se mueve
GROSS_DEG = 45.0        # grados: umbral de "error grueso" de direccion


def build_encoder(fold, dev):
    cfg = Config.fromfile(f'configs/sapiens_mae/lidar/config_rangeview_rect_fold{fold}.py')
    mae = MODELS.build({**cfg.model, 'data_preprocessor': cfg.data_preprocessor})
    sd = torch.load(f'work_dirs/rv_rect_fold{fold}/epoch_1000.pth',
                    map_location='cpu').get('state_dict')
    mae.load_state_dict(sd, strict=False)
    enc = mae.backbone.to(dev)
    enc.eval()          # OJO: MAEViT.eval() devuelve None en este fork, no encadenar
    return enc


def metrics(model, lat, scenes_samples, dev):
    ang, dmag = [], []
    for (t, s) in scenes_samples:
        with torch.no_grad():
            traj, _ = model(lat[t], (s['feat'] / tdm.SCALE).to(dev), s['n'])
        pred = (s['cv'] + traj[0, :s['n']].cpu() * tdm.SCALE).numpy()
        gt = s['gt'][:s['n']].numpy()
        m = s['wpm'][:s['n']].numpy() > 0
        for i in range(s['n']):
            k = np.where(m[i])[0]
            if not len(k):
                continue
            g, p = gt[i][k[-1]], pred[i][k[-1]]
            ng = np.linalg.norm(g)
            if ng < MOVING_MIN:
                continue
            c = np.dot(g, p) / (ng * np.linalg.norm(p) + 1e-9)
            ang.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
            dmag.append(np.linalg.norm(p) - ng)
    return np.array(ang), np.array(dmag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--seeds', nargs='+', type=int, default=list(range(8)))
    ap.add_argument('--n-wp', type=int, default=6)          # 3s
    ap.add_argument('--out', default='work_dirs/angular')
    args = ap.parse_args()
    tdm.N_WP = args.n_wp                 # ANTES de construir samples y modelos
    dev = 'cuda'
    init_default_scope('mmpretrain')
    os.makedirs(args.out, exist_ok=True)
    folds = make_folds()[1]
    src = {0: 'work_dirs/horizon_domain',
           **{f: f'work_dirs/horizon_fold{f}' for f in (1, 2, 3, 4)}}

    csv_path = f'{args.out}/angular_results.csv'
    new = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(['fold', 'seed', 'arch', 'n_moving', 'ang_median',
                        'ang_mean', 'frac_gross', 'mag_bias'])
        for fi in args.folds:
            enc = build_encoder(fi, dev)
            # features + samples: se calculan una vez por fold y se reusan
            lat_all, samples = {}, {}
            for sc in folds[fi]:
                lat_all[sc] = tdm.encode_sweeps(enc, sc, range(11), dev)
                samples[sc] = [(t, tdm.build_sample(sc, t)) for t in range(11)]
                samples[sc] = [(t, s) for t, s in samples[sc] if s['n'] > 0]
            del enc
            torch.cuda.empty_cache()
            for seed in args.seeds:
                for arch, cls, fname in (
                        ('wayformer', tdm.MiniWayformerDecoder, 'decoder_mini.pth'),
                        ('baseline', tdm.MiniBaseline, 'decoder_mini_baseline.pth')):
                    p = f'{src[fi]}/{arch}_h3s_f{fi}s{seed}/{fname}'
                    if not os.path.exists(p):
                        print(f'  [falta] {p}')
                        continue
                    model = cls().to(dev)
                    model.load_state_dict(torch.load(p, map_location=dev), strict=False)
                    model.eval()
                    A, D = [], []
                    for sc in folds[fi]:
                        a, d = metrics(model, lat_all[sc], samples[sc], dev)
                        A.append(a); D.append(d)
                    A = np.concatenate(A); D = np.concatenate(D)
                    w.writerow([fi, seed, arch, len(A), f'{np.median(A):.3f}',
                                f'{A.mean():.3f}', f'{(A > GROSS_DEG).mean():.4f}',
                                f'{D.mean():.3f}'])
                    fh.flush()
                print(f'[fold {fi}] seed {seed} listo')
            del lat_all, samples
            torch.cuda.empty_cache()
    print(f'\n-> {csv_path}')


if __name__ == '__main__':
    main()
