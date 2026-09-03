# curva_overfit10.py — la curva de generalización de la prueba de 10 sweeps.
#
# POR QUE EXISTE. El item 5 del checklist de Claudine se da por cumplido con
# "loss 2.72 -> 0.055", que es PERDIDA DE ENTRENAMIENTO sobre los mismos 10
# sweeps: solo prueba que el modelo puede memorizarlos. La unica medicion fuera
# de train (item 11: 3.52 sin entrenar / 3.39 con 10 sweeps) se tomo A MANO y en
# la EPOCA 6000. Pero el item 6, con 100 sweeps, encontro que "la generalizacion
# pica ~ep1000 y luego memoriza": si eso pasa tambien con 10, ese 3.39 es el
# modelo ya memorizado, no el mejor. No se podia verificar porque el config
# original tiene max_keep_ckpts=2 y en disco quedaron solo epoch_5500 y 6000.
#
# Este script convierte "10 sweeps OK" de una afirmacion en una medicion: recorre
# TODOS los checkpoints y mide la perdida enmascarada en tres poblaciones.
#
# LAS TRES POBLACIONES (misma idea que diagnostico_encoder_mae.py, que ya sirvio
# para el experimento 21):
#   train        — las 10 imagenes que optimizo. Si baja, memoriza.
#   val_intra    — 2a81 sweep 9, el UNICO sweep de la escena de train que no vio.
#                  Separa "memorizo estas 10 imagenes" de "aprendio esta escena".
#   val_escenas  — 5 escenas nunca vistas x 11 sweeps = 55 imagenes. Es lo que de
#                  verdad importa: si cruza entre ESCENAS. Antes esto era n=1.
#
# DOS REFERENCIAS sin las cuales los numeros no significan nada: el modelo SIN
# ENTRENAR (mismo config, pesos iniciales) y el mejor checkpoint. La comparacion
# contra el no-entrenado es la que el item 11 ya usaba (3.52).
#
# MASCARAS PAREADAS. La perdida depende de que parches se enmascaren: se fija la
# semilla antes de cada forward, asi que la mascara nº k es IDENTICA en las tres
# poblaciones y en todos los checkpoints. Las diferencias no pueden venir del
# sorteo. Se promedian --mascaras sorteos por imagen.
#
# No entrena ni escribe en work_dirs mas que su propio CSV.
#
# Uso:  conda activate sapiens_gpu
#       cd sapiens/pretrain && python curva_overfit10.py
import argparse
import csv
import glob
import os
import re
import statistics as st

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmcv.transforms import Compose
from mmpretrain.registry import MODELS

B = '/home/lcad/lidar_sweep_viewer/waymo_clean/range_png_rect'
POBLACIONES = {
    'train':       f'{B}/ov10_train',
    'val_intra':   f'{B}/ov10_val_intra',
    'val_escenas': f'{B}/ov10_val_escenas',
}


def cargar(pipe, cfg, path, dev):
    """Devuelve el tensor normalizado igual que lo hace eval_rect_loss.py.
    PackInputs entrega BGR y el preprocessor tiene to_rgb=True, asi que hay que
    invertir los canales a mano: sin eso se mide sobre una imagen distinta de la
    que vio el entrenamiento."""
    data = pipe(dict(img_path=path))
    img = data['inputs'].unsqueeze(0).float().to(dev)
    mean = torch.tensor(cfg.data_preprocessor['mean']).view(1, 3, 1, 1).to(dev)
    std = torch.tensor(cfg.data_preprocessor['std']).view(1, 3, 1, 1).to(dev)
    return (img[:, [2, 1, 0]] - mean) / std


@torch.no_grad()
def perdidas(modelo, imgs, n_mascaras):
    out = []
    for img in imgs:
        acc = []
        for s in range(n_mascaras):
            torch.manual_seed(s)
            torch.cuda.manual_seed_all(s)
            latent, mask, ids_restore = modelo.backbone(img)
            pred = modelo.neck(latent, ids_restore)
            loss = modelo.head.loss(pred, img, mask)
            acc.append(float(loss['loss'] if isinstance(loss, dict) else loss))
        out.append(sum(acc) / len(acc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='configs/sapiens_mae/lidar/rect_overfit10_val.py')
    ap.add_argument('--work-dir', default='work_dirs/rect_ov10_val')
    ap.add_argument('--mascaras', type=int, default=4)
    ap.add_argument('--cada', type=int, default=1,
                    help='evaluar 1 de cada N checkpoints (para una pasada rapida)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    init_default_scope('mmpretrain')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = Config.fromfile(args.cfg)
    cfg.model['data_preprocessor'] = cfg.data_preprocessor
    pipe = Compose(cfg.train_pipeline)

    # Las imagenes se cargan UNA vez y se reusan en todos los checkpoints: ademas
    # de ahorrar, garantiza que todos vean exactamente los mismos tensores.
    imgs = {}
    for k, d in POBLACIONES.items():
        ps = sorted(glob.glob(f'{d}/*.png'))
        if not ps:
            raise SystemExit(f'{d} no tiene PNG. Generarlos con '
                             f'utilities/make_rect_png_scenes.py (ver la cabecera '
                             f'de configs/.../rect_overfit10_val.py).')
        imgs[k] = [cargar(pipe, cfg, p, dev) for p in ps]
        print(f'  {k:12s} {len(ps):3d} imágenes')

    cks = sorted(glob.glob(f'{args.work_dir}/epoch_*.pth'),
                 key=lambda p: int(re.search(r'epoch_(\d+)', p).group(1)))
    cks = cks[::args.cada]
    print(f'\n{len(cks)} checkpoints · {args.mascaras} máscaras por imagen '
          f'(semillas 0..{args.mascaras - 1}, pareadas)\n')

    out = args.out or f'{args.work_dir}/curva_overfit10.csv'
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    filas = []

    for etiqueta, ruta in [('sin_entrenar', None)] + [(re.search(r'epoch_(\d+)', c).group(1), c)
                                                     for c in cks]:
        modelo = MODELS.build(cfg.model)
        if ruta is None:
            modelo.init_weights()
        else:
            sd = torch.load(ruta, map_location='cpu')
            sd = sd.get('state_dict', sd)
            inc = modelo.load_state_dict(sd, strict=False)
            # Mismo guard que eval_fase1_seeds.py (hallazgo 10): strict=False
            # acepta en silencio un checkpoint que no case en NADA y deja pesos
            # aleatorios. Acá eso se leería como "el modelo no aprendió".
            cargadas = len(sd) - len(inc.unexpected_keys)
            if cargadas <= 0:
                raise SystemExit(f'{ruta} no aportó ni un peso a {args.cfg}.')
        modelo = modelo.to(dev).eval()

        fila = {'epoca': etiqueta}
        for k in POBLACIONES:
            v = perdidas(modelo, imgs[k], args.mascaras)
            fila[k] = st.mean(v)
            fila[k + '_sd'] = st.stdev(v) if len(v) > 1 else 0.0
        filas.append(fila)
        print(f'  época {etiqueta:>12s}  train {fila["train"]:.4f}  '
              f'val_intra {fila["val_intra"]:.4f}  '
              f'val_escenas {fila["val_escenas"]:.4f} ± {fila["val_escenas_sd"]:.4f}')
        del modelo
        torch.cuda.empty_cache()

    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)

    ent = [f for f in filas if f['epoca'] != 'sin_entrenar']
    base = next(f for f in filas if f['epoca'] == 'sin_entrenar')
    if ent:
        mejor = min(ent, key=lambda f: f['val_escenas'])
        ultimo = ent[-1]
        print(f'\n=== LA CURVA, en {out} ===')
        print(f'  sin entrenar            val_escenas {base["val_escenas"]:.4f}')
        print(f'  MEJOR (época {mejor["epoca"]:>5s})      val_escenas {mejor["val_escenas"]:.4f}'
              f'   ({100 * (1 - mejor["val_escenas"] / base["val_escenas"]):+.1f}% vs sin entrenar)')
        print(f'  última (época {ultimo["epoca"]:>4s})     val_escenas {ultimo["val_escenas"]:.4f}'
              f'   ({100 * (1 - ultimo["val_escenas"] / base["val_escenas"]):+.1f}% vs sin entrenar)')
        if mejor['epoca'] != ultimo['epoca']:
            print(f'\n  El pico NO está al final: medir en la última época subestima '
                  f'el resultado en {100 * (ultimo["val_escenas"] / mejor["val_escenas"] - 1):.1f}%.')


if __name__ == '__main__':
    main()
