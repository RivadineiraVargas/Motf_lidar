# diagnostico_encoder_mae.py — ¿los encoders del CV generalizan, o memorizaron?
#
# LA PREGUNTA. Los cinco encoders de work_dirs/f1cv se pre-entrenaron con
# `max_windows=1`: UNA ventana por escena, 8 escenas de train = 8 muestras, mil
# épocas. Y ningún f1cv_mae_fold*.py declara val_dataloader, así que la pérdida
# 1.29 -> 0.019-0.087 segun el fold (n=5) de los logs es de ENTRENAMIENTO sobre
# esas ocho muestras. Nunca se
# midió la reconstrucción fuera de ellas.
#
# Importa porque toda la conclusión "la escena no aporta" depende de que la
# escena que ve el decoder signifique algo. Si el encoder memorizó ocho ventanas,
# en validación entrega ruido, y que el gate converja a 0.0042 deja de ser "el
# modelo descarta la escena" para ser "el modelo descarta ruido": la condición no
# quedaría refutada sino SIN PROBAR (el mismo error del experimento 17).
#
# TRES POBLACIONES que separan las dos hipótesis:
#   train_vistas  — las 8 ventanas exactas que el MAE optimizó
#   train_nuevas  — las otras 6 ventanas/escena de las MISMAS escenas (t0=1..6)
#   val           — las 7 ventanas de cada una de las 2 escenas RETENIDAS del fold
# Lectura:
#   vistas << nuevas ~ val   -> memorizó las ventanas exactas
#   vistas ~ nuevas << val   -> aprendió esas escenas, no generaliza a otras
#   vistas ~ nuevas ~ val    -> generaliza; la preocupación era infundada
#
# DOS REFERENCIAS sin las cuales los números no significan nada:
#   aleatorio — el MISMO modelo sin entrenar. Es el "no aprendió nada".
#   trivial   — la pérdida de predecir 0 en todo (MSE contra ocupación es la
#               fracción de vóxeles ocupados). Es el "no hace falta un ViT".
#
# MÁSCARAS PAREADAS. La pérdida MAE depende de qué vóxeles se enmascaren. Se fija
# la semilla justo antes de cada forward, así que la máscara nº k es IDÉNTICA en
# las tres poblaciones y en los dos modelos: la comparación es pareada y las
# diferencias no pueden venir del sorteo. Se promedian K máscaras por muestra.
#
# MODO TRAIN, A PROPÓSITO. MAEViT4D.forward solo enmascara bajo `if self.training`;
# en eval devuelve mask=ceros y la pérdida sale 0/1e-6 = 0 para cualquier modelo
# (comprobado: entrenado y aleatorio daban 0.0000 idénticos). Para medir
# reconstrucción hay que enmascarar, así que el modelo se deja en train() y se
# verifica antes que ningún módulo cambie de comportamiento por eso: sin dropout
# activo y sin BatchNorm, solo LayerNorm, que es idéntica en los dos modos.
# El gradiente está apagado por @torch.no_grad y no hay optimizador: no entrena.
#
# No entrena ni escribe nada en work_dirs: solo lee checkpoints y hace forwards.
#
# Uso:  conda activate sapiens_gpu
#       cd sapiens/pretrain && python diagnostico_encoder_mae.py
import argparse
import gc
import re
import statistics as st

import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.registry import init_default_scope

import mmpretrain.datasets.lidar_sequence          # noqa: F401  (registra el dataset)
import mmpretrain.models.backbones.mae_vit_4d      # noqa: F401
import mmpretrain.models.selfsup.mae_4d            # noqa: F401
import mmpretrain.models.heads.mae_head_4d         # noqa: F401
from mmpretrain.registry import DATASETS, MODELS

D = 'configs/sapiens_mae/lidar'
WD = 'work_dirs/f1cv'


def escenas_val(cfg_path):
    """Las escenas retenidas se leen del comentario del config, que es la fuente
    única de la definición del fold (mismo criterio que run_noclip_cv.sh)."""
    t = open(cfg_path).read()
    m = re.search(r'val RETENIDA del fold \d+: \[(.*?)\]', t)
    if not m:
        raise SystemExit(f'{cfg_path}: no pude leer las escenas de validación')
    return re.findall(r"'([0-9a-f]{16})'", m.group(1))


def construir_pobl(cfg, escenas, max_windows, t0_desde=0):
    """Devuelve los tensores de entrada ya voxelizados. Se materializan una sola
    vez y se reusan en los dos modelos: además de ahorrar, garantiza que
    entrenado y aleatorio ven EXACTAMENTE los mismos datos.

    No se usa len(ds) ni ds[i] sobre el dataset filtrado: mmengine serializa
    data_list en __init__ (serialize_data=True) y __len__ sigue contando los
    ítems originales, así que filtrar data_list a mano deja el largo mintiendo.
    Se recorre data_list, que es lo que este dataset lee en __getitem__."""
    d = dict(cfg.train_dataloader.dataset)
    d['scenes'] = list(escenas)
    d['max_windows'] = max_windows
    ds = DATASETS.build(d)
    idx = [i for i, it in enumerate(ds.data_list) if it['t0'] >= t0_desde]
    return [ds[i]['inputs'] for i in idx]


def verificar_modo_train(modelo):
    """train() solo es seguro si nada más depende del modo. Si aparece dropout
    activo o una capa con estadísticas móviles, las pérdidas dejarían de ser
    comparables con las del entrenamiento y hay que enterarse acá, no después."""
    malos = []
    for n, m in modelo.named_modules():
        if isinstance(m, nn.modules.dropout._DropoutNd) and m.p > 0:
            malos.append(f'{n} (dropout p={m.p})')
        if isinstance(m, nn.modules.batchnorm._NormBase):
            malos.append(f'{n} ({type(m).__name__})')
    if malos:
        raise SystemExit('train() no es seguro acá: ' + ', '.join(malos[:5]))


@torch.no_grad()
def perdidas(modelo, xs, n_mascaras, dev):
    """Devuelve la lista de pérdidas por muestra, cada una promediada sobre
    n_mascaras sorteos con semillas 0..n-1 (las mismas para toda población)."""
    out = []
    for x in xs:
        x = x.unsqueeze(0).to(dev)
        acc = []
        for s in range(n_mascaras):
            torch.manual_seed(s)
            torch.cuda.manual_seed_all(s)
            acc.append(float(modelo.loss(x)['loss']))
        out.append(sum(acc) / len(acc))
    return out


def trivial(xs):
    """MSE de predecir 0 en todos los vóxeles. Con ocupación en {0,1} es la
    fracción ocupada. No depende de la máscara: es el mismo valor en promedio."""
    return [float((x ** 2).mean()) for x in xs]


def resumen(v):
    if not v:
        return 'n=0'
    if len(v) == 1:
        return f'{v[0]:.4f} (n=1)'
    return f'{st.mean(v):.4f} ± {st.stdev(v):.4f} (n={len(v)})'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--mascaras', type=int, default=8)
    ap.add_argument('--epoch', default='epoch_1000.pth')
    args = ap.parse_args()

    init_default_scope('mmpretrain')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'dispositivo: {dev} · {args.mascaras} máscaras por muestra (semillas 0..'
          f'{args.mascaras - 1}, pareadas entre poblaciones y modelos)\n')

    tabla = {}
    for F in args.folds:
        cfg_path = f'{D}/f1cv_mae_fold{F}.py'
        ckpt_path = f'{WD}/mae_fold{F}/{args.epoch}'
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

        fila = {k: {'trivial': resumen(trivial(xs))} for k, xs in pobl.items()}

        for etiqueta, cargar in (('entrenado', True), ('aleatorio', False)):
            modelo = MODELS.build(cfg.model)
            if cargar:
                ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                pesos = ck['state_dict']
                inc = modelo.load_state_dict(pesos, strict=False)
                # Mismo guard que eval_fase1_seeds.py (ARREGLO 30/08, hallazgo 10):
                # strict=False acepta en silencio un checkpoint que no case en NADA
                # y deja el modelo con pesos ALEATORIOS. Acá eso sería peor que un
                # ADE plausible: mediríamos ruido y lo reportaríamos como "el
                # encoder entrenado", que es justo la conclusión de este script.
                cargadas = len(pesos) - len(inc.unexpected_keys)
                if cargadas <= 0:
                    raise SystemExit(
                        f'{ckpt_path} no aportó ni un peso a {cfg_path}. Medir así '
                        f'compara un modelo ALEATORIO contra otro modelo aleatorio.')
                esperadas = len(modelo.state_dict())
                if cargadas < 0.5 * esperadas:
                    print(f'  !! AVISO: solo {cargadas}/{esperadas} tensores '
                          f'cargados — revisar que config y checkpoint casen.')
                del ck, pesos
                gc.collect()
            modelo.to(dev)
            verificar_modo_train(modelo)
            modelo.train()      # imprescindible para que haya máscara; ver cabecera
            for k, xs in pobl.items():
                fila[k][etiqueta] = resumen(perdidas(modelo, xs, args.mascaras, dev))
            del modelo
            gc.collect()
            torch.cuda.empty_cache()

        tabla[F] = fila
        print(f'  {"población":14s} {"entrenado":22s} {"aleatorio":22s} trivial')
        for k in pobl:
            print(f'  {k:14s} {fila[k]["entrenado"]:22s} '
                  f'{fila[k]["aleatorio"]:22s} {fila[k]["trivial"]}')
        print()

    print('=== RESUMEN (media de la pérdida por población, modelo entrenado) ===')
    print(f'{"fold":6s} {"train_vistas":>13s} {"train_nuevas":>13s} {"val":>13s}')
    for F, fila in tabla.items():
        v = [fila[k]['entrenado'].split(' ')[0] for k in
             ('train_vistas', 'train_nuevas', 'val')]
        print(f'{F:<6d} {v[0]:>13s} {v[1]:>13s} {v[2]:>13s}')


if __name__ == '__main__':
    main()
