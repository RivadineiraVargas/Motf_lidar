# recon_dos_ckpts.py — reconstruccion de los DOS encoders de cada fold, con
# semillas de mascara FRESCAS.
#
# POR QUE NO ALCANZA CON LA CURVA. En el experimento 26, la "mejor epoca" de cada
# fold se eligio como el minimo de 91 checkpoints medidos con las mascaras 0..3.
# Ese minimo esta sesgado hacia abajo por construccion —en el fold 0 cae a -2,69
# sd—, asi que su ventaja MEDIDA sobre la epoca 1000 es mayor que la real.
#
# El experimento 27 pregunta si la reconstruccion predice el ADE. Si entrara con
# la ventaja sesgada, cualquier correlacion que saliera estaria inflada por el
# mismo sesgo en el eje x. Este script vuelve a medir los dos checkpoints con
# mascaras que NO participaron de la seleccion (semillas 100..103), lo que da una
# estimacion limpia de la diferencia y ademas cuantifica cuanto se encogio.
#
# No entrena ni escribe checkpoints: solo lee y hace forwards.
#
# Uso:  conda activate sapiens_gpu && python recon_dos_ckpts.py
import csv, gc, glob, os, re, statistics as st
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpretrain.registry import MODELS
from diagnostico_encoder_mae import D, escenas_val, construir_pobl, verificar_modo_train

BASE = 'work_dirs/f1cv_curva'
SEMILLAS = [100, 101, 102, 103]     # frescas: la seleccion uso 0..3


@torch.no_grad()
def perdidas_offset(modelo, xs, semillas, dev):
    """Igual que diagnostico_encoder_mae.perdidas pero con semillas explicitas,
    para poder usar sorteos que no participaron de la seleccion. No se modifica
    la funcion original: la adenda y la curva del exp. 26 dependen de ella."""
    out = []
    for x in xs:
        x = x.unsqueeze(0).to(dev)
        acc = []
        for s in semillas:
            torch.manual_seed(s)
            torch.cuda.manual_seed_all(s)
            acc.append(float(modelo.loss(x)['loss']))
        out.append(sum(acc) / len(acc))
    return out


def medir(cfg, ruta, xs, dev):
    modelo = MODELS.build(cfg.model)
    ck = torch.load(ruta, map_location='cpu', weights_only=False)
    pesos = ck.get('state_dict', ck)
    inc = modelo.load_state_dict(pesos, strict=False)
    cargadas = len(pesos) - len(inc.unexpected_keys)
    if cargadas <= 0:
        raise SystemExit(f'{ruta} no aporto ni un peso.')
    del ck, pesos; gc.collect()
    modelo.to(dev); verificar_modo_train(modelo); modelo.train()
    v = perdidas_offset(modelo, xs, SEMILLAS, dev)
    del modelo; gc.collect(); torch.cuda.empty_cache()
    return v


def main():
    init_default_scope('mmpretrain')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    filas = []
    print(f'mascaras {SEMILLAS} (frescas; la seleccion del exp. 26 uso 0..3)\n')
    print(f"{'fold':<5}{'ep_mejor':>9}{'recon_mejor':>13}{'recon_1000':>12}"
          f"{'dif':>9}{'rel':>8}{'dif_curva':>11}")
    print('-' * 68)
    for F in range(5):
        cfg_path = f'{D}/f1cv_mae_fold{F}.py'
        cfg = Config.fromfile(cfg_path)
        va = escenas_val(cfg_path)
        xs = construir_pobl(cfg, va, 7)

        cks = {int(re.search(r'epoch_(\d+)', p).group(1)): p
               for p in glob.glob(f'{BASE}/mae_fold{F}/epoch_*.pth')}
        ep_mejor = min(e for e in cks if e != 1000)
        if len(cks) != 2:
            raise SystemExit(f'fold {F}: esperaba 2 checkpoints, hay {sorted(cks)}')

        vm = medir(cfg, cks[ep_mejor], xs, dev)
        vu = medir(cfg, cks[1000], xs, dev)
        m, u = st.mean(vm), st.mean(vu)

        # cuanto decia la curva (mascaras 0..3, con el sesgo de seleccion)
        cur = {r['epoca']: float(r['val'])
               for r in csv.DictReader(open(f'{BASE}/curva_fold{F}.csv'))}
        dif_curva = cur[str(ep_mejor)] - cur['1000']

        filas.append(dict(fold=F, epoca_mejor=ep_mejor, recon_mejor=m,
                          recon_1000=u, dif=m - u, rel=(m - u) / u * 100,
                          dif_curva=dif_curva))
        print(f'{F:<5}{ep_mejor:>9}{m:>13.4f}{u:>12.4f}{m-u:>+9.4f}'
              f'{(m-u)/u*100:>+7.1f}%{dif_curva:>+11.4f}')

    out = f'{BASE}/recon_dos_ckpts.csv'
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader(); w.writerows(filas)

    d = [f['dif'] for f in filas]
    dc = [f['dif_curva'] for f in filas]
    print('-' * 68)
    print(f'  diferencia media con mascaras frescas : {st.mean(d):+.4f}')
    print(f'  diferencia media segun la curva (0..3): {st.mean(dc):+.4f}')
    print(f'  encogimiento por regresion a la media : '
          f'{(1 - st.mean(d) / st.mean(dc)) * 100:.0f} %')
    print(f'  folds donde el "mejor" sigue siendo mejor: '
          f'{sum(1 for x in d if x < 0)}/5')
    print(f'\n  CSV: {out}')


if __name__ == '__main__':
    main()
