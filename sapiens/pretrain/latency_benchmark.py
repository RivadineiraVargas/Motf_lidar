"""
latency_benchmark.py — cuánto tarda el pipeline por sweep, en la GPU local.

Motivación: el objetivo final del proyecto es correr sobre el vehículo del LCAD
(integración con el stack astro, ver lidar_sweep_viewer_main.cpp). Antes de
encarar esa etapa conviene saber si la arquitectura entra en tiempo real: el
Velodyne del auto entrega sweeps a ~10 Hz, o sea un presupuesto de 100 ms por
sweep para TODO (encoder + decoder), y eso sin contar percepción ni control.

Mide por separado, con warmup y sincronización de CUDA (sin sync, los tiempos de
GPU son mentira porque las llamadas son asíncronas):
  - encoder MAE: 1 range-view (64x2650 -> PNG rectangular) -> tokens
  - decoder: cross-attn sobre los tokens + K slots -> trayectorias
  - total por sweep

Separa I/O de cómputo: `encode_sweeps` lee el .npy en CADA llamada, así que
medirlo directo mezcla disco + preproceso + forward y lo atribuye todo al
encoder. Acá el tensor se precarga y se cronometra solo el forward.

NO mide: extracción del range-view desde la nube cruda (en el auto, ese paso
existe y hay que sumarlo), ni postproceso.
"""
import argparse, time
import numpy as np
import torch
import train_decoder_mini as tdm
from cross_validate_decoder import make_folds
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpretrain.registry import MODELS


def timeit(fn, reps, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()          # imprescindible: CUDA es asíncrono
        ts.append((time.perf_counter() - t0) * 1000)
    return np.array(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--reps', type=int, default=30)
    ap.add_argument('--n-wp', type=int, default=6)
    args = ap.parse_args()
    tdm.N_WP = args.n_wp
    dev = 'cuda'
    init_default_scope('mmpretrain')
    print(f'GPU: {torch.cuda.get_device_name(0)}')

    cfg = Config.fromfile(f'configs/sapiens_mae/lidar/config_rangeview_rect_fold{args.fold}.py')
    mae = MODELS.build({**cfg.model, 'data_preprocessor': cfg.data_preprocessor})
    sd = torch.load(f'work_dirs/rv_rect_fold{args.fold}/epoch_1000.pth',
                    map_location='cpu').get('state_dict')
    mae.load_state_dict(sd, strict=False)
    enc = mae.backbone.to(dev)
    enc.eval()                       # OJO: MAEViT.eval() devuelve None, no encadenar

    scene = make_folds()[1][args.fold][0]
    lat = tdm.encode_sweeps(enc, scene, [0], dev)     # (1,L,384)
    s = tdm.build_sample(scene, 10)
    feat = (s['feat'] / tdm.SCALE).to(dev)
    print(f'escena {scene}: {s["n"]} objetos, {lat[0].shape[1]} tokens de escena')

    dec = tdm.MiniWayformerDecoder().to(dev)
    dec.eval()

    # forward puro: el tensor de entrada se precarga UNA vez
    x = tdm.sweep_tensor(scene, 0).to(dev)
    enc.mask_ratio = 0.0

    def fwd_enc():
        return enc(x)[0]

    with torch.no_grad():
        t_io = timeit(lambda: tdm.sweep_tensor(scene, 0).to(dev), args.reps)
        t_enc = timeit(fwd_enc, args.reps)
        t_dec = timeit(lambda: dec(lat[0], feat, s['n']), args.reps)

    print(f'\n{"etapa":<26} {"media":>9} {"sd":>8} {"p95":>9}')
    for name, t in (('lectura .npy + preproceso', t_io),
                    ('encoder MAE (forward)', t_enc),
                    ('decoder (K slots)', t_dec)):
        print(f'{name:<26} {t.mean():>7.1f}ms {t.std():>7.1f} {np.percentile(t,95):>7.1f}ms')
    tot = t_enc.mean() + t_dec.mean()
    print(f'{"TOTAL computo (enc+dec)":<26} {tot:>7.1f}ms')
    print(f'{"  (+ I/O del .npy)":<26} {tot + t_io.mean():>7.1f}ms')
    # --- media precisión: la optimización obvia, el encoder domina el costo ---
    enc_h = enc.half()
    xh = x.half()
    dec_h = dec.half()
    lat_h = {0: lat[0].half()}
    feat_h = feat.half()
    with torch.no_grad():
        t_enc_h = timeit(lambda: enc_h(xh)[0], args.reps)
        t_dec_h = timeit(lambda: dec_h(lat_h[0], feat_h, s['n']), args.reps)
    tot_h = t_enc_h.mean() + t_dec_h.mean()
    print(f'\n{"fp16 encoder":<26} {t_enc_h.mean():>7.1f}ms  '
          f'(x{t_enc.mean()/t_enc_h.mean():.2f} mas rapido)')
    print(f'{"fp16 TOTAL computo":<26} {tot_h:>7.1f}ms  -> {1000/tot_h:.1f} Hz')

    print(f'\npresupuesto a 10 Hz = 100 ms/sweep  ->  usa el {100*tot/100:.0f}% '
          f'({"ENTRA" if tot < 100 else "NO ENTRA"})')
    print(f'tasa máxima sostenible: {1000/tot:.1f} Hz')


if __name__ == '__main__':
    main()
