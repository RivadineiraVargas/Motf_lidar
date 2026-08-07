#!/bin/bash
# run_fold4_experiment.sh — Encadena: espera a que termine el encoder de
# dominio del fold 4 y dispara el decoder a 3s con 8 semillas.
#
# Cierra el hueco del bloque de experimentos 7-9, que es todo del fold 0: la
# varianza ENTRE folds era la dominante (sd 0.326 a 8s vs 0.089 entre
# semillas), asi que el efecto medido (-20.4% a 3s, p=0.0006) necesita un
# segundo split para descartar que sea propio del fold 0.
#
# El fold 4 es el caso ADVERSARIAL: con el encoder generico era el peor de los
# cinco (diff +0.834, la escena danaba 3/3 semillas).
#
# --folds 4 es OBLIGATORIO: el encoder se pre-entreno en las 20 escenas de
# train del fold 4, usarlo en otro fold seria FUGA.
set -e
cd /home/lcad/lidar_sweep_viewer/sapiens/pretrain
source /home/lcad/miniconda3/etc/profile.d/conda.sh
conda activate sapiens_gpu

echo "[espera] aguardando fin del encoder del fold 4..."
while pgrep -f "config_rangeview_rect_[f]old4.py" >/dev/null 2>&1; do sleep 60; done

CKPT=work_dirs/rv_rect_fold4/epoch_1000.pth
if [ ! -f "$CKPT" ]; then
    echo "ABORTA: no existe $CKPT — el encoder no llego a las 1000 epocas."
    ls -la work_dirs/rv_rect_fold4/*.pth 2>/dev/null || true
    exit 1
fi
echo "[ok] encoder listo: $CKPT"

python -u horizon_sweep.py \
    --enc "$CKPT" \
    --folds 4 --seeds 0 1 2 3 4 5 6 7 --horizons 3s \
    --archs wayformer baseline \
    --cache work_dirs/cache_fold4_domain \
    --out work_dirs/horizon_fold4 \
    --epochs 100
echo "=== FOLD 4 COMPLETO ==="
