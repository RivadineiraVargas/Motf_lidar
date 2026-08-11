#!/bin/bash
# run_gated_folds_1234.sh — completa el gate en los 4 folds que faltan.
#
# Contexto (10/08): la CV de 5 folds cerró y el efecto de la escena a 3s NO
# sobrevive — entre folds +0.077 ± 0.292, t=0.589, no significativo. El fold 3
# es un outlier fuerte EN CONTRA (+0.570, 0/8 semillas, +40%): ahí el wayformer
# quedó MUCHO peor que el baseline, o sea que el decoder no logró IGNORAR la
# escena cuando no servía.
#
# Hipótesis de este experimento: el gate (escalar aprendible tanh(scene_gate)
# sobre la rama de cross-attn) es justamente una válvula de amplitud que puede
# cerrarse hasta 0 y degradar con gracia al baseline. En el fold 0 ya se vio
# que (a) converge solo a 0.092 ± 0.005 desde 8 inicializaciones en 0.5, y
# (b) empata con el ungated (+0.033, t=1.16) => no cuesta nada donde hay señal.
# Si cierra en el fold 3 y mantiene la ganancia del fold 0, la media de los 5
# folds podría irse a negativo de verdad.
#
# Solo se corre wayformer_gated: los baselines de los 5 folds ya están en los
# CSV y horizon_sweep.py aparea contra ellos. El gate del fold 0 ya está hecho
# (work_dirs/horizon_domain), por eso arranca en el 1.
#
# Las features ya están cacheadas (cache_fold{F}_domain, 2.7GB c/u) => no hay
# que reentrenar ningún encoder de 12.5h. ~30 min por fold, ~2h total.
#
# --folds $F es OBLIGATORIO: el encoder del fold F vio en auto-supervisado las
# escenas retenidas de los OTROS folds; usarlo fuera del suyo sería FUGA.
#
# Sin `set -e`: si un fold falla se registra y se sigue con el siguiente.
# horizon_sweep.py appendea al CSV y saltea las combinaciones ya hechas, así
# que relanzar esto es seguro y resumible.
cd /home/lcad/lidar_sweep_viewer/sapiens/pretrain
source /home/lcad/miniconda3/etc/profile.d/conda.sh
conda activate sapiens_gpu

for F in 1 2 3 4; do
    echo "############ FOLD $F (gated) — inicio $(date '+%d/%m %H:%M') ############"

    CKPT=work_dirs/rv_rect_fold${F}/epoch_1000.pth
    if [ ! -f "$CKPT" ]; then
        echo "!!! FOLD $F: falta el encoder $CKPT — salteo"
        continue
    fi

    python -u horizon_sweep.py \
        --enc "$CKPT" \
        --folds $F --seeds 0 1 2 3 4 5 6 7 --horizons 3s \
        --archs wayformer_gated \
        --cache work_dirs/cache_fold${F}_domain \
        --out work_dirs/horizon_fold${F} \
        --epochs 100 \
        || echo "!!! FOLD $F: falló el decoder gated — sigo con el próximo"

    echo "############ FOLD $F (gated) — fin $(date '+%d/%m %H:%M') ############"
done
echo "=== GATE FOLDS 1-4 COMPLETO ==="
