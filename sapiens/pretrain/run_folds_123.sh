#!/bin/bash
# run_folds_123.sh — Completa la validacion cruzada de 5 folds.
#
# Contexto: el efecto de la escena a 3s resulto DEPENDER DEL SPLIT
#   fold 0: -0.186 +- 0.089  t=-5.94  p=0.0006  8/8   (-20.4%)
#   fold 4: -0.024 +- 0.115  t=-0.59  4/8            (-1.3%, nulo)
# Con sd ENTRE folds de 0.115 y solo 2 folds medidos, no se puede concluir.
# Este script corre los 3 que faltan (1, 2, 3) para promediar sobre los 5.
#
# Por fold: encoder MAE re-pre-entrenado en SUS 20 escenas de train (~12.5h)
# -> decoder a 3s, 8 semillas, wayformer + baseline (~1h). Secuencial: la GPU
# no entra en paralelo con un encoder. Total estimado ~40h.
#
# --folds N es OBLIGATORIO en cada paso: el encoder de un fold vio en
# auto-supervisado las escenas retenidas de los OTROS folds; usarlo fuera del
# suyo seria FUGA.
#
# Sin `set -e`: si un fold falla, se registra y se sigue con el siguiente en
# vez de perder las decenas de horas restantes.
cd /home/lcad/lidar_sweep_viewer/sapiens/pretrain
source /home/lcad/miniconda3/etc/profile.d/conda.sh
conda activate sapiens_gpu

for F in 1 2 3; do
    echo "############ FOLD $F — inicio $(date '+%d/%m %H:%M') ############"

    CKPT=work_dirs/rv_rect_fold${F}/epoch_1000.pth
    if [ -f "$CKPT" ]; then
        echo "[fold $F] encoder ya existe, salteo el entrenamiento"
    else
        echo "[fold $F] PASO 1/2: encoder (~12.5h)"
        # --resume retoma del ultimo checkpoint del work_dir (se guarda c/250 ep).
        # Sin esto, un corte reiniciaba de cero: el 08/08 la maquina se suspendio
        # en la epoca 403 del fold 3, la GPU quedo en Xid 154 (reboot requerido) y
        # al relanzar se habrian perdido las 5.5h hechas. Es no-op si no hay
        # checkpoint previo, asi que es seguro dejarlo siempre.
        python -u tools/train.py configs/sapiens_mae/lidar/config_rangeview_rect_fold${F}.py --resume
        if [ ! -f "$CKPT" ]; then
            echo "!!! FOLD $F FALLO: no se genero $CKPT — sigo con el proximo fold"
            continue
        fi
    fi
    echo "[fold $F] encoder listo: $CKPT"

    echo "[fold $F] PASO 2/2: decoder 3s, 8 semillas"
    python -u horizon_sweep.py \
        --enc "$CKPT" \
        --folds $F --seeds 0 1 2 3 4 5 6 7 --horizons 3s \
        --archs wayformer baseline \
        --cache work_dirs/cache_fold${F}_domain \
        --out work_dirs/horizon_fold${F} \
        --epochs 100 \
        || echo "!!! FOLD $F: fallo el decoder — sigo con el proximo fold"

    echo "############ FOLD $F — fin $(date '+%d/%m %H:%M') ############"
done
echo "=== FOLDS 1-3 COMPLETOS ==="
