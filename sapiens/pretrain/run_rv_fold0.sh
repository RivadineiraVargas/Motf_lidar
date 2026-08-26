#!/bin/bash
# run_rv_fold0.sh — RANGE-VIEW, fold 0, mismo protocolo que la CV de vóxeles.
#
# OBJETIVO: comparación cabeza a cabeza vóxel vs range-view sobre EL MISMO split,
# con el mismo decoder, el mismo horizonte (3s) y el mismo control de arquitectura.
# Es la comparación que pide la Sec. 6 del plan de Claudine y que hasta ahora no
# existía con protocolo riguroso (el 1.303 vs 1.685 citado era 1 semilla, sin control).
#
# Comparable porque: mismas 8 escenas de train, mismas 2 retenidas, y el encoder
# de range-view (mae_encoder_rangeview.pth) se pre-entrenó SOLO en esas 8
# -> sin fuga, igual que el de vóxeles.
#
# Espera a que la CV de vóxeles cierre su fold 0 y la detiene antes del fold 1
# (decisión del usuario: comparar fold 0 en ambos pipelines y después seguir en
# range-view). Los checkpoints de vóxeles quedan: run_fase1_cv.sh los saltea si
# alguna vez se retoma.
cd /home/lcad/lidar_sweep_viewer/sapiens/pretrain
source /home/lcad/miniconda3/etc/profile.d/conda.sh
conda activate sapiens_gpu
D=configs/sapiens_mae/lidar
CSV=work_dirs/rvcv/rv_results.csv
mkdir -p work_dirs/rvcv

echo "[relevo] esperando el fin del fold 0 de vóxeles..."
until grep -q "FOLD 0 — fin" work_dirs/f1cv/driver.log 2>/dev/null; do sleep 30; done
echo "[relevo] fold 0 de vóxeles cerrado -> detengo la CV antes del fold 1"
pkill -f "run_fase1_cv.sh" 2>/dev/null
sleep 3
pkill -f "f1cv_mae_fold" 2>/dev/null      # por si ya arrancó el encoder del fold 1
sleep 2

VAL="7e2f727866c69ea0 82f90331a1dfe968"
for S in 0 1 2 3 4 5 6 7; do
    for V in baseline gate0 gated; do
        case $V in
          baseline) CFG=$D/rvcv_base_fold0.py; OPT="" ;;
          gate0)    CFG=$D/rvcv_dec_fold0.py;  OPT="model.gate_init=0.0 model.freeze_gate=True" ;;
          gated)    CFG=$D/rvcv_dec_fold0.py;  OPT="model.gate_init=0.5" ;;
        esac
        WD=work_dirs/rvcv/${V}_f0s${S}
        if [ ! -f "$WD/epoch_100.pth" ]; then
            python -u tools/train.py $CFG --work-dir $WD \
                --cfg-options randomness.seed=$S $OPT > $WD.log 2>&1 \
                || { echo "!!! falló $V seed $S (ver $WD.log)"; continue; }
        fi
        python -u eval_fase1_seeds.py --cfg $CFG --ckpt $WD/epoch_100.pth \
            --variant $V --seed $S --fold 0 --val-scenes $VAL --out $CSV \
            2>&1 | grep "^\[eval\]"
    done
    echo "----- semilla $S lista ($(date '+%H:%M')) -----"
done
echo "=== RANGE-VIEW FOLD 0 COMPLETO ==="
