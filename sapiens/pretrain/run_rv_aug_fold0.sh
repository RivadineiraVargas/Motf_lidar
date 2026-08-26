#!/bin/bash
# run_rv_aug_fold0.sh — range-view fold 0 CON augmentación, para igualar el lado
# de vóxeles y hacer válida la comparación absoluta entre representaciones.
cd /home/lcad/lidar_sweep_viewer/sapiens/pretrain
source /home/lcad/miniconda3/etc/profile.d/conda.sh
conda activate sapiens_gpu
D=configs/sapiens_mae/lidar
CSV=work_dirs/rvaug/rvaug_results.csv
mkdir -p work_dirs/rvaug
VAL="7e2f727866c69ea0 82f90331a1dfe968"
for S in 0 1 2 3 4 5 6 7; do
    for V in baseline gate0 gated; do
        case $V in
          baseline) CFG=$D/rvaug_base_fold0.py; OPT="" ;;
          gate0)    CFG=$D/rvaug_dec_fold0.py;  OPT="model.gate_init=0.0 model.freeze_gate=True" ;;
          gated)    CFG=$D/rvaug_dec_fold0.py;  OPT="model.gate_init=0.5" ;;
        esac
        WD=work_dirs/rvaug/${V}_f0s${S}
        if [ ! -f "$WD/epoch_100.pth" ]; then
            python -u tools/train.py $CFG --work-dir $WD --cfg-options randomness.seed=$S $OPT \
                > $WD.log 2>&1 || { echo "!!! fallo $V seed $S"; continue; }
        fi
        python -u eval_fase1_seeds.py --cfg $CFG --ckpt $WD/epoch_100.pth \
            --variant $V --seed $S --fold 0 --val-scenes $VAL --out $CSV 2>&1 | grep "^\[eval\]"
    done
    echo "----- semilla $S lista ($(date '+%H:%M')) -----"
done
# completa el baseline sin augmentacion que falto (semilla 0)
python -u tools/train.py $D/rvcv_base_fold0.py --work-dir work_dirs/rvcv/baseline_f0s0 \
    --cfg-options randomness.seed=0 > work_dirs/rvcv/baseline_f0s0.log 2>&1 \
 && python -u eval_fase1_seeds.py --cfg $D/rvcv_base_fold0.py \
    --ckpt work_dirs/rvcv/baseline_f0s0/epoch_100.pth --variant baseline --seed 0 --fold 0 \
    --val-scenes $VAL --out work_dirs/rvcv/rv_results.csv 2>&1 | grep "^\[eval\]"
echo "=== RANGE-VIEW CON AUGMENTACION COMPLETO ==="
