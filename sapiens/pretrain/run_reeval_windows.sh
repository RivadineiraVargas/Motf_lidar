#!/bin/bash
# run_reeval_windows.sh — re-evalúa los checkpoints del fold 0 (vóxeles y
# range-view) con 7 ventanas temporales por objeto en vez de 1.
# No reentrena nada: 51 -> 319 muestras de test, x6.3, sin tocar el train (310).
cd /home/lcad/lidar_sweep_viewer/sapiens/pretrain
source /home/lcad/miniconda3/etc/profile.d/conda.sh
conda activate sapiens_gpu
D=configs/sapiens_mae/lidar
VAL="7e2f727866c69ea0 82f90331a1dfe968"
run () {  # $1=dir  $2=cfg_dec  $3=cfg_base  $4=csv
    for S in 0 1 2 3 4 5 6 7; do
        for V in baseline gate0 gated; do
            [ "$V" = baseline ] && CFG=$3 || CFG=$2
            CK=$1/${V}_f0s${S}/epoch_100.pth
            [ -f "$CK" ] || continue
            python -u eval_fase1_seeds.py --cfg $CFG --ckpt $CK --variant $V \
                --seed $S --fold 0 --val-scenes $VAL --eval-windows 7 --out $4 \
                2>&1 | grep "^\[eval\]"
        done
    done
}
echo "### VOXELES ###"
run work_dirs/f1cv $D/f1cv_dec_fold0.py $D/f1cv_base_fold0.py work_dirs/reeval7_voxel.csv
echo "### RANGE-VIEW (augmentado) ###"
run work_dirs/rvaug $D/rvaug_dec_fold0.py $D/rvaug_base_fold0.py work_dirs/reeval7_rv.csv
echo "=== RE-EVALUACION 7 VENTANAS COMPLETA ==="
