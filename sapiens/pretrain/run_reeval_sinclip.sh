#!/bin/bash
# run_reeval_sinclip.sh — re-evalúa los checkpoints del fold 0 (vóxeles y
# range-view) SIN el recorte del objetivo, y con 7 ventanas por objeto.
#
# El clip a ±5 desvíos del histórico equivale a ±2.5 m, pero en coordenadas
# relativas al ego los objetos se desplazan ~40 m en 3 s (movimiento del objeto
# MÁS el del propio vehículo). Resultado: el 32% de los valores del futuro se
# recortaban, subestimando el movimiento real en 80%.
# Los modelos se ENTRENARON contra el objetivo recortado; esto mide su error
# real contra la trayectoria completa. Las comparaciones entre modelos siguen
# siendo válidas en ambos regímenes (todos comparten el mismo objetivo).
cd /home/lcad/lidar_sweep_viewer/sapiens/pretrain
source /home/lcad/miniconda3/etc/profile.d/conda.sh
conda activate sapiens_gpu
D=configs/sapiens_mae/lidar
VAL="7e2f727866c69ea0 82f90331a1dfe968"
run(){ for S in 0 1 2 3 4 5 6 7; do for V in baseline gate0 gated; do
    [ "$V" = baseline ] && CFG=$3 || CFG=$2
    CK=$1/${V}_f0s${S}/epoch_100.pth; [ -f "$CK" ] || continue
    python -u eval_fase1_seeds.py --cfg $CFG --ckpt $CK --variant $V --seed $S \
      --fold 0 --val-scenes $VAL --eval-windows 7 --sin-clip --out $4 2>&1 | grep "^\[eval\]"
  done; done; }
echo "### VOXELES ###"
run work_dirs/f1cv  $D/f1cv_dec_fold0.py  $D/f1cv_base_fold0.py  work_dirs/sinclip_voxel.csv
echo "### RANGE-VIEW ###"
run work_dirs/rvaug $D/rvaug_dec_fold0.py $D/rvaug_base_fold0.py work_dirs/sinclip_rv.csv
echo "=== RE-EVALUACION SIN CLIP COMPLETA ==="
