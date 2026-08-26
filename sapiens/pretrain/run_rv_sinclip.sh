#!/bin/bash
cd /home/lcad/lidar_sweep_viewer/sapiens/pretrain
source /home/lcad/miniconda3/etc/profile.d/conda.sh; conda activate sapiens_gpu
D=configs/sapiens_mae/lidar; VAL="7e2f727866c69ea0 82f90331a1dfe968"
for S in 0 1 2 3 4 5 6 7; do for V in baseline gate0 gated; do
  [ "$V" = baseline ] && CFG=$D/rvaug_base_fold0.py || CFG=$D/rvaug_dec_fold0.py
  CK=work_dirs/rvaug/${V}_f0s${S}/epoch_100.pth; [ -f "$CK" ] || continue
  python -u eval_fase1_seeds.py --cfg $CFG --ckpt $CK --variant $V --seed $S --fold 0 \
    --val-scenes $VAL --eval-windows 7 --sin-clip --out work_dirs/sinclip_rv.csv 2>&1 | grep "^\[eval\]"
done; done
echo "=== RANGE-VIEW SIN CLIP COMPLETO ==="
