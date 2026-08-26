#!/bin/bash
# run_noclip.sh — reentrena el fold 0 (voxeles) con el objetivo SIN recortar.
# Primera medicion del proyecto en que el modelo aprende la tarea real.
cd /home/lcad/lidar_sweep_viewer/sapiens/pretrain
source /home/lcad/miniconda3/etc/profile.d/conda.sh; conda activate sapiens_gpu
D=configs/sapiens_mae/lidar; VAL="7e2f727866c69ea0 82f90331a1dfe968"
mkdir -p work_dirs/noclip
for S in 0 1 2 3 4 5 6 7; do
  for V in baseline gate0 gated; do
    case $V in
      baseline) CFG=$D/noclip_base_fold0.py; OPT="" ;;
      gate0)    CFG=$D/noclip_dec_fold0.py;  OPT="model.gate_init=0.0 model.freeze_gate=True" ;;
      gated)    CFG=$D/noclip_dec_fold0.py;  OPT="model.gate_init=0.5" ;;
    esac
    WD=work_dirs/noclip/${V}_f0s${S}
    [ -f "$WD/epoch_100.pth" ] || python -u tools/train.py $CFG --work-dir $WD --resume \
        --cfg-options randomness.seed=$S $OPT > $WD.log 2>&1 || { echo "!!! fallo $V s$S"; continue; }
    python -u eval_fase1_seeds.py --cfg $CFG --ckpt $WD/epoch_100.pth --variant $V \
        --seed $S --fold 0 --val-scenes $VAL --eval-windows 7 --sin-clip \
        --out work_dirs/noclip/noclip_results.csv 2>&1 | grep "^\[eval\]"
  done
  echo "----- semilla $S lista ($(date '+%H:%M')) -----"
done
echo "=== REENTRENAMIENTO SIN RECORTE COMPLETO ==="
