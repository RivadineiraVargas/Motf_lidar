#!/bin/bash
# run_geo.sh — ¿el objetivo geometrico + mas muestras arregla el encoder?
#
# Cambia DOS cosas respecto del exp. 16, y solo esas:
#   - objetivo del MAE: centroide por voxel en vez de ocupacion (GeoMAE)
#   - muestras de pre-entrenamiento: 56 en vez de 8 (7 ventanas por escena)
# El decoder, las semillas, el fold, el horizonte y la evaluacion son identicos,
# para que la comparacion contra work_dirs/noclip sea directa.
cd /home/lcad/lidar_sweep_viewer/sapiens/pretrain
source /home/lcad/miniconda3/etc/profile.d/conda.sh; conda activate sapiens_gpu
D=configs/sapiens_mae/lidar; VAL="7e2f727866c69ea0 82f90331a1dfe968"
mkdir -p work_dirs/geo

ENC=work_dirs/geo/mae_encoder_fold0.pth
if [ ! -f "$ENC" ]; then
    echo "### encoder geometrico ($(date '+%H:%M')) ###"
    python -u tools/train.py $D/geo_mae_fold0.py --resume > work_dirs/geo/mae.log 2>&1
    CK=work_dirs/geo/mae_fold0/epoch_1000.pth
    [ -f "$CK" ] || { echo "!!! no se genero el encoder"; exit 1; }
    python -u extract_mae_encoder.py $CK $ENC >> work_dirs/geo/mae.log 2>&1
fi
echo "### encoder listo ($(date '+%H:%M')) ###"

for S in 0 1 2 3 4 5 6 7; do
  for V in baseline gate0 gated; do
    case $V in
      baseline) CFG=$D/geo_base_fold0.py; OPT="" ;;
      gate0)    CFG=$D/geo_dec_fold0.py;  OPT="model.gate_init=0.0 model.freeze_gate=True" ;;
      gated)    CFG=$D/geo_dec_fold0.py;  OPT="model.gate_init=0.5" ;;
    esac
    WD=work_dirs/geo/${V}_f0s${S}
    [ -f "$WD/epoch_100.pth" ] || python -u tools/train.py $CFG --work-dir $WD \
        --cfg-options randomness.seed=$S $OPT > $WD.log 2>&1 || { echo "!!! fallo $V s$S"; continue; }
    python -u eval_fase1_seeds.py --cfg $CFG --ckpt $WD/epoch_100.pth --variant $V \
        --seed $S --fold 0 --val-scenes $VAL --eval-windows 7 --sin-clip \
        --out work_dirs/geo/geo_results.csv 2>&1 | grep "^\[eval\]"
  done
  echo "----- semilla $S lista ($(date '+%H:%M')) -----"
done
echo "=== EXPERIMENTO GEOMETRICO COMPLETO ==="
