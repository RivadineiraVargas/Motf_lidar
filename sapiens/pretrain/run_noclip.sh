#!/bin/bash
# run_noclip.sh — reentrena el fold 0 (voxeles) con el objetivo SIN recortar.
# Primera medicion del proyecto en que el modelo aprende la tarea real.
cd /home/lcad/lidar_sweep_viewer/sapiens/pretrain
source /home/lcad/miniconda3/etc/profile.d/conda.sh; conda activate sapiens_gpu

# Evalúa solo si esa combinación no está YA en el CSV. Antes la evaluación estaba
# fuera del guard de reanudación y `eval_fase1_seeds.py` appendea sin comprobar:
# relanzar una corrida cortada —lo que estos scripts dicen soportar— duplicaba
# filas, y una sola fila duplicada mueve la media ponderada ~19%. Condicionarlo a
# NUEVO=1 no alcanzaba: si el corte cae entre entrenar y evaluar, el checkpoint
# existe y la fila nunca se escribiría. La fuente de verdad es el CSV.
ya_evaluado() {   # $1=csv  $2=fold  $3=variante  $4=semilla
    [ -f "$1" ] && grep -q "^$2,$3,$4," "$1"
}
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
    NUEVO=0; [ -f "$WD/epoch_100.pth" ] || { NUEVO=1; } ; [ $NUEVO -eq 1 ] && python -u tools/train.py $CFG --work-dir $WD \
        --cfg-options randomness.seed=$S $OPT > $WD.log 2>&1 || { echo "!!! fallo $V s$S"; continue; }
    ya_evaluado work_dirs/noclip/noclip_results.csv 0 $V $S || python -u eval_fase1_seeds.py --cfg $CFG --ckpt $WD/epoch_100.pth --variant $V \
        --seed $S --fold 0 --val-scenes $VAL --eval-windows 7 --sin-clip \
        --out work_dirs/noclip/noclip_results.csv 2>&1 | grep "^\[eval\]"
  done
  echo "----- semilla $S lista ($(date '+%H:%M')) -----"
done
echo "=== REENTRENAMIENTO SIN RECORTE COMPLETO ==="
