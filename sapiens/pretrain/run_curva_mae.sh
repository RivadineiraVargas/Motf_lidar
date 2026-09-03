#!/bin/bash
# run_curva_mae.sh — re-pre-entrena el MAE de vóxeles guardando checkpoints densos
# y mide la curva de reconstrucción fuera de las escenas de train.
#
# QUE RESPONDE. La adenda del exp. 23 midió las épocas 600/800/1000 y no encontró
# degradación, pero esas son el ÚLTIMO 40 %: el config tiene
# `checkpoint=dict(interval=200, max_keep_ckpts=3)` y no queda nada anterior en
# disco. En range-view el óptimo estaba en la época 50 de 6000 (0,8 % de la
# corrida); el equivalente acá sería la época ~8, y un pico así de temprano se
# vería exactamente como la meseta plana que midió la adenda.
#
# Importa porque TODA la Fase 1 (exp. 19-22) usó epoch_1000, y es el conjunto que
# respondió que la escena no aporta (0/5 folds).
#
# COMO. Mismo config y misma semilla (randomness.seed=0), solo se cambia el hook
# de checkpoint: interval=10, max_keep_ckpts=-1 (guardar todos) y
# save_optimizer=False, que baja cada archivo de 3,76 GB a ~1,25 GB. Son 100
# checkpoints por fold, ~125 GB, y se BORRAN al terminar de evaluarlos dejando
# solo el mejor y el 1000 — con 409 GB libres se hace de a un fold por vez.
#
# CONTROL DE SANIDAD. Como la semilla es la misma, la época 1000 de la curva debe
# reproducir el valor de la adenda para ese fold. curva_mae_voxel.py lo compara y
# avisa si no cierra; si no cierra, la curva no es comparable con la adenda.
#
# OJO: esto NO toca los encoders originales de work_dirs/f1cv. Escribe en
# work_dirs/f1cv_curva/. Los experimentos 19-22 siguen reproducibles.
#
# Uso:  ./run_curva_mae.sh 0          # piloto de un fold
#       ./run_curva_mae.sh 0 1 2 3 4  # los cinco
cd /home/lcad/lidar_sweep_viewer/sapiens/pretrain
source /home/lcad/miniconda3/etc/profile.d/conda.sh
conda activate sapiens_gpu

D=configs/sapiens_mae/lidar
BASE=work_dirs/f1cv_curva
mkdir -p $BASE
FOLDS=${@:-0}

for F in $FOLDS; do
    WD=$BASE/mae_fold$F
    echo "######## FOLD $F — inicio $(date '+%d/%m %H:%M') ########"

    LIBRE=$(df --output=avail -BG . | tail -1 | tr -dc '0-9')
    [ "$LIBRE" -ge 200 ] || { echo "!!! solo ${LIBRE}G libres, hacen falta ~200G — salteo"; continue; }

    if [ ! -f "$WD/epoch_1000.pth" ]; then
        python -u tools/train.py $D/f1cv_mae_fold${F}.py --work-dir $WD \
            --cfg-options default_hooks.checkpoint.interval=10 \
                          default_hooks.checkpoint.max_keep_ckpts=-1 \
                          default_hooks.checkpoint.save_optimizer=False \
            > $WD.train.log 2>&1 \
            || { echo "!!! fallo entrenando fold $F — ver $WD.train.log"; continue; }
    else
        echo "  (ya entrenado, reuso los checkpoints)"
    fi
    N=$(ls $WD/epoch_*.pth 2>/dev/null | wc -l)
    echo "  $N checkpoints · $(du -sh $WD 2>/dev/null | cut -f1) · $(date '+%H:%M')"

    python -u curva_mae_voxel.py --fold $F --work-dir $WD --mascaras 4 \
        --out $BASE/curva_fold${F}.csv > $WD.curva.log 2>&1 \
        || { echo "!!! fallo midiendo la curva del fold $F — ver $WD.curva.log"; continue; }
    tail -8 $WD.curva.log

    # Liberar disco: quedan el mejor y el 1000. El CSV ya tiene la curva entera.
    MEJOR=$(python3 -c "
import csv
f=[r for r in csv.DictReader(open('$BASE/curva_fold${F}.csv')) if r['epoca']!='sin_entrenar']
print(min(f,key=lambda r: float(r['val']))['epoca'])")
    for c in $WD/epoch_*.pth; do
        e=$(basename $c .pth); e=${e#epoch_}
        [ "$e" = "$MEJOR" ] || [ "$e" = "1000" ] || rm -f "$c"
    done
    echo "----- fold $F listo ($(date '+%d/%m %H:%M')) · mejor época $MEJOR · quedan $(du -sh $WD | cut -f1) -----"
done

echo "=== CURVAS COMPLETAS ==="
echo "Comparar los folds:  head -1 $BASE/curva_fold0.csv; grep -h . $BASE/curva_fold*.csv"
