#!/bin/bash
# run_reeval_tipo.sh — re-evalua los 40 checkpoints del exp. 28 con desglose por
# tipo de agente. NO re-entrena nada: reutiliza 95 GB de computo ya hecho.
#
# POR QUE. La extraccion guardo todos los tracks sin filtrar por tipo y sin
# guardar el object_type, asi que autos, peatones y ciclistas se entrenan y se
# miden juntos. Recuperando el tipo del tamanyo de la caja:
#
#     fold   TRAIN veh    VAL veh
#       0      88,1 %     100,0 %
#       1      93,7 %      74,8 %
#       2      88,4 %      96,5 %
#       3      89,0 %      92,2 %
#       4      88,1 %      91,1 %
#
# El entrenamiento es homogeneo (~88-94 %) pero la POBLACION DE EVALUACION varia
# de 74,8 % a 100 %. Los folds no son intercambiables, y el test entre folds
# asume que lo son. El fold 1 —el de menor proporcion de vehiculos— es ademas el
# que dio el efecto mas negativo del exp. 28 (-0,260): hay que descartar que la
# composicion lo explique.
#
# Escribe un CSV APARTE. El original (objcentrico_results.csv, 15 columnas) no se
# toca: es el que respalda el exp. 28 ya documentado y comiteado.
#
# OJO — ESTO ES POST HOC. Mirar por tipo DESPUES de ver el resultado global es la
# mecanica de la trampa 29. Vale como DESCRIPCION de la poblacion y para detectar
# un confound; NO como resultado nuevo. Si el efecto sobre vehiculos sale mayor,
# hay que fijarlo de antemano y validarlo, no reportarlo como confirmacion.
cd /home/lcad/lidar_sweep_viewer/sapiens/pretrain
source /home/lcad/miniconda3/etc/profile.d/conda.sh
conda activate sapiens_gpu

ya_evaluado() {
    [ -f "$1" ] || return 1
    [ "${5:-1}" -gt 0 ] || return 1
    local n; n=$(grep -c "^$2,$3,$4," "$1")
    [ "$n" -ge "$5" ]
}

D=configs/sapiens_mae/lidar
CSV=work_dirs/objcentrico/objcentrico_tipo_results.csv
OBJ=train_dataloader.dataset.centrar_en_objeto=True

for F in 0 1 2 3 4; do
    VAL=$(python3 -c "
import re
t = open('$D/f1cv_mae_fold${F}.py').read()
m = re.search(r'val RETENIDA del fold \d+: \[(.*?)\]', t)
print(' '.join(re.findall(r\"'([0-9a-f]{16})'\", m.group(1))))")
    [ -n "$VAL" ] || { echo "!!! fold $F sin escenas — salteo"; continue; }
    NV=$(echo $VAL | wc -w)
    for PAR in "gate0_obj:model.gate_init=0.0 model.freeze_gate=True" \
               "gated_obj:model.gate_init=0.05"; do
        V=${PAR%%:*}; OPT=${PAR#*:}
        for S in 0 1 2 3; do
            WD=work_dirs/objcentrico/${V}_f${F}s${S}
            [ -f "$WD/epoch_100.pth" ] || { echo "!!! falta $WD/epoch_100.pth"; continue; }
            ya_evaluado $CSV $F $V $S $NV || {
                python -u eval_fase1_seeds.py --cfg $D/noclip_dec_fold${F}.py \
                    --ckpt $WD/epoch_100.pth --variant $V --seed $S --fold $F \
                    --val-scenes $VAL --eval-windows 7 --sin-clip \
                    --cfg-options $OBJ $OPT --out $CSV > $WD.tipo.log 2>&1 \
                    || echo "!!! fallo evaluando $V fold $F semilla $S"
            }
        done
    done
    echo "----- fold $F re-evaluado ($(date '+%H:%M')) -----"
done
echo "=== RE-EVALUACION POR TIPO COMPLETA ==="
echo "  python agregar_resultados.py $CSV --comparar gated_obj:gate0_obj --por-fold --poblacion vehiculos"
