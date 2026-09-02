#!/bin/bash
# run_hist11.sh — ¿predice mejor con la historia COMPLETA (1,1 s) que con 0,5 s?
#
# LA PREGUNTA. WOMD-LiDAR da 11 frames de LiDAR por escena = 1,1 s, que es
# exactamente la ventana de historia del benchmark de Waymo (ver
# docs/ESTUDIO_WAYFORMER.md:38). Todos los configs de Fase 1 usan history_len=5:
# venimos prediciendo 3 s de futuro con 0,5 s de pasado y tirando mas de la mitad
# del contexto que ya esta en el disco.
#
# POR QUE EL BASELINE Y NO LAS VARIANTES CON ESCENA. BaselineTrajectoryModel es
# puramente cinematico: no toca el encoder. Cambiar history_len es una linea de
# config y cuesta 1,6 min por corrida (medido: baseline_f0s0 arranco 09:26:53 y
# cerro epoch_100 a las 09:28:32, contra 12,7 min de gate0). Las variantes con
# escena necesitarian re-pre-entrenar los 5 encoders, porque su patch_embed es
# Linear(history_len, 1024). Este script decide si eso vale la pena, por ~1 h.
#
# EL COSTO DE LA HISTORIA LARGA, medido antes de correr nada: pasar de 5 a 11
# cuesta el 13% de las ventanas de entrenamiento (236 -> 206 en el fold 0). No es
# el 7x que uno supondria: el train ya toma una sola ventana por objeto.
#
# LA TRAMPA QUE ESTE SCRIPT EVITA. Con history_len=11 solo entra la ventana f0=0,
# asi que la evaluacion por defecto pasaria de 183 ventanas / 29 objetos a 24 / 24:
# poblaciones distintas, ADE incomparables. Es el mismo error de fondo que costo el
# resultado del 30/08. Por eso los DOS brazos se evaluan con --poblacion-hist 11:
# se quedan las ventanas cuyo futuro arranca en el frame 11 y cuyo objeto existe
# tambien con historia 11 — el de h=5 usa su ventana f0=6, el de h=11 su f0=0, y
# ambos predicen los frames 11..40 de LOS MISMOS objetos. Medido: la poblacion de
# h=11 es subconjunto de la de h=5 en los 5 folds (24, 59, 40, 25, 82 objetos).
#
# POR ESO SE RE-EVALUA base5. Las filas de work_dirs/noclipcv son de la poblacion
# vieja (183 ventanas) y NO se pueden comparar con estas. Los checkpoints del
# baseline h=5 ya existen: se re-evaluan, no se re-entrenan.
#
# COSTO: 40 entrenamientos x ~2 min + 80 evaluaciones = ~1,5 h.
cd /home/lcad/lidar_sweep_viewer/sapiens/pretrain
source /home/lcad/miniconda3/etc/profile.d/conda.sh
conda activate sapiens_gpu

ya_evaluado() {   # $1=csv  $2=fold  $3=variante  $4=semilla  $5=nº de escenas esperadas
    [ -f "$1" ] || return 1
    [ "${5:-1}" -gt 0 ] || return 1
    local n; n=$(grep -c "^$2,$3,$4," "$1")
    [ "$n" -ge "$5" ]
}

D=configs/sapiens_mae/lidar
CSV=work_dirs/hist11/hist11_results.csv
mkdir -p work_dirs/hist11

for F in 0 1 2 3 4; do
    VAL=$(python3 -c "
import re
t = open('$D/f1cv_mae_fold${F}.py').read()
m = re.search(r'val RETENIDA del fold \d+: \[(.*?)\]', t)
print(' '.join(re.findall(r\"'([0-9a-f]{16})'\", m.group(1))))")
    [ -n "$VAL" ] || { echo "!!! fold $F: sin escenas de validacion — salteo"; continue; }
    NV=$(echo $VAL | wc -w)
    echo "######## FOLD $F — inicio $(date '+%d/%m %H:%M')  (val: $VAL) ########"

    for S in 0 1 2 3 4 5 6 7; do
        # --- brazo nuevo: historia 1,1 s ---
        WD=work_dirs/hist11/base11_f${F}s${S}
        [ -f "$WD/epoch_100.pth" ] || python -u tools/train.py $D/hist11_base_fold${F}.py \
            --work-dir $WD --cfg-options randomness.seed=$S > $WD.log 2>&1 \
            || { echo "!!! fallo entrenando base11 fold $F semilla $S"; continue; }
        ya_evaluado $CSV $F base11 $S $NV || \
            python -u eval_fase1_seeds.py --cfg $D/hist11_base_fold${F}.py \
                --ckpt $WD/epoch_100.pth --variant base11 --seed $S --fold $F \
                --val-scenes $VAL --eval-windows 7 --sin-clip \
                --poblacion-hist 11 --out $CSV 2>&1 | grep "^\[eval\]"

        # --- brazo de referencia: historia 0,5 s, checkpoint YA entrenado ---
        CK=work_dirs/noclipcv/baseline_f${F}s${S}/epoch_100.pth
        [ -f "$CK" ] || { echo "!!! falta $CK — sin referencia para f$F s$S"; continue; }
        ya_evaluado $CSV $F base5 $S $NV || \
            python -u eval_fase1_seeds.py --cfg $D/noclip_base_fold${F}.py \
                --ckpt $CK --variant base5 --seed $S --fold $F \
                --val-scenes $VAL --eval-windows 7 --sin-clip \
                --poblacion-hist 11 --out $CSV 2>&1 | grep "^\[eval\]"
    done
    echo "######## FOLD $F — fin $(date '+%d/%m %H:%M') ########"
done

echo "=== HISTORIA COMPLETA: CORRIDA TERMINADA ==="
echo "Para leer el resultado (n = folds, no corridas):"
echo "  python agregar_resultados.py $CSV --por-fold --comparar base11:base5"
