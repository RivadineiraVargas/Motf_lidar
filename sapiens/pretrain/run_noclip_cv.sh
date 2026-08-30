#!/bin/bash
# run_noclip_cv.sh — completa la CV de 5 folds en el protocolo VIGENTE.
#
# POR QUE. El 30/08, al estrenar agregar_resultados.py, su aviso automatico de
# "un solo fold" salto en TODOS los CSV del track vigente: f1cv, noclip, geo, jm,
# rvcv, rvaug. Existe un unico encoder (mae_encoder_fold0.pth) y las 48 corridas
# de f1cv son del fold 0. La CV de 5 folds de Fase 1 NUNCA SE CORRIO.
#
# O sea que los experimentos 15 a 18 —incluido el resultado mas firme del
# proyecto, la CAPACIDAD a -10% con 8/8 semillas y t=-5.35— descansan sobre UN
# SOLO SPLIT de escenas. Es exactamente el patron que ya se llevo puestos dos
# resultados: el 18/07 ("la escena ayuda") y el 06/08 (-20.4%, p=0.0006, 8/8
# semillas, que al promediar los 5 folds quedo en +0.077, t=0.589, nada). La
# varianza ENTRE folds es ~3x la de semillas (sd 0.29 vs 0.098): con 10 escenas
# domina QUE escenas caen de cada lado del corte.
#
# POR QUE NO SIRVE run_fase1_cv.sh. Ese script evalua SIN --sin-clip y SIN
# --eval-windows 7, o sea con el objetivo recortado y 51 muestras: el protocolo
# viejo del experimento 15. Sus numeros no son comparables con los 16-18. Este
# script corre el protocolo vigente (clip_norm=None, norm_scale=10.0, evaluacion
# sin recorte con 7 ventanas = 319 muestras) en los folds que faltan.
#
# ANTIFUGA. El MAE se re-pre-entrena desde cero en las 8 escenas de train de CADA
# fold; el decoder de cada fold carga el encoder de SU fold. Verificado antes de
# generar las configs: ninguna escena de validacion aparece en el train ni en el
# pre-entrenamiento de su propio fold, y las 10 escenas cubren validacion una vez
# cada una. El --resume del encoder es seguro: f1cv_mae_fold*.py no tiene
# load_from que perder (ver el bug de c6c9e05).
#
# TRES VARIANTES x 8 SEMILLAS por fold, epoca FIJA 100 (sin sesgo de seleccion):
#   baseline — MLP, sin escena
#   gate0    — CONTROL DE ARQUITECTURA: modelo completo, gate CONGELADO en 0
#   gated    — gate aprendible desde 0.5
#   gate0 - baseline = aporte de la CAPACIDAD   (el efecto a validar)
#   gated - gate0    = aporte de la ESCENA
#
# COSTO MEDIDO (no estimado): el encoder del fold 0 tardo 18.5 min (12:05->12:24
# del 24/08) y la tanda de 24 decoders de noclip tardo 4h47 (17:06->21:53 del
# 27/08). => ~5 h por fold, ~20 h por los cuatro.
#
# Es resumible: saltea encoders y checkpoints que ya existan, y el CSV se
# appendea. Sin `set -e`: si un fold falla se registra y sigue con el proximo.
cd /home/lcad/lidar_sweep_viewer/sapiens/pretrain
source /home/lcad/miniconda3/etc/profile.d/conda.sh
conda activate sapiens_gpu
D=configs/sapiens_mae/lidar
CSV=work_dirs/noclipcv/noclipcv_results.csv
mkdir -p work_dirs/noclipcv

for F in 1 2 3 4; do
    echo "######## FOLD $F — inicio $(date '+%d/%m %H:%M') ########"

    ENC=work_dirs/f1cv/mae_encoder_fold${F}.pth
    if [ ! -f "$ENC" ]; then
        echo "[fold $F] pre-entrenando el encoder (~19 min)"
        python -u tools/train.py $D/f1cv_mae_fold${F}.py --resume \
            > work_dirs/f1cv/mae_fold${F}.log 2>&1
        CK=work_dirs/f1cv/mae_fold${F}/epoch_1000.pth
        [ -f "$CK" ] || { echo "!!! fold $F: no se genero el MAE — salteo el fold"; continue; }
        python -u extract_mae_encoder.py $CK $ENC >> work_dirs/f1cv/mae_fold${F}.log 2>&1
    fi
    [ -f "$ENC" ] || { echo "!!! fold $F: falta el encoder — salteo"; continue; }
    echo "[fold $F] encoder listo"

    # Las escenas retenidas se leen del comentario del config del encoder, que es
    # la fuente unica de la definicion del fold.
    VAL=$(python3 -c "
import re
t = open('$D/f1cv_mae_fold${F}.py').read()
m = re.search(r'val RETENIDA del fold \d+: \[(.*?)\]', t)
print(' '.join(re.findall(r\"'([0-9a-f]{16})'\", m.group(1))))")
    [ -n "$VAL" ] || { echo "!!! fold $F: no pude leer las escenas de validacion — salteo"; continue; }
    echo "[fold $F] validacion: $VAL"

    for S in 0 1 2 3 4 5 6 7; do
        for V in baseline gate0 gated; do
            case $V in
              baseline) CFG=$D/noclip_base_fold${F}.py; OPT="" ;;
              gate0)    CFG=$D/noclip_dec_fold${F}.py;  OPT="model.gate_init=0.0 model.freeze_gate=True" ;;
              gated)    CFG=$D/noclip_dec_fold${F}.py;  OPT="model.gate_init=0.5" ;;
            esac
            WD=work_dirs/noclipcv/${V}_f${F}s${S}
            if [ ! -f "$WD/epoch_100.pth" ]; then
                python -u tools/train.py $CFG --work-dir $WD \
                    --cfg-options randomness.seed=$S $OPT > $WD.log 2>&1 \
                    || { echo "!!! fallo $V fold $F semilla $S"; continue; }
            fi
            python -u eval_fase1_seeds.py --cfg $CFG --ckpt $WD/epoch_100.pth \
                --variant $V --seed $S --fold $F --val-scenes $VAL \
                --eval-windows 7 --sin-clip --out $CSV 2>&1 | grep "^\[eval\]"
        done
        echo "----- fold $F semilla $S lista ($(date '+%d/%m %H:%M')) -----"
    done
    echo "######## FOLD $F — fin $(date '+%d/%m %H:%M') ########"
done

echo "=== CV DE 5 FOLDS COMPLETA ==="
echo "Para agregar (el fold 0 vive en work_dirs/noclip/noclip_results.csv):"
echo "  python agregar_resultados.py work_dirs/noclip/noclip_results.csv $CSV --por-fold"
