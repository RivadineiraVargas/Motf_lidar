#!/bin/bash
# run_fase1_cv.sh — validación cruzada de FASE 1 (vóxeles, 10 escenas, 3s).
#
# POR QUÉ: el 23/08 medimos, con 8 semillas y control de arquitectura, que en
# Fase 1 la escena aporta -0.397 ADE (t=-7.14, 8/8, -22.2%). Es el primer
# positivo del proyecto que sobrevive al control que faltó en Fase 2. PERO es UN
# SOLO SPLIT, y este proyecto ya vio cinco resultados de un split evaporarse al
# promediar folds (uno tenía p=0.0006 y 8/8 semillas). Sin CV no se puede afirmar.
#
# DISEÑO: 5 folds sobre las 10 escenas (8 train / 2 val cada uno). El fold 0 ES
# el split original, para ubicar el -22.2% en contexto.
#
# ANTIFUGA: el MAE 4D se re-pre-entrena desde cero en las 8 escenas de train de
# CADA fold. Usar el encoder de un fold en otro sería fuga auto-supervisada: vio
# sin etiquetas escenas que allí serían de validación. Verificado antes de lanzar
# que ninguna escena de val de un fold aparece en su pretrain ni en su train.
#
# TRES VARIANTES x 8 SEMILLAS por fold:
#   baseline — MLP, sin escena
#   gate0    — CONTROL DE ARQUITECTURA: modelo completo con gate CONGELADO en 0
#   gated    — gate aprendible desde 0.5
# Separa lo que Fase 2 tuvo confundido 14 experimentos:
#   gated - gate0    = aporte de la ESCENA (arquitectura igualada)
#   gate0 - baseline = aporte de la CAPACIDAD
#
# Se evalúa en ÉPOCA FIJA (100), no en el mejor checkpoint: así no se hereda el
# sesgo de selección sobre el propio test (hallazgo H1 de la auditoría del 23/08).
#
# COSTO: ~20 min de encoder por fold + 8 semillas x (2 + 15 + 15 min) ≈ 4.5 h/fold
# => ~23 h en total. Sin `set -e`: si un fold falla se registra y sigue.
cd /home/lcad/lidar_sweep_viewer/sapiens/pretrain
source /home/lcad/miniconda3/etc/profile.d/conda.sh
conda activate sapiens_gpu

# Evalúa solo si esa combinación no está YA en el CSV. Antes la evaluación estaba
# fuera del guard de reanudación y `eval_fase1_seeds.py` appendea sin comprobar:
# relanzar una corrida cortada —lo que estos scripts dicen soportar— duplicaba
# filas, y una sola fila duplicada mueve la media ponderada ~19%. Condicionarlo a
# NUEVO=1 no alcanzaba: si el corte cae entre entrenar y evaluar, el checkpoint
# existe y la fila nunca se escribiría. La fuente de verdad es el CSV.
ya_evaluado() {   # $1=csv  $2=fold  $3=variante  $4=semilla  $5=nº de escenas esperadas
    # Exige que estén TODAS las filas, no una. `eval_fase1_seeds.py` escribe una
    # fila POR ESCENA: si una evaluación previa murió después de la primera, dar
    # la combinación por hecha dejaría el fold con medio resultado, en silencio.
    [ -f "$1" ] || return 1
    # Si el nº esperado de escenas es 0 —p.ej. porque VAL quedó vacío— NO se puede
    # concluir "ya evaluado": `[ n -ge 0 ]` es cierto siempre y saltearíamos la
    # evaluación de un fold entero en silencio, tras horas de entrenamiento.
    [ "${5:-1}" -gt 0 ] || return 1
    local n; n=$(grep -c "^$2,$3,$4," "$1")
    [ "$n" -ge "$5" ]
}
D=configs/sapiens_mae/lidar
CSV=work_dirs/f1cv/f1cv_results.csv
mkdir -p work_dirs/f1cv

for F in 0 1 2 3 4; do
    echo "######## FOLD $F — inicio $(date '+%d/%m %H:%M') ########"
    ENC=work_dirs/f1cv/mae_encoder_fold${F}.pth
    if [ ! -f "$ENC" ]; then
        python -u tools/train.py $D/f1cv_mae_fold${F}.py --resume \
            > work_dirs/f1cv/mae_fold${F}.log 2>&1
        CK=work_dirs/f1cv/mae_fold${F}/epoch_1000.pth
        [ -f "$CK" ] || { echo "!!! fold $F: no se generó el MAE — salteo"; continue; }
        python -u extract_mae_encoder.py $CK $ENC >> work_dirs/f1cv/mae_fold${F}.log 2>&1
    fi
    echo "[fold $F] encoder listo"

    VAL=$(python3 -c "
import re;t=open('$D/f1cv_mae_fold${F}.py').read()
print(' '.join(v.strip().strip(\"'\") for v in re.search(r'val RETENIDA del fold \d+: \[(.*?)\]',t).group(1).split(',')))")
    [ -n "$VAL" ] || { echo "!!! fold $F: no pude leer las escenas de validacion — salteo"; continue; }

    for S in 0 1 2 3 4 5 6 7; do
        for V in baseline gate0 gated; do
            case $V in
              baseline) CFG=$D/f1cv_base_fold${F}.py; OPT="" ;;
              gate0)    CFG=$D/f1cv_dec_fold${F}.py;  OPT="model.gate_init=0.0 model.freeze_gate=True" ;;
              gated)    CFG=$D/f1cv_dec_fold${F}.py;  OPT="model.gate_init=0.5" ;;
            esac
            WD=work_dirs/f1cv/${V}_f${F}s${S}
            NUEVO=0
            if [ ! -f "$WD/epoch_100.pth" ]; then
                NUEVO=1
                python -u tools/train.py $CFG --work-dir $WD \
                    --cfg-options randomness.seed=$S $OPT > $WD.log 2>&1 \
                    || { echo "!!! falló $V fold $F seed $S"; continue; }
            fi
            ya_evaluado $CSV $F $V $S $(echo $VAL | wc -w) || python -u eval_fase1_seeds.py --cfg $CFG --ckpt $WD/epoch_100.pth \
                --variant $V --seed $S --fold $F --val-scenes $VAL --out $CSV \
                2>&1 | grep "^\[eval\]"
        done
    done
    echo "######## FOLD $F — fin $(date '+%d/%m %H:%M') ########"
done
echo "=== FASE 1 CV COMPLETA ==="
