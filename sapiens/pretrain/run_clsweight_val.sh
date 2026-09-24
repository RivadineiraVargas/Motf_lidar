#!/bin/bash
# run_clsweight_val.sh — validar cls_weight=0,05 SIN el sesgo de haberlo elegido.
#
# DE DONDE VIENE EL 0,05. El barrido de run_clsweight.sh probó 0,01 / 0,05 / 0,2
# sobre los folds 0 y 1 (4 semillas cada uno); el 1,0 ya estaba medido en
# work_dirs/multimodal. Efecto sobre el ADE del modo más probable contra k=1:
#     0,01   +1,122   0/2 folds
#     0,05   -0,264   2/2 folds   (-7,6 % fold 0, -8,1 % fold 1)
#     0,2    +0,020   0/2 folds
#     1,0    +0,298   0/5 folds
# El 0,05 es el único que le gana al k=1, y su efecto relativo es casi idéntico en
# los dos folds. Pero con n=2 folds NO puede haber significancia (p=0,19), y sobre
# todo: esos son los folds donde se lo ELIGIÓ. Reportar ahí sería sesgo de
# selección — el error de la regla 2 del CLAUDE.md.
#
# EL DISEÑO. Dos poblaciones, y se reportan por separado:
#
#   A) TEST INDEPENDIENTE — folds 2, 3 y 4, que nunca participaron de la elección.
#      Es el número defendible. n = 3 folds x 8 semillas.
#
#   B) TABLA COMPLETA — se agregan las semillas 4-7 de los folds 0 y 1 para poder
#      publicar los 5 folds. Ese número lleva SIEMPRE la advertencia de que los
#      folds 0-1 se usaron para elegir el hiperparámetro.
#
# Si (A) no da el mismo signo que el barrido, el 0,05 era ruido y se descarta. Con
# n=3 folds la significancia sigue siendo difícil; lo que se mira es el SIGNO y la
# consistencia entre folds, no solo la p.
#
# Solo se entrena el k6w005 (el baseline_k1 se reusa de work_dirs/multimodal):
# 3 folds x 8 semillas + 2 folds x 4 semillas = 32 corridas de ~1,8 min = ~58 min.
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
CSV=work_dirs/clsweight_val/clsweight_val_results.csv
mkdir -p work_dirs/clsweight_val
V=k6w005
W=0.05

# folds 2-4: las 8 semillas (test independiente).  folds 0-1: solo 4-7, porque
# 0-3 ya están medidas en work_dirs/clsweight y no hay que repetirlas.
corridas() {
    case $1 in
        0|1) echo "4 5 6 7" ;;
        *)   echo "0 1 2 3 4 5 6 7" ;;
    esac
}

for F in 2 3 4 0 1; do
    VAL=$(python3 -c "
import re
t = open('$D/f1cv_mae_fold${F}.py').read()
m = re.search(r'val RETENIDA del fold \d+: \[(.*?)\]', t)
print(' '.join(re.findall(r\"'([0-9a-f]{16})'\", m.group(1))))")
    [ -n "$VAL" ] || { echo "!!! fold $F sin escenas de validacion — salteo"; continue; }
    NV=$(echo $VAL | wc -w)
    echo "######## FOLD $F — inicio $(date '+%d/%m %H:%M') ########"

    # Solo se entrena el k6w005. El baseline_k1 ya está medido en los 5 folds x 8
    # semillas en work_dirs/multimodal, con ESTE mismo config y estas mismas
    # semillas; el agregador une los dos CSV y el pareo por (fold, semilla) sale
    # igual. Re-correrlo duplicaría el costo sin agregar información.
    for S in $(corridas $F); do
        WD=work_dirs/clsweight_val/${V}_f${F}s${S}
        [ -f "$WD/epoch_100.pth" ] || \
            python -u tools/train.py $D/noclip_base_fold${F}.py --work-dir $WD \
                --cfg-options randomness.seed=$S model.num_modes=6 model.cls_weight=$W \
                > $WD.log 2>&1 \
            || { echo "!!! fallo entrenando $V fold $F semilla $S"; continue; }
        ya_evaluado $CSV $F $V $S $NV || {
            python -u eval_fase1_seeds.py --cfg $D/noclip_base_fold${F}.py \
                --ckpt $WD/epoch_100.pth --variant $V --seed $S --fold $F \
                --val-scenes $VAL --eval-windows 7 --sin-clip \
                --cfg-options model.num_modes=6 --out $CSV > $WD.eval.log 2>&1 \
                || echo "!!! fallo evaluando $V fold $F semilla $S — ver $WD.eval.log"
            grep "^\[eval\]" $WD.eval.log
        }
    done
    echo "----- fold $F listo ($(date '+%d/%m %H:%M')) -----"
done

echo "=== VALIDACION COMPLETA ==="
echo "A) TEST INDEPENDIENTE (folds 2,3,4 — nunca usados para elegir):"
echo "   python agregar_resultados.py $CSV work_dirs/multimodal/multimodal_results.csv \\
       --comparar k6w005:baseline_k1 --por-fold   # y mirar solo los folds 2,3,4"
echo "B) Tabla completa de 5 folds (folds 0-1 SESGADOS por la eleccion):"
echo "   python agregar_resultados.py $CSV work_dirs/clsweight/clsweight_results.csv \\"
echo "       work_dirs/multimodal/multimodal_results.csv --comparar k6w005:baseline_k1 --por-fold"
