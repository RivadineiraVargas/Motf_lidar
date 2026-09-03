#!/bin/bash
# run_clsweight.sh — la clasificación ahoga a la regresión: ¿se arregla bajándole el peso?
#
# EL DIAGNÓSTICO. En la corrida multimodal (run_multimodal.sh, cls_weight=1.0) el
# brazo baseline dio, sobre los 5 folds x 8 semillas ya COMPLETOS:
#     ADE del modo más probable   k=1 2,988  ->  k=6 3,285   (+0,298, p=0,036, 0/5 folds)
#     minADE_6                    k=1 2,988  ->  k=6 2,264   (-24,2%, p=0,005, 5/5 folds)
# Las dos filas dicen cosas opuestas, y la primera es la que importa: la
# predicción REAL empeoró mientras la métrica de la literatura mejoraba.
#
# LA CAUSA, leída de las curvas de entrenamiento:
#     época   1:  wta_reg 0,1165   wta_cls 1,5961
#     época  25:  wta_reg 0,0149   wta_cls 1,2856
#     época 100:  wta_reg 0,0365   wta_cls 1,4535
# `cls` es ~40x mayor que `reg`, así que con cls_weight=1.0 el gradiente lo domina
# la clasificación. Peor: la regresión EMPEORA después de la época 25 (0,0149 ->
# 0,0365) — el modelo desaprende a predecir mientras persigue clasificar. Y `cls`
# se queda en ~1,45 contra el 1,79 del azar puro (log 6): el clasificador casi no
# aprende.
#
# QUE EXPLICA EL RESULTADO. El winner-takes-all SI especializa los modos —por eso
# minADE_6 = 2,264 es mucho mejor que k=1—, o sea que las hipótesis buenas ESTAN
# AHI. Pero al predecir se elige por argmax de los logits, y ese clasificador no
# distingue cuál sirve. La brecha 3,285 vs 2,264 es exactamente el costo de elegir
# mal, no de predecir mal.
#
# EL BARRIDO. Solo cls_weight; todo lo demás idéntico. 3 pesos x 2 folds x 4
# semillas = 24 corridas de ~1,6 min = ~40 min. cls_weight=1.0 ya está medido en
# work_dirs/multimodal, así que sirve de cuarto punto sin volver a correrlo.
#
# OJO CON EL SESGO DE SELECCION: esto es un BARRIDO para elegir un
# hiperparámetro, no un resultado. El peso que gane hay que validarlo después
# sobre los 5 folds completos. Elegir y reportar sobre los mismos folds seria
# justo el error que este proyecto ya cometió (ver la regla 2 del CLAUDE.md).
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
CSV=work_dirs/clsweight/clsweight_results.csv
mkdir -p work_dirs/clsweight

for F in 0 1; do
    VAL=$(python3 -c "
import re
t = open('$D/f1cv_mae_fold${F}.py').read()
m = re.search(r'val RETENIDA del fold \d+: \[(.*?)\]', t)
print(' '.join(re.findall(r\"'([0-9a-f]{16})'\", m.group(1))))")
    [ -n "$VAL" ] || { echo "!!! fold $F sin escenas de validacion — salteo"; continue; }
    NV=$(echo $VAL | wc -w)
    echo "######## FOLD $F — inicio $(date '+%d/%m %H:%M') ########"

    for W in 0.01 0.05 0.2; do
        # nombre de variante sin punto: el CSV se filtra con grep "^fold,variante,"
        V="k6w$(echo $W | tr -d '.')"
        for S in 0 1 2 3; do
            WD=work_dirs/clsweight/${V}_f${F}s${S}
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
        echo "----- fold $F  cls_weight=$W listo ($(date '+%d/%m %H:%M')) -----"
    done
done

echo "=== BARRIDO DE cls_weight COMPLETO ==="
echo "Contra el k=1 y el cls_weight=1.0 de la corrida principal:"
echo "  python agregar_resultados.py $CSV work_dirs/multimodal/multimodal_results.csv \\"
echo "      --por-fold --comparar k6w001:baseline_k1 k6w005:baseline_k1 k6w02:baseline_k1"
echo "Y la brecha entre lo que elegimos y lo mejor disponible:"
echo "  python agregar_resultados.py $CSV --metrica minade"
