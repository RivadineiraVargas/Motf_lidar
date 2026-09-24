#!/bin/bash
# run_multimodal.sh — ¿predecir K hipótesis mejora la predicción, o solo la vara?
#
# LA PREGUNTA. Wayformer, MTR y MotionLM predicen 6 modos y reportan minADE;
# nosotros predecíamos UNA trayectoria. El futuro es genuinamente multimodal —el
# auto dobla o sigue— y un modelo de k=1 aprende el PROMEDIO de los futuros
# posibles, que no es ninguno de ellos. El plan de Claudine lo pide en este orden
# (Sec. 10): "k=1 al inicio; k>1 con winner-takes-all después".
#
# LO QUE HAY QUE MIRAR, y por qué son DOS métricas:
#   ade_all    — error del modo MÁS PROBABLE. Comparable con los experimentos
#                15-22 y con el baseline. **Es el que dice si mejoramos.**
#   minade_all — el mejor de los 6. La métrica de WOMD, comparable con la
#                literatura. NO sirve para decir si mejoramos: con K modos el
#                mínimo siempre es menor o igual. Medido con un modelo de DOS
#                épocas —o sea basura— el minADE_6 ya salía 20-40% mejor que el
#                modo elegido. Reportar solo minADE es premiar el cambio de vara.
#
# EL DISEÑO. Mismos 5 folds x 8 semillas, misma época fija 100, misma evaluación
# (--sin-clip --eval-windows 7) que la CV del experimento 19. Cambia UNA variable:
# num_modes 1 -> 6. Dos arquitecturas:
#   baseline — MLP cinemático, sin escena. ~1,6 min por corrida.
#   gate0    — modelo completo con la escena CONGELADA en 0. Es el que hoy tiene
#              el mejor ADE del proyecto (2,781), así que es contra quien hay que
#              medir. ~12,7 min por corrida.
#
# POR QUÉ SE RE-EVALÚAN LOS BRAZOS k=1. Sus checkpoints ya existen (noclipcv) y su
# ade_all es idéntico al quinto decimal —verificado: 2.84464 contra 2.84464—, así
# que NO se re-entrena nada. Se re-evalúan para que las cuatro variantes queden en
# UN SOLO CSV con el mismo esquema de 15 columnas. Comparar entre CSV de corridas
# distintas es lo que ya costó caro en este proyecto.
#
# cls_weight=1.0 (el default): al arrancar, la pérdida de clasificación vale
# log(6)=1,79 y la de regresión ~1,0, o sea que son comparables y ninguna ahoga a
# la otra. Si al mirar las curvas wta_cls domina y wta_reg no baja, ESE es el
# parámetro a tocar, no la arquitectura.
#
# COSTO: 40 x 1,6 min (baseline_k6) + 40 x 12,7 min (gate0_k6) + 80 evaluaciones
#        ~= 10 h. El baseline va PRIMERO para tener señal temprana.
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
CSV=work_dirs/multimodal/multimodal_results.csv
mkdir -p work_dirs/multimodal

# evaluar <variante> <config> <checkpoint> <fold> <semilla> <nº escenas> [opciones]
evaluar() {
    local V=$1 CFG=$2 CK=$3 F=$4 S=$5 NV=$6; shift 6
    ya_evaluado $CSV $F $V $S $NV && return 0
    local LOG=work_dirs/multimodal/${V}_f${F}s${S}.eval.log
    python -u eval_fase1_seeds.py --cfg $CFG --ckpt $CK --variant $V --seed $S \
        --fold $F --val-scenes $VAL --eval-windows 7 --sin-clip "$@" \
        --out $CSV > $LOG 2>&1 \
        || echo "!!! fallo evaluando $V fold $F semilla $S — ver $LOG"
    grep "^\[eval\]" $LOG
}

for ARCH in baseline gate0; do
  case $ARCH in
    baseline) CFG=$D/noclip_base_fold ; OPT_K1="" ;;
    gate0)    CFG=$D/noclip_dec_fold  ; OPT_K1="model.gate_init=0.0 model.freeze_gate=True" ;;
  esac
  echo "######## $ARCH — inicio $(date '+%d/%m %H:%M') ########"

  for F in 0 1 2 3 4; do
    VAL=$(python3 -c "
import re
t = open('$D/f1cv_mae_fold${F}.py').read()
m = re.search(r'val RETENIDA del fold \d+: \[(.*?)\]', t)
print(' '.join(re.findall(r\"'([0-9a-f]{16})'\", m.group(1))))")
    [ -n "$VAL" ] || { echo "!!! fold $F: sin escenas de validacion — salteo"; continue; }
    NV=$(echo $VAL | wc -w)

    for S in 0 1 2 3 4 5 6 7; do
        # --- k=6: se entrena ---
        WD=work_dirs/multimodal/${ARCH}_k6_f${F}s${S}
        [ -f "$WD/epoch_100.pth" ] || \
            python -u tools/train.py ${CFG}${F}.py --work-dir $WD \
                --cfg-options randomness.seed=$S model.num_modes=6 $OPT_K1 \
                > $WD.log 2>&1 \
            || { echo "!!! fallo entrenando ${ARCH}_k6 fold $F semilla $S"; continue; }
        evaluar ${ARCH}_k6 ${CFG}${F}.py $WD/epoch_100.pth $F $S $NV \
                --cfg-options model.num_modes=6

        # --- k=1: checkpoint YA entrenado en noclipcv, solo se re-evalua ---
        CK1=work_dirs/noclipcv/${ARCH}_f${F}s${S}/epoch_100.pth
        [ -f "$CK1" ] || { echo "!!! falta $CK1 — sin referencia k=1 para f$F s$S"; continue; }
        evaluar ${ARCH}_k1 ${CFG}${F}.py $CK1 $F $S $NV
    done
    echo "----- $ARCH fold $F listo ($(date '+%d/%m %H:%M')) -----"
  done
  echo "######## $ARCH — fin $(date '+%d/%m %H:%M') ########"
done

echo "=== MULTIMODAL COMPLETO ==="
echo "Lo que dice si MEJORAMOS (modo más probable, comparable con los exp. 15-22):"
echo "  python agregar_resultados.py $CSV --por-fold \\"
echo "      --comparar baseline_k6:baseline_k1 gate0_k6:gate0_k1"
echo "Lo comparable con la literatura (minADE_6, NO sirve para comparar k6 vs k1):"
echo "  python agregar_resultados.py $CSV --por-fold --metrica minade"
