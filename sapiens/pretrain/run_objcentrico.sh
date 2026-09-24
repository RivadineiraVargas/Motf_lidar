#!/bin/bash
# run_objcentrico.sh — ¿la escena aporta cuando CONTIENE al objeto?
#
# LO QUE SE MIDIO Y ABRE ESTE EXPERIMENTO. La caja de vóxeles de Fase 1 cubre
# ±10 m alrededor del EGO. Medido sobre las 236 ventanas del fold 0, el objeto a
# predecir está a 32,7 m del ego (mediana) y solo el 11 % de las ventanas lo
# tienen dentro de la caja durante toda su historia (el futuro completo, 7,2 %).
#
# O sea que en el 89 % de los casos el encoder mira una región que NO CONTIENE al
# objeto. Eso explica de una sola vez los cuatro negativos:
#   exp. 19-20  la escena no aporta, el gate cierra a 0,0042   -> no hay objeto que ver
#   exp. 19     más capacidad no ayuda (p=0,102)               -> capacidad sobre lo irrelevante
#   exp. 22     la historia completa no ayuda                  -> más frames de lo irrelevante
#   exp. 27     la reconstrucción no predice el ADE (r=+0,34)  -> reconstruye el entorno del EGO
#
# EL CAMBIO. `centrar_en_objeto=True` traslada la nube por -centers[0] antes de
# voxelizar: mismos 300 tokens, mismo costo, pero el objeto queda en el centro de
# su propia escena — que es lo que hacen Wayformer, MTR y BEVTraj. Verificado:
# el objeto pasa de estar dentro en el 11,0 % de las ventanas al 100,0 %, la
# ocupación baja de 35,9 % a 27,4 % y ninguna grilla queda vacía. La trayectoria
# no cambia (max|dif| = 0): la única diferencia entre brazos es la ESCENA.
#
# EL CONTROL. Mismo diseño que el exp. 19-20, para que los números sean
# comparables fila a fila:
#   gate0_obj  gate congelado en 0 -> la escena NO llega al decoder (control de
#              arquitectura: misma capacidad, sin escena)
#   gated_obj  gate aprendible con gate_init=0.05 -> la versión CORREGIDA del
#              exp. 20; con 0.5 la rama entra a media amplitud, cuesta media
#              corrida de entrenamiento e infla el efecto al triple
#
# LA REFERENCIA A BATIR (ego-céntrico, ya medido, 5 folds x 8 semillas):
#   gated    - gate0  = +0,723  p=0,038  0/5 folds   <- la escena PERJUDICA
#   gated005 - gate0  = +0,276  p=0,139  0/5 folds
# Si con la caja centrada en el objeto ese signo se da vuelta, la pregunta
# central de la tesis cambia de respuesta.
#
# EL LIMITE, DICHO DE ANTEMANO. El encoder MAE sigue siendo el ego-céntrico:
# LidarSequenceDataset no conoce los objetos, así que centrar el
# PRE-ENTRENAMIENTO es un cambio aparte y mayor. Hay entonces un desajuste de
# dominio — el encoder aprendió sobre grillas densas cerca del sensor (35,9 %) y
# acá recibe grillas dispersas y lejanas (27,4 %). Eso juega EN CONTRA de este
# experimento: si aun así la escena empieza a aportar, la conclusión es más
# fuerte, no más débil. Y el exp. 27 da razones para creer que el desajuste pesa
# poco (la calidad del encoder no predijo el ADE, r=+0,34) — aunque midió
# variación DENTRO de un dominio, no un cambio de dominio.
#
# n = 5 folds x 4 semillas x 2 variantes = 40 corridas de ~12,4 min = ~8,3 h.
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
CSV=work_dirs/objcentrico/objcentrico_results.csv
mkdir -p work_dirs/objcentrico
OBJ=train_dataloader.dataset.centrar_en_objeto=True

for F in 0 1 2 3 4; do
    VAL=$(python3 -c "
import re
t = open('$D/f1cv_mae_fold${F}.py').read()
m = re.search(r'val RETENIDA del fold \d+: \[(.*?)\]', t)
print(' '.join(re.findall(r\"'([0-9a-f]{16})'\", m.group(1))))")
    [ -n "$VAL" ] || { echo "!!! fold $F sin escenas de validacion — salteo"; continue; }
    NV=$(echo $VAL | wc -w)
    echo "######## FOLD $F — inicio $(date '+%d/%m %H:%M') ########"

    for PAR in "gate0_obj:model.gate_init=0.0 model.freeze_gate=True" \
               "gated_obj:model.gate_init=0.05"; do
        V=${PAR%%:*}; OPT=${PAR#*:}
        for S in 0 1 2 3; do
            WD=work_dirs/objcentrico/${V}_f${F}s${S}
            [ -f "$WD/epoch_100.pth" ] || \
                python -u tools/train.py $D/noclip_dec_fold${F}.py --work-dir $WD \
                    --cfg-options randomness.seed=$S $OBJ $OPT \
                    > $WD.log 2>&1 \
                || { echo "!!! fallo entrenando $V fold $F semilla $S"; continue; }
            ya_evaluado $CSV $F $V $S $NV || {
                # OJO: el flag va tambien en la EVALUACION. Sin esto se entrenaria
                # con la caja del objeto y se mediria con la del ego.
                python -u eval_fase1_seeds.py --cfg $D/noclip_dec_fold${F}.py \
                    --ckpt $WD/epoch_100.pth --variant $V --seed $S --fold $F \
                    --val-scenes $VAL --eval-windows 7 --sin-clip \
                    --cfg-options $OBJ $OPT --out $CSV > $WD.eval.log 2>&1 \
                    || echo "!!! fallo evaluando $V fold $F semilla $S — ver $WD.eval.log"
                grep "^\[eval\]" $WD.eval.log
            }
        done
        echo "----- fold $F  $V listo ($(date '+%d/%m %H:%M')) -----"
    done
done

echo "=== OBJETO-CENTRICO COMPLETO ==="
echo "  python agregar_resultados.py $CSV --comparar gated_obj:gate0_obj --por-fold"
echo "Contra la referencia ego-centrica del exp. 19-20:"
echo "  python agregar_resultados.py work_dirs/noclipcv/noclipcv_results.csv \\"
echo "      work_dirs/gateinit/gateinit_results.csv --comparar gated005:gate0 --por-fold"
