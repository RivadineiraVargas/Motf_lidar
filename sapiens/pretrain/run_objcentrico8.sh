#!/bin/bash
# run_objcentrico8.sh — replica el exp. 28 con las semillas 4-7.
#
# POR QUE. El exp. 28 midio, sobre 5 folds x 4 semillas, que centrar la caja de
# voxeles en el objeto mejora -0,290 +- 0,229 con p=0,0475 y 5/5 folds. Ese p
# esta JUSTO bajo el umbral, y este proyecto ya vio dos veces un efecto asi darse
# vuelta al validarlo (el cls_weight=0,05 del exp. 25, que daba -7,6 % y -8,1 %
# con los dos folds de acuerdo y en los folds retenidos salio +0,224).
#
# Lo que da confianza en el exp. 28 no es la p sino los 5/5 folds y que haya un
# mecanismo medido detras —el objeto dentro de la caja pasa del 11,0 % al
# 100,0 %—, no una hipotesis post hoc. Pero con n=5 folds y p=0,047 no alcanza
# para darlo por establecido. Duplicar las semillas reduce el ruido DENTRO de
# cada fold, que es lo que hace ruidoso el efecto por fold.
#
# CSV APARTE. objcentrico_results.csv respalda el exp. 28 ya documentado y
# comiteado con n=4 semillas; agregarle filas cambiaria un resultado publicado.
# Este escribe en objcentrico8_results.csv y el analisis final pasa LOS DOS al
# agregador, que los une por (fold, variante, semilla).
#
# max_keep_ckpts=1: el evaluador solo usa epoch_100. Con el default (2) cada
# corrida deja ademas un epoch_90 que nadie mira — 48 GB de basura en las 40
# corridas del exp. 28, que hubo que borrar despues.
#
# n = 5 folds x 4 semillas (4..7) x 2 variantes = 40 corridas de ~12,4 min = ~8,3 h.
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
CSV=work_dirs/objcentrico8/objcentrico8_results.csv
mkdir -p work_dirs/objcentrico8
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
        for S in 4 5 6 7; do
            WD=work_dirs/objcentrico8/${V}_f${F}s${S}
            [ -f "$WD/epoch_100.pth" ] || \
                python -u tools/train.py $D/noclip_dec_fold${F}.py --work-dir $WD \
                    --cfg-options randomness.seed=$S $OBJ $OPT \
                                  default_hooks.checkpoint.max_keep_ckpts=1 \
                    > $WD.log 2>&1 \
                || { echo "!!! fallo entrenando $V fold $F semilla $S"; continue; }
            ya_evaluado $CSV $F $V $S $NV || {
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

echo "=== REPLICA CON 8 SEMILLAS COMPLETA ==="
echo "El resultado de 8 semillas une LOS DOS CSV:"
echo "  python agregar_resultados.py work_dirs/objcentrico/objcentrico_results.csv \\"
echo "      $CSV --comparar gated_obj:gate0_obj --por-fold"
