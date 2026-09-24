#!/bin/bash
# run_rv_nativo.sh — range-view a RESOLUCION NATIVA, 5 folds (exp. 31).
#
# POR QUE. La range-view nativa es 64 x 2650 a 0,136 grados por columna. El
# pipeline la diezmaba con AZ_STRIDE=5 y la recortaba a 512 columnas: 128 tokens,
# MENOS que los 300 de voxeles, tirando el 80 % de la resolucion angular.
#
# Y tiene dos ventajas sobre los voxeles que estan MEDIDAS:
#   objeto dentro de la escena : 96,6 %  (voxeles +-10 m: 11,0 % — el defecto del exp. 28)
#   un auto a 33 m ocupa       : 57 x 8 px nativos  (voxeles de 2 m: 2,2 x 1)
# Ese 57 x 8 es del mismo orden que el 45 x 20 de Occupancy-MAE, el paper cuyo
# MAE de voxeles SI funciona (para deteccion).
#
# Con az_stride=1 y los mismos parches de 16: 660 tokens, solo 2,2x los 300 de
# voxeles. Es el mejor rendimiento por token de todas las opciones evaluadas.
#
# LO QUE ESTE EXPERIMENTO CIERRA. Hoy la tesis dice "la escena no aporta" y la
# objecion obvia es: "lo mediste con voxeles de 2 m donde un auto ocupa dos
# celdas y el objeto ni estaba en la caja". Despues de esto la respuesta es "si,
# a resolucion angular completa, con el objeto presente en el 96,6 % de los
# casos, 5 folds y 8 semillas". Se corre por eso, no porque se espere que
# funcione: el gate cerro en los 30 experimentos anteriores.
#
# LA SENYAL A MIRAR es el GATE, no el ADE. Si con range-view nativo se estabiliza
# en 0,1-0,2 en vez de ~0,003, seria la primera vez que el modelo encuentra algo
# util en la escena. Se ve ya en los dos primeros folds.
#
# TRES LUGARES tienen que declarar la MISMA tokenizacion, y por eso van juntos en
# variables aca abajo: el dataset (az_stride), el backbone (num_tokens) y el neck
# (num_patches). Los dos primeros ahora fallan ruidosamente si no coinciden —ver
# _ensure_pos_embed, que antes reemplazaba el pos_embed en silencio y ademas lo
# descongelaba—; el tercero revienta con un RuntimeError de shapes.
#
# ETAPA 1: 5 encoders MAE de range-view a 660 tokens. Medido: 0,4 h cada uno
# (el historial decia 12,5 h; ese dato estaba desactualizado por 30x).
# ETAPA 2: decoder, 5 folds x 8 semillas x 2 variantes.
#   gated_rv  gate aprendible (gate_init=0.05, la version corregida del exp. 20)
#   gate0_rv  gate congelado en 0: la escena NO llega. Control de arquitectura.
# COSTO, corregido. Las 16 corridas de cada fold cuestan 28,3 min CADA UNA,
# tambien las de gate0. Yo habia estimado 4,7 min para gate0 suponiendo que el
# gate en 0 evitaba el computo de la escena: NO lo evita. _encode_scene() se
# ejecuta siempre y el gate solo multiplica su SALIDA por cero, asi que el ViT
# procesa los 660 tokens igual.
#
# Tampoco se puede reusar el gate0 del track de voxeles, aunque alli gate0_obj
# reprodujo gate0 hasta el ultimo decimal. Medido: construir el modelo con
# num_tokens=300 y con 660 bajo la MISMA semilla da decoders distintos
# (max|dif| = 0,22 en la primera capa), porque el pos_embed de otro tamanyo
# consume otra cantidad de numeros aleatorios y desplaza el RNG. En voxeles
# funcionaba porque los dos brazos tenian 300 tokens y solo cambiaban los datos.
#
#   5 encoders x 26 min                       =  2,2 h
#   5 folds x 16 corridas x 28,3 min          = 37,7 h
#   TOTAL ~40 h.
cd /home/lcad/lidar_sweep_viewer/sapiens/pretrain
source /home/lcad/miniconda3/etc/profile.d/conda.sh
conda activate sapiens_gpu

AZ=1            # sin diezmar el azimut
NTOK=660        # (64/16) * (2640/16)

ya_evaluado() {
    [ -f "$1" ] || return 1
    [ "${5:-1}" -gt 0 ] || return 1
    local n; n=$(grep -c "^$2,$3,$4," "$1")
    [ "$n" -ge "$5" ]
}

D=configs/sapiens_mae/lidar
CSV=work_dirs/rv_nativo/rv_nativo_results.csv
mkdir -p work_dirs/mae_rv_nativo work_dirs/rv_nativo

echo "######## ETAPA 1 — encoders de range-view a $NTOK tokens ########"
for F in 0 1 2 3 4; do
    CFG=$D/mae_rangeview_pretrain.py; [ $F -gt 0 ] && CFG=$D/mae_rangeview_fold${F}.py
    WD=work_dirs/mae_rv_nativo/mae_fold$F
    ENC=work_dirs/mae_rv_nativo/enc_fold${F}.pth
    if [ ! -f "$ENC" ]; then
        [ -f "$WD/epoch_1000.pth" ] || \
            python -u tools/train.py $CFG --work-dir $WD \
                --cfg-options train_dataloader.dataset.az_stride=$AZ \
                              model.backbone.num_tokens=$NTOK \
                              model.neck.num_patches=$NTOK \
                              default_hooks.checkpoint.max_keep_ckpts=1 \
                > $WD.log 2>&1 \
            || { echo "!!! fallo pre-entrenando fold $F — ver $WD.log"; continue; }
        python -u extract_mae_encoder.py $WD/epoch_1000.pth $ENC > /dev/null 2>&1 \
            || { echo "!!! fallo extrayendo encoder del fold $F"; continue; }
    fi
    echo "----- encoder fold $F listo ($(date '+%d/%m %H:%M')) -----"
done

echo "######## ETAPA 2 — decoder con range-view nativa ########"
for F in 0 1 2 3 4; do
    ENC=work_dirs/mae_rv_nativo/enc_fold${F}.pth
    [ -f "$ENC" ] || { echo "!!! fold $F sin encoder — salteo"; continue; }
    VAL=$(python3 -c "
import re
t = open('$D/f1cv_mae_fold${F}.py').read()
m = re.search(r'val RETENIDA del fold \d+: \[(.*?)\]', t)
print(' '.join(re.findall(r\"'([0-9a-f]{16})'\", m.group(1))))")
    [ -n "$VAL" ] || { echo "!!! fold $F sin escenas de validacion — salteo"; continue; }
    NV=$(echo $VAL | wc -w)
    echo "######## FOLD $F — inicio $(date '+%d/%m %H:%M') ########"

    for PAR in "gate0_rv:model.gate_init=0.0 model.freeze_gate=True" \
               "gated_rv:model.gate_init=0.05"; do
        V=${PAR%%:*}; OPT=${PAR#*:}
        DS="train_dataloader.dataset.az_stride=$AZ model.encoder.num_tokens=$NTOK"
        for S in 0 1 2 3 4 5 6 7; do
            WD=work_dirs/rv_nativo/${V}_f${F}s${S}
            [ -f "$WD/epoch_100.pth" ] || \
                python -u tools/train.py $D/rvcv_dec_fold${F}.py --work-dir $WD \
                    --cfg-options randomness.seed=$S $DS $OPT \
                                  default_hooks.checkpoint.max_keep_ckpts=1 \
                    > $WD.log 2>&1 \
                || { echo "!!! fallo entrenando $V fold $F semilla $S — ver $WD.log"; continue; }
            ya_evaluado $CSV $F $V $S $NV || {
                python -u eval_fase1_seeds.py --cfg $D/rvcv_dec_fold${F}.py \
                    --ckpt $WD/epoch_100.pth --variant $V --seed $S --fold $F \
                    --val-scenes $VAL --eval-windows 7 --sin-clip \
                    --cfg-options $DS $OPT --out $CSV > $WD.eval.log 2>&1 \
                    || echo "!!! fallo evaluando $V fold $F semilla $S — ver $WD.eval.log"
                grep "^\[eval\]" $WD.eval.log
            }
        done
        echo "----- fold $F  $V listo ($(date '+%d/%m %H:%M')) -----"
    done
done

echo "=== RANGE-VIEW NATIVA COMPLETO ==="
echo "  la escena aporta?"
echo "    python agregar_resultados.py $CSV --comparar gated_rv:gate0_rv --por-fold"
echo "  el gate (la senyal que importa):"
echo "    awk -F, 'NR>1 && \$2==\"gated_rv\"{s[\$1]+=\$11;n[\$1]++} END{for(f in s) printf \"  fold %s: %+.4f\\n\", f, s[f]/n[f]}' $CSV | sort"
