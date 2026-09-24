#!/bin/bash
# run_densidad.sh — ¿enriquecer la representacion de entrada hace que la escena
# aporte?
#
# EL ESLABON QUE NUNCA SE TOCO. De los tres —representacion, encoder, consumo en
# el decoder— el encoder es el UNICO medido y funciona (exp. 21: generaliza,
# 43,5 % mejor que trivial en escenas retenidas). Los otros dos no se tocaron en
# 28 experimentos.
#
# LA ENTRADA SON 1.500 BITS (trampa 32). 300 voxeles x 5 frames de ocupacion
# BINARIA. Medido sobre 2.230 voxeles ocupados de 25 ventanas del fold 0, los
# puntos por voxel van (percentiles 10/25/50/75/90/99):
#     2 / 7 / 20 / 62 / 202 / 1.711        maximo 5.395
# El 6,6 % tiene un solo punto y el 67,5 % mas de diez. Un voxel con 1 punto y
# otro con 5.395 valen HOY exactamente lo mismo: 1,0. Se colapsan cuatro ordenes
# de magnitud a un bit, y se comprimen ~6.345 puntos LiDAR a 1.500 bits.
#
# EL CAMBIO. `densidad=True`: el voxel guarda log1p(n)/log1p(1000), recortado a 1.
# Escala logaritmica porque el rango abarca cuatro ordenes; FIJA y no normalizada
# por muestra porque si no el mismo voxel valdria distinto segun que mas haya en
# la escena. Verificado: los voxeles ocupados coinciden 100 % con el binario y se
# pasa de 1 valor unico a 572 distintos, con solo 1,1 % saturando.
#
# NO CAMBIA LA FORMA de los tokens —sigue siendo (300, 5)— asi que patch_embed
# = Linear(history_len, embed_dim) no se toca y los checkpoints del encoder
# siguen cargando. Por eso se puede medir SIN re-pre-entrenar el MAE.
#
# LA COMPARACION. Solo se entrena gated_dens: el brazo gated_obj (binario) ya
# tiene 8 semillas medidas en work_dirs/objcentrico{,8}, con este mismo config y
# estas mismas semillas, y gate0 da lo mismo con o sin densidad porque el gate
# congelado en 0 anula la escena. El agregador une los CSV y el pareo por
# (fold, semilla) sale igual. Eso ahorra la mitad del computo.
#
#   gated_dens vs gated_obj   -> aporta la densidad?
#   gated_dens vs gate0_obj   -> con densidad, la escena finalmente aporta?
#
# OCHO SEMILLAS DESDE EL ARRANQUE. El exp. 28 se corrio con 4, dio p=0,047 y la
# replica con 8 lo dejo en p=0,086. No repetir ese camino.
#
# EL LIMITE. El MAE fue pre-entrenado sobre entradas BINARIAS; darle densidades
# continuas es un cambio de distribucion de entrada y puede degradar aunque la
# informacion sea mayor. Si no mejora, esa es la PRIMERA hipotesis a descartar
# re-pre-entrenando con densidad (20 min por fold, medido).
#
# n = 5 folds x 8 semillas = 40 corridas de ~12,4 min = ~8,3 h.
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
CSV=work_dirs/densidad/densidad_results.csv
mkdir -p work_dirs/densidad
OPT_DS="train_dataloader.dataset.centrar_en_objeto=True train_dataloader.dataset.densidad=True"

for F in 0 1 2 3 4; do
    VAL=$(python3 -c "
import re
t = open('$D/f1cv_mae_fold${F}.py').read()
m = re.search(r'val RETENIDA del fold \d+: \[(.*?)\]', t)
print(' '.join(re.findall(r\"'([0-9a-f]{16})'\", m.group(1))))")
    [ -n "$VAL" ] || { echo "!!! fold $F sin escenas — salteo"; continue; }
    NV=$(echo $VAL | wc -w)
    echo "######## FOLD $F — inicio $(date '+%d/%m %H:%M') ########"
    for S in 0 1 2 3 4 5 6 7; do
        WD=work_dirs/densidad/gated_dens_f${F}s${S}
        [ -f "$WD/epoch_100.pth" ] || \
            python -u tools/train.py $D/noclip_dec_fold${F}.py --work-dir $WD \
                --cfg-options randomness.seed=$S $OPT_DS model.gate_init=0.05 \
                              default_hooks.checkpoint.max_keep_ckpts=1 \
                > $WD.log 2>&1 \
            || { echo "!!! fallo entrenando fold $F semilla $S"; continue; }
        ya_evaluado $CSV $F gated_dens $S $NV || {
            # el flag de densidad va TAMBIEN en la evaluacion: sin esto se
            # entrenaria con densidad y se mediria con binario.
            python -u eval_fase1_seeds.py --cfg $D/noclip_dec_fold${F}.py \
                --ckpt $WD/epoch_100.pth --variant gated_dens --seed $S --fold $F \
                --val-scenes $VAL --eval-windows 7 --sin-clip \
                --cfg-options $OPT_DS model.gate_init=0.05 --out $CSV \
                > $WD.eval.log 2>&1 \
                || echo "!!! fallo evaluando fold $F semilla $S — ver $WD.eval.log"
            grep "^\[eval\]" $WD.eval.log
        }
    done
    echo "----- fold $F listo ($(date '+%d/%m %H:%M')) -----"
done

echo "=== DENSIDAD COMPLETA ==="
O="work_dirs/objcentrico/objcentrico_results.csv work_dirs/objcentrico8/objcentrico8_results.csv"
echo "  aporta la densidad?"
echo "    python agregar_resultados.py $CSV $O --comparar gated_dens:gated_obj --por-fold"
echo "  con densidad, la escena aporta?"
echo "    python agregar_resultados.py $CSV $O --comparar gated_dens:gate0_obj --por-fold"
