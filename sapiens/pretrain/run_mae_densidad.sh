#!/bin/bash
# run_mae_densidad.sh — re-pre-entrena el MAE con densidad y vuelve a medir.
#
# LA HIPOTESIS QUE QUEDA VIVA. El exp. 29 dio que enriquecer la entrada de
# ocupacion binaria a densidad continua no cambia nada: -0,013 +- 0,100 (p=0,78,
# 3/5 folds) contra el binario, y +0,029 contra el control sin escena. Le dimos
# al modelo 572 valores distintos en vez de 2 y no se movio.
#
# Pero el encoder venia pre-entrenado MIL EPOCAS sobre entradas BINARIAS. Un
# valor de 0,44 es algo que nunca vio. Es plausible que la informacion este ahi
# y el encoder no sepa leerla — como dar un texto mejor escrito en un alfabeto
# que no se aprendio. Sin descartar esto, "la densidad no sirve" queda con una
# explicacion alternativa abierta.
#
# ETAPA 1 — re-pre-entrenar los 5 encoders MAE con densidad=True. 20 min por
# fold (medido en el exp. 26), ~1,7 h. Escribe en work_dirs/mae_dens/, NO toca
# work_dirs/f1cv/: los encoders de los exp. 19-28 siguen intactos.
#
# ETAPA 2 — re-correr el decoder con densidad + esos encoders. 5 folds x 8
# semillas = 40 corridas de ~12,4 min = ~8,3 h.
#
# LA COMPARACION. Contra gated_dens (exp. 29: densidad con encoder BINARIO), que
# ya tiene 8 semillas medidas. Aisla exactamente el efecto de que el encoder
# hable el mismo idioma que la entrada:
#     gated_maedens vs gated_dens  -> importaba el desajuste encoder/entrada?
#     gated_maedens vs gate0_obj   -> con todo alineado, la escena aporta?
#
# LA ESCALA ES LA MISMA EN LAS DOS ETAPAS: los dos datasets llaman a
# aplicar_densidad() de trajectory_dataset.py. Verificado que 1/20/202/1711
# puntos dan 0,100/0,441/0,769/1,000 en ambos. Si divergieran, el encoder
# aprenderia una distribucion y recibiria otra sin ningun error visible.
#
# EXPECTATIVA, DICHA DE ANTEMANO: el exp. 27 midio que la calidad del encoder no
# predice el ADE (r=+0,34). Si mejorar mucho el encoder no movia la prediccion,
# adaptarlo a los grises probablemente tampoco. Esto se corre para CERRAR la
# hipotesis, no porque se espere que funcione.
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
CSV=work_dirs/maedens/maedens_results.csv
mkdir -p work_dirs/mae_dens work_dirs/maedens
DS="train_dataloader.dataset.centrar_en_objeto=True train_dataloader.dataset.densidad=True"

echo "######## ETAPA 1 — pre-entrenar los encoders con densidad ########"
for F in 0 1 2 3 4; do
    WD=work_dirs/mae_dens/mae_fold$F
    ENC=work_dirs/mae_dens/enc_fold${F}.pth
    if [ ! -f "$ENC" ]; then
        [ -f "$WD/epoch_1000.pth" ] || \
            python -u tools/train.py $D/f1cv_mae_fold${F}.py --work-dir $WD \
                --cfg-options train_dataloader.dataset.densidad=True \
                              default_hooks.checkpoint.max_keep_ckpts=1 \
                > $WD.log 2>&1 \
            || { echo "!!! fallo pre-entrenando fold $F"; continue; }
        python -u extract_mae_encoder.py $WD/epoch_1000.pth $ENC > /dev/null 2>&1 \
            || { echo "!!! fallo extrayendo encoder del fold $F"; continue; }
    fi
    echo "----- encoder fold $F listo ($(date '+%d/%m %H:%M')) -----"
done

echo "######## ETAPA 2 — decoder con densidad + encoder de densidad ########"
for F in 0 1 2 3 4; do
    ENC=work_dirs/mae_dens/enc_fold${F}.pth
    [ -f "$ENC" ] || { echo "!!! fold $F sin encoder — salteo"; continue; }
    VAL=$(python3 -c "
import re
t = open('$D/f1cv_mae_fold${F}.py').read()
m = re.search(r'val RETENIDA del fold \d+: \[(.*?)\]', t)
print(' '.join(re.findall(r\"'([0-9a-f]{16})'\", m.group(1))))")
    [ -n "$VAL" ] || { echo "!!! fold $F sin escenas — salteo"; continue; }
    NV=$(echo $VAL | wc -w)
    echo "######## FOLD $F — inicio $(date '+%d/%m %H:%M') ########"
    for S in 0 1 2 3 4 5 6 7; do
        WD=work_dirs/maedens/gated_maedens_f${F}s${S}
        [ -f "$WD/epoch_100.pth" ] || \
            python -u tools/train.py $D/noclip_dec_fold${F}.py --work-dir $WD \
                --cfg-options randomness.seed=$S $DS load_from=$ENC \
                              model.gate_init=0.05 \
                              default_hooks.checkpoint.max_keep_ckpts=1 \
                > $WD.log 2>&1 \
            || { echo "!!! fallo entrenando fold $F semilla $S"; continue; }
        ya_evaluado $CSV $F gated_maedens $S $NV || {
            python -u eval_fase1_seeds.py --cfg $D/noclip_dec_fold${F}.py \
                --ckpt $WD/epoch_100.pth --variant gated_maedens --seed $S --fold $F \
                --val-scenes $VAL --eval-windows 7 --sin-clip \
                --cfg-options $DS model.gate_init=0.05 --out $CSV \
                > $WD.eval.log 2>&1 \
                || echo "!!! fallo evaluando fold $F semilla $S"
            grep "^\[eval\]" $WD.eval.log
        }
    done
    echo "----- fold $F listo ($(date '+%d/%m %H:%M')) -----"
done

echo "=== MAE CON DENSIDAD COMPLETO ==="
echo "  importaba el desajuste encoder/entrada?"
echo "    python agregar_resultados.py $CSV work_dirs/densidad/densidad_results.csv \\"
echo "        --comparar gated_maedens:gated_dens --por-fold"
echo "  con todo alineado, la escena aporta?"
echo "    python agregar_resultados.py $CSV work_dirs/objcentrico/objcentrico_results.csv \\"
echo "        work_dirs/objcentrico8/objcentrico8_results.csv \\"
echo "        --comparar gated_maedens:gate0_obj --por-fold"
