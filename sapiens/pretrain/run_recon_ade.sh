#!/bin/bash
# run_recon_ade.sh — ¿la calidad de reconstruccion del MAE predice el ADE?
#
# EL ESLABON QUE NUNCA SE MIDIO. Los experimentos 17, 21, 23 y 26 miden pérdida
# de reconstruccion del encoder y sacan conclusiones sobre el pipeline. Pero
# NUNCA se midio que un encoder que reconstruye mejor produzca una trayectoria
# mejor. Todo el diagnostico de encoders descansa en un supuesto sin verificar.
#
# EL DISENYO. El exp. 26 dejo, por fold, dos encoders del MISMO pre-entrenamiento
# que difieren en reconstruccion. Medidos con mascaras FRESCAS (semillas 100-103,
# que no participaron de la seleccion; ver recon_dos_ckpts.py):
#
#     fold  epoca   recon vs ep1000
#       0     530      +1,4 %   (el "mejor" resulto PEOR: era ruido)
#       1     450      -8,5 %
#       2     960      -0,4 %
#       3      30     -16,1 %
#       4     100     -24,4 %
#
# Ese rango —de -24,4 % a +1,4 %— es el eje x. El eje y es el ADE. Si la
# reconstruccion predice el desempenyo rio abajo, el efecto en ADE deberia seguir
# ese orden. Si no lo sigue, medir reconstruccion no dice nada sobre trayectorias
# y hay que releer los exp. 17/21/23/26 con esa luz.
#
# use_gate=False, Y ES LO CENTRAL DEL DISENYO. Con el gate aprendible el modelo
# lo cierra a ~0,004: la escena no llega al decoder y cambiar de encoder no
# moveria NADA: mediriamos que el gate se cierra, que ya lo sabemos. Con
# use_gate=False la rama de escena queda SIEMPRE activa (ver
# trajectory_model_attn.py:50-53), asi que la calidad del encoder puede
# expresarse. Es la condicion mas favorable posible a que la reconstruccion
# importe: si no aparece aca, no aparece.
#
# Los DOS encoders salen del MISMO work_dir (work_dirs/f1cv_curva), no uno de ahi
# y otro de work_dirs/f1cv: la unica diferencia entre los brazos tiene que ser la
# EPOCA, no la corrida de pre-entrenamiento.
#
# n = 5 folds x 4 semillas x 2 encoders = 40 corridas. El test es entre FOLDS.
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
CUR=work_dirs/f1cv_curva
CSV=work_dirs/recon_ade/recon_ade_results.csv
mkdir -p work_dirs/recon_ade

for F in 0 1 2 3 4; do
    VAL=$(python3 -c "
import re
t = open('$D/f1cv_mae_fold${F}.py').read()
m = re.search(r'val RETENIDA del fold \d+: \[(.*?)\]', t)
print(' '.join(re.findall(r\"'([0-9a-f]{16})'\", m.group(1))))")
    [ -n "$VAL" ] || { echo "!!! fold $F sin escenas de validacion — salteo"; continue; }
    NV=$(echo $VAL | wc -w)

    MEJOR=$(ls $CUR/mae_fold$F/epoch_*.pth | grep -v epoch_1000 | head -1)
    [ -f "$MEJOR" ] || { echo "!!! fold $F sin checkpoint temprano — salteo"; continue; }
    echo "######## FOLD $F — $(basename $MEJOR) vs epoch_1000 — $(date '+%d/%m %H:%M') ########"

    # Los dos encoders, del MISMO pre-entrenamiento.
    E_MEJOR=$CUR/enc_fold${F}_mejor.pth
    E_ULT=$CUR/enc_fold${F}_1000.pth
    [ -f "$E_MEJOR" ] || python -u extract_mae_encoder.py $MEJOR $E_MEJOR > /dev/null 2>&1 \
        || { echo "!!! fallo extrayendo $E_MEJOR"; continue; }
    [ -f "$E_ULT" ] || python -u extract_mae_encoder.py $CUR/mae_fold$F/epoch_1000.pth $E_ULT > /dev/null 2>&1 \
        || { echo "!!! fallo extrayendo $E_ULT"; continue; }

    for PAR in "enc_mejor:$E_MEJOR" "enc_1000:$E_ULT"; do
        V=${PAR%%:*}; ENC=${PAR#*:}
        for S in 0 1 2 3; do
            WD=work_dirs/recon_ade/${V}_f${F}s${S}
            [ -f "$WD/epoch_100.pth" ] || \
                python -u tools/train.py $D/noclip_dec_fold${F}.py --work-dir $WD \
                    --cfg-options randomness.seed=$S load_from=$ENC model.use_gate=False \
                    > $WD.log 2>&1 \
                || { echo "!!! fallo entrenando $V fold $F semilla $S"; continue; }
            ya_evaluado $CSV $F $V $S $NV || {
                python -u eval_fase1_seeds.py --cfg $D/noclip_dec_fold${F}.py \
                    --ckpt $WD/epoch_100.pth --variant $V --seed $S --fold $F \
                    --val-scenes $VAL --eval-windows 7 --sin-clip \
                    --cfg-options model.use_gate=False --out $CSV > $WD.eval.log 2>&1 \
                    || echo "!!! fallo evaluando $V fold $F semilla $S — ver $WD.eval.log"
                grep "^\[eval\]" $WD.eval.log
            }
        done
        echo "----- fold $F  $V listo ($(date '+%d/%m %H:%M')) -----"
    done
done

echo "=== RECONSTRUCCION vs ADE COMPLETO ==="
echo "  python agregar_resultados.py $CSV --comparar enc_mejor:enc_1000 --por-fold"
echo "Y despues correlacionar el efecto por fold contra $CUR/recon_dos_ckpts.csv"
