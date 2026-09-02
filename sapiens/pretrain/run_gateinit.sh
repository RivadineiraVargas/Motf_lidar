#!/bin/bash
# run_gateinit.sh — ¿el daño de la escena es la escena, o es el arranque del gate?
#
# LA PREGUNTA. La CV de 5 folds cerró el 31/08 04:33 con un resultado que sí
# sobrevivió al promedio entre folds: la escena EMPEORA el ADE en +0.723
# (t=+3.05, p=0.038, 0/5 folds a favor). Es el primer efecto de este proyecto
# que no se evapora al pasar de un fold a cinco.
#
# PERO hay un confundido conocido. Midiendo las curvas de pérdida de
# entrenamiento del fold 0 se vio que `gated` pasa sus primeras ~10 épocas
# paralizado —pérdida clavada en 0.254 mientras `gate0` ya baja a 0.062— y que
# la pérdida que alcanza en la época 100, `gate0` ya la tenía en la 41. O sea
# que `gated` compite con el equivalente a 41 épocas de las 100, y ese handicap
# NO tiene nada que ver con si la escena aporta información.
#
# LA CAUSA. `gate_init=0.5` mete la rama de escena a media amplitud desde el
# primer paso, cuando todavía no aprendió nada: el decoder recibe 15 números
# útiles (la historia) más 64 de ruido, y gasta épocas aprendiendo a callarlos.
#
# POR QUÉ 0.05 Y NO 0. El comentario del modelo dice que gate_init=0 es un
# "candado de gradiente: nunca abre". ReZero (Bachlechner 2020, arXiv:2003.04887,
# en papers/) muestra que eso es falso en general: con el escalar en cero los
# pesos de la rama no reciben gradiente, pero el ESCALAR sí se mueve en el primer
# paso (su ecuación 8), y de ahí en más la rama aprende. Aun así, nuestra rama
# CONCATENA en vez de sumar —no es la identidad con gate=0, como en ReZero— así
# que el argumento no transfiere limpio. 0.05 es la salida conservadora, estilo
# LayerScale: bajo pero distinto de cero, esquiva la discusión entera.
#
# DISEÑO. Solo la variante nueva: `baseline` y `gate0` ya están medidos en
# work_dirs/noclipcv y no cambian. 5 folds x 8 semillas = 40 corridas, todo lo
# demás idéntico a la CV (mismo encoder por fold, época fija 100, evaluación
# --sin-clip --eval-windows 7). Los 5 encoders ya existen: no se reentrena nada.
#
# CÓMO SE LEE. Al terminar, comparar contra el MISMO control:
#   gated    - gate0  = +0.723  (medido, gate_init=0.5)
#   gated005 - gate0  = ?
# Si el efecto se derrumba, el +72% era el arranque del gate y no la escena.
# Si se mantiene, la escena perjudica de verdad y el handicap era secundario.
#
# COSTO: ~9.7 min por corrida x 40 = ~6.5 h.
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
CSV=work_dirs/gateinit/gateinit_results.csv
mkdir -p work_dirs/gateinit

for F in 0 1 2 3 4; do
    ENC=work_dirs/f1cv/mae_encoder_fold${F}.pth
    [ -f "$ENC" ] || { echo "!!! fold $F: falta $ENC — salteo"; continue; }

    VAL=$(python3 -c "
import re
t = open('$D/f1cv_mae_fold${F}.py').read()
m = re.search(r'val RETENIDA del fold \d+: \[(.*?)\]', t)
print(' '.join(re.findall(r\"'([0-9a-f]{16})'\", m.group(1))))")
    [ -n "$VAL" ] || { echo "!!! fold $F: sin escenas de validación — salteo"; continue; }
    echo "######## FOLD $F — inicio $(date '+%d/%m %H:%M')  (val: $VAL) ########"

    for S in 0 1 2 3 4 5 6 7; do
        WD=work_dirs/gateinit/gated005_f${F}s${S}
        [ -f "$WD/epoch_100.pth" ] || python -u tools/train.py $D/noclip_dec_fold${F}.py \
            --work-dir $WD --cfg-options randomness.seed=$S model.gate_init=0.05 \
            > $WD.log 2>&1 || { echo "!!! fallo fold $F semilla $S"; continue; }
        ya_evaluado $CSV $F gated005 $S $(echo $VAL | wc -w) || \
            python -u eval_fase1_seeds.py --cfg $D/noclip_dec_fold${F}.py \
                --ckpt $WD/epoch_100.pth --variant gated005 --seed $S --fold $F \
                --val-scenes $VAL --eval-windows 7 --sin-clip --out $CSV 2>&1 | grep "^\[eval\]"
    done
    echo "######## FOLD $F — fin $(date '+%d/%m %H:%M') ########"
done

echo "=== GATE_INIT COMPLETO ==="
echo "Para comparar contra el mismo control:"
echo "  python agregar_resultados.py work_dirs/noclipcv/noclipcv_results.csv $CSV \\"
echo "      --por-fold --comparar gated:gate0 gated005:gate0"
