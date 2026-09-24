#!/usr/bin/env bash
# run_padk6.sh — el mejor resultado del proyecto, ahora con k=6.
#
# QUE ES "EL MEJOR RESULTADO". Tras el exp. 32, la configuracion con menor ADE de
# moviles es `base_pad64`: el MLP cinematico (512-512-512) con 64 columnas de CEROS
# pegadas a la entrada. ADE 2,776 contra 3,036 del baseline pelado, -0,260 en 5/5
# folds. No gana por informacion ni por capacidad — gana porque nn.Linear inicializa
# con cota 1/sqrt(in_features) y 79 entradas dan pesos 2,3x mas chicos que 15.
# Es un artefacto, pero es el mejor numero que tenemos y la pregunta de si mejora
# con multimodalidad es legitima.
#
# LO QUE YA SABEMOS DE k=6, y por que acota lo que este experimento puede mostrar.
# Medido dos veces (exps. 24 y 25), en 5 folds y con dos cls_weight distintos:
#
#     ADE real   +0,303  p=0,036  0/5 folds   <- k=6 EMPEORA la prediccion
#     minADE_6   -0,728  p=0,005  5/5 folds   <- el oraculo "mejora" 29 %
#
# minADE_6 elige la mejor de 6 hipotesis DESPUES de ver la respuesta. Va a mejorar
# aca tambien, con o sin pad. **Si el titular de este experimento termina siendo
# "minADE_6 mejoro 29 %", no aprendimos nada** — ya lo sabiamos.
#
# LO QUE SE PRE-REGISTRA, antes de ver un numero:
#
#   PRINCIPAL   pad64_k6 - baseline_k6, ADE de moviles, pareado por (fold,semilla),
#               n=5 FOLDS. Ambos brazos a k=6, asi que el oraculo no interviene.
#     H_a  ~= -0,260  -> la ventaja de la escala de inicializacion SOBREVIVE a k=6
#     H_b  ~=  0      -> la ventaja era especifica de k=1
#
#   SECUNDARIA  pad64_k6 - base_pad64 (k=1): ¿k=6 sigue empeorando el ADE en la
#               mejor configuracion? Prediccion desde el baseline: ~+0,30, 0/5.
#
#   minADE_6 se REPORTA pero NO es titular. Cualquier mejora ahi es el oraculo.
#
# CONTROL DE SANIDAD. pad0_k6 debe reproducir `baseline_k6` (mismo config, mismas
# semillas, mismo eval, cls_weight=1.0 por default en los dos). A k=1 el equivalente
# dio max|dif| = 0,0000 en 40 semillas; si a k=6 NO reproduce, el problema es la
# interaccion pad_dim/num_modes y el resultado no significa nada. Mirar esto PRIMERO.
#
# cls_weight queda en el default 1.0 para parear exacto con baseline_k6. El exp. 25
# ya midio que ningun cls_weight probado mejora la prediccion real.
#
# 2 brazos x 5 folds x 8 semillas = 80 corridas de ~1,4 min = ~2 h.
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
CSV=work_dirs/padk6/padk6_results.csv
mkdir -p work_dirs/padk6

for F in 0 1 2 3 4; do
    VAL=$(python3 -c "
import re
t = open('$D/f1cv_mae_fold${F}.py').read()
m = re.search(r'val RETENIDA del fold \d+: \[(.*?)\]', t)
print(' '.join(re.findall(r\"'([0-9a-f]{16})'\", m.group(1))))")
    [ -n "$VAL" ] || { echo "!!! fold $F sin escenas de validacion — salteo"; continue; }
    NV=$(echo $VAL | wc -w)
    echo "######## FOLD $F — inicio $(date '+%d/%m %H:%M') ########"

    for PAR in "pad0_k6:0" "pad64_k6:64"; do
        V=${PAR%%:*}; P=${PAR#*:}
        OPT="model.pad_dim=$P model.num_modes=6"
        for S in 0 1 2 3 4 5 6 7; do
            WD=work_dirs/padk6/${V}_f${F}s${S}
            [ -f "$WD/epoch_100.pth" ] || \
                python -u tools/train.py $D/noclip_base_fold${F}.py --work-dir $WD \
                    --cfg-options randomness.seed=$S $OPT \
                    > $WD.log 2>&1 \
                || { echo "!!! fallo entrenando $V fold $F semilla $S"; continue; }
            ya_evaluado $CSV $F $V $S $NV || {
                # Los DOS flags van tambien en la evaluacion: sin num_modes=6 el
                # modelo se construye con 1 modo y el checkpoint no carga; sin
                # pad_dim la primera capa queda de 15 y tampoco.
                python -u eval_fase1_seeds.py --cfg $D/noclip_base_fold${F}.py \
                    --ckpt $WD/epoch_100.pth --variant $V --seed $S --fold $F \
                    --val-scenes $VAL --eval-windows 7 --sin-clip \
                    --cfg-options $OPT --out $CSV > $WD.eval.log 2>&1 \
                    || echo "!!! fallo evaluando $V fold $F semilla $S — ver $WD.eval.log"
                grep "^\[eval\]" $WD.eval.log
            }
        done
        echo "----- fold $F  $V listo ($(date '+%d/%m %H:%M')) -----"
    done
done

echo "=== EL MEJOR RESULTADO CON k=6, COMPLETO ==="
echo "1) Sanidad — pad0_k6 debe reproducir baseline_k6 (mirar ESTO primero):"
echo "   python agregar_resultados.py $CSV work_dirs/multimodal/multimodal_results.csv \\"
echo "       --comparar pad0_k6:baseline_k6 --por-fold"
echo "2) PRINCIPAL — ¿la ventaja de la inicializacion sobrevive a k=6?"
echo "   python agregar_resultados.py $CSV --comparar pad64_k6:pad0_k6 --por-fold"
echo "3) SECUNDARIA — ¿k=6 sigue empeorando el ADE en la mejor config?"
echo "   python agregar_resultados.py $CSV work_dirs/initscale/initscale_results.csv \\"
echo "       --comparar pad64_k6:base_pad64 --por-fold"
