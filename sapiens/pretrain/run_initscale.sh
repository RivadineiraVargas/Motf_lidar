#!/usr/bin/env bash
# run_initscale.sh — ¿el -0,217 de `gate0` sobre el baseline es la ARQUITECTURA
# o solo la ESCALA DE INICIALIZACION?
#
# EL HALLAZGO QUE LO ORIGINA (09/09). `gate0` le gana al baseline cinematico por
# -0,217 en 5/5 folds (t=-2,24) y ese es el mejor resultado del proyecto. Se leia
# como "el decoder con cross-attention aporta capacidad sobre el MLP". Pero al
# revisarlo:
#
#   - baseline_model.py y trajectory_model_attn.py tienen el MISMO MLP
#     (512-512-512) y la misma mode_head.
#   - Los configs noclip_base_fold*.py y noclip_dec_fold*.py son identicos en
#     lr=1e-3, weight_decay=1e-4, batch_size=16, max_epochs=100, history_len=5,
#     pred_len=30, voxel_res=2.0 y spatial_range. MISMO dataset.
#   - En `gate0` el gate esta congelado en 0: la columna `gate` vale 0.00000 en
#     las 40 corridas del CSV, o sea scene_feat es el vector EXACTAMENTE cero.
#
# Queda una sola diferencia: el ancho de la primera capa.
#
#     baseline:  Linear(15, 512)   cota 1/sqrt(15) = 0,2582   std(hist) = 0,14851
#     gate0:     Linear(79, 512)   cota 1/sqrt(79) = 0,1125   std(hist) = 0,06454
#
# Las MISMAS 15 entradas, con pesos iniciales 2,3x mas chicos. Si eso explica el
# -0,217, el mejor resultado del proyecto no mide arquitectura: mide una eleccion
# de scene_dim=64 que nadie tomo a proposito.
#
# EL DISEÑO. Se agrego `pad_dim` a BaselineTrajectoryModel: pega N columnas de
# CEROS a la entrada, sin informacion, solo para reproducir la geometria de la
# primera capa de `gate0`. Verificado antes de correr:
#   pad_dim=0  -> input_dim=15, std(hist)=0,14851  (identico al original)
#   pad_dim=64 -> input_dim=79, std(hist)=0,06497  (reproduce el 0,06454 de gate0)
#   perturbar las 64 columnas de pad en +100 cambia la salida en 0,00000000
#
# LO QUE SE PRE-REGISTRA, antes de ver un numero:
#
#   H1 (artefacto):  base_pad64 - base_pad0 ~= -0,217. El -0,217 es escala de
#                    inicializacion y NO hay evidencia de que la arquitectura
#                    aporte. Se retracta el "mejor resultado del proyecto".
#   H0 (real):       base_pad64 - base_pad0 ~= 0. El -0,217 viene de algo del
#                    modelo con atencion que no es el ancho de entrada, y hay que
#                    seguir buscandolo (cross_attn/scene_norm muertos, el
#                    optimizador sobre parametros sin gradiente, el stream de RNG).
#
#   Metrica: ADE de objetos moviles, pareado por (fold, semilla), n=5 FOLDS.
#   Un resultado intermedio (~-0,10) se reporta como parcialmente explicado; no
#   se elige post-hoc cual de las dos historias contar.
#
# CONTROL DE SANIDAD GRATIS: base_pad0 debe reproducir `baseline_k1` de
# work_dirs/multimodal (mismo config, mismas semillas, mismo eval). Si NO lo
# reproduce, el problema es otro y este experimento no significa nada — mirar eso
# ANTES que el resultado.
#
# POR QUE EN EL BASELINE Y NO EN gate0: una corrida de gate0 tarda 29 min porque
# _encode_scene corre igual aunque su salida se multiplique por cero; el baseline
# tarda ~1,4 min. El efecto que se busca vive en la primera capa, que es identica
# en los dos. 2 brazos x 5 folds x 8 semillas = 80 corridas de ~1,4 min = ~2 h.
#
# NO CORRER EN PARALELO con el exp. 31: la GPU esta al 100 % de utilizacion y los
# dos irian a media velocidad. Encolar.
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
CSV=work_dirs/initscale/initscale_results.csv
mkdir -p work_dirs/initscale

for F in 0 1 2 3 4; do
    VAL=$(python3 -c "
import re
t = open('$D/f1cv_mae_fold${F}.py').read()
m = re.search(r'val RETENIDA del fold \d+: \[(.*?)\]', t)
print(' '.join(re.findall(r\"'([0-9a-f]{16})'\", m.group(1))))")
    [ -n "$VAL" ] || { echo "!!! fold $F sin escenas de validacion — salteo"; continue; }
    NV=$(echo $VAL | wc -w)
    echo "######## FOLD $F — inicio $(date '+%d/%m %H:%M') ########"

    for PAR in "base_pad0:0" "base_pad64:64"; do
        V=${PAR%%:*}; P=${PAR#*:}
        OPT="model.pad_dim=$P"
        for S in 0 1 2 3 4 5 6 7; do
            WD=work_dirs/initscale/${V}_f${F}s${S}
            [ -f "$WD/epoch_100.pth" ] || \
                python -u tools/train.py $D/noclip_base_fold${F}.py --work-dir $WD \
                    --cfg-options randomness.seed=$S $OPT \
                    > $WD.log 2>&1 \
                || { echo "!!! fallo entrenando $V fold $F semilla $S"; continue; }
            ya_evaluado $CSV $F $V $S $NV || {
                # El flag va TAMBIEN en la evaluacion: sin esto se construye el
                # modelo con input_dim=15 y el checkpoint de 79 no carga.
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

echo "=== CONTROL DE ESCALA DE INICIALIZACION COMPLETO ==="
echo "1) Sanidad — base_pad0 debe reproducir baseline_k1 (mirar ESTO primero):"
echo "   python agregar_resultados.py $CSV work_dirs/multimodal/multimodal_results.csv \\"
echo "       --comparar base_pad0:baseline_k1 --por-fold"
echo "2) El resultado:"
echo "   python agregar_resultados.py $CSV --comparar base_pad64:base_pad0 --por-fold"
