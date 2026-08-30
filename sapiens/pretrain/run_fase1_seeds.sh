#!/bin/bash
# run_fase1_seeds.sh — verifica el punto de partida de FASE 1 (vóxeles, 10 escenas).
#
# Por qué: el "+25% a 3s / Val ADE 1.303" de Fase 1 es de UNA corrida, UNA semilla,
# sin validación cruzada y SIN control de arquitectura. Este proyecto ya vio cinco
# resultados de una semilla evaporarse al muestrear bien (ver §6 de
# CONTEXTO_AUDITORIA.md). Antes de invertir en mejorar ese número hay que saber si
# existe.
#
# Tres variantes x 8 semillas:
#   baseline  — MLP solo con histórico (BaselineTrajectoryModel)
#   gate0     — CONTROL DE ARQUITECTURA: mismo modelo con escena, gate CONGELADO
#               en 0. La escena aporta exactamente 0 pero conserva cross_attn,
#               scene_proj y el mismo decoder. Es el control que faltó en Fase 2
#               durante 14 experimentos (exp. 14 de EXPERIMENTOS_DECODER.md).
#   gated     — gate aprendible desde 0.5 (la configuración que dio el +25%)
#
# Así se separan las dos preguntas que Fase 2 tuvo confundidas:
#   gate0 - baseline = cuánto aporta la CAPACIDAD del modelo
#   gated - gate0    = cuánto aporta la ESCENA
#
# ~100 épocas por corrida; 24 corridas en total.
cd /home/lcad/lidar_sweep_viewer/sapiens/pretrain
source /home/lcad/miniconda3/etc/profile.d/conda.sh
conda activate sapiens_gpu

# Evalúa solo si esa combinación no está YA en el CSV. Antes la evaluación estaba
# fuera del guard de reanudación y `eval_fase1_seeds.py` appendea sin comprobar:
# relanzar una corrida cortada —lo que estos scripts dicen soportar— duplicaba
# filas, y una sola fila duplicada mueve la media ponderada ~19%. Condicionarlo a
# NUEVO=1 no alcanzaba: si el corte cae entre entrenar y evaluar, el checkpoint
# existe y la fila nunca se escribiría. La fuente de verdad es el CSV.
ya_evaluado() {   # $1=csv  $2=fold  $3=variante  $4=semilla  $5=nº de escenas esperadas
    [ -f "$1" ] || return 1
    # Si el nº esperado de escenas es 0 —p.ej. porque VAL quedó vacío— NO se puede
    # concluir "ya evaluado": `[ n -ge 0 ]` es cierto siempre y saltearíamos la
    # evaluación de un fold entero en silencio, tras horas de entrenamiento.
    [ "${5:-1}" -gt 0 ] || return 1
    # Cuenta filas, no busca una: eval_fase1_seeds.py escribe una fila POR ESCENA.
    local n; n=$(grep -c "^$2,$3,$4," "$1")
    [ "$n" -ge "$5" ]
}
# El CSV histórico de este script (fase1_results.csv) tiene el esquema VIEJO, de 9
# columnas sin `fold`. Pero `eval_fase1_seeds.py` hoy escribe SIEMPRE `fold` como
# primera columna (11 campos): appendear ahí mezclaría esquemas y `agregar_resultados`
# reventaría al hacer int(seed) sobre 'gated'. Así que las corridas nuevas van a un
# CSV propio con el esquema actual, y el viejo queda como registro histórico.

CSV=work_dirs/fase1_seeds/fase1_results_v2.csv
mkdir -p work_dirs/fase1_seeds

run () {
    VAR=$1; CFG=$2; SEED=$3; shift 3
    WD=work_dirs/fase1_seeds/${VAR}_s${SEED}
    if [ ! -f "$WD/epoch_100.pth" ]; then
        python -u tools/train.py $CFG --work-dir $WD \
            --cfg-options randomness.seed=$SEED "$@" > $WD.log 2>&1 \
            || { echo "!!! falló $VAR seed $SEED (ver $WD.log)"; return; }
    fi
    ya_evaluado "$CSV" 0 "$VAR" "$SEED" 2 || python -u eval_fase1_seeds.py --cfg $CFG --ckpt $WD/epoch_100.pth \
        --variant $VAR --seed $SEED --fold 0 --out $CSV 2>&1 | grep "^\[eval\]"
}

BASE=configs/sapiens_mae/lidar/clean10_baseline.py
GATED=configs/sapiens_mae/lidar/clean10_gated_init.py

for S in 0 1 2 3 4 5 6 7; do
    echo "===== semilla $S ($(date '+%H:%M:%S')) ====="
    run baseline $BASE  $S
    run gate0    $GATED $S model.gate_init=0.0 model.freeze_gate=True
    run gated    $GATED $S model.gate_init=0.5
done
echo "=== FASE 1: 3 variantes x 8 semillas COMPLETO ==="
