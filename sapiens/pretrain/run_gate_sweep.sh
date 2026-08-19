#!/bin/bash
# run_gate_sweep.sh — ¿cuánta escena conviene dejar entrar?
#
# Contexto: el resultado más reproducible del proyecto es que el gate aprendido
# converge a 0.0968 ± 0.0135 en los 5 folds, desde 40 inicializaciones en 0.5.
# Pero eso solo dice DÓNDE aterriza el modelo, no si ese punto es bueno.
#
# Este barrido congela el gate en valores fijos y mide el ADE resultante, o sea
# construye la curva ADE vs cantidad-de-escena. Convierte toda la línea de
# investigación en una figura interpretable:
#   - si el mínimo cae en 0.0  -> "la cantidad óptima de escena es CERO"
#   - si cae cerca de 0.10     -> la escena aporta poco pero real, y el modelo
#                                 lo calibra solo (corrobora los 5 folds)
#
# Control interno: gatefix0.0 debe reproducir el baseline (la rama de escena
# queda anulada). Si no lo hace, hay un bug y el resto del barrido no vale.
#
# Folds 0 y 3 = los dos extremos medidos (la escena ayudaba -20.4% / dañaba
# +40.0%). Si la curva tiene la misma forma en ambos, el óptimo es una
# propiedad del método; si no, es otra cosa que depende del split.
#
# Sale a los work_dirs que YA tienen los baselines de esos folds, para que
# horizon_sweep.py aparee contra ellos sin recalcularlos.
#
# 6 valores x 8 semillas x 2 folds = 96 corridas, ~5h. Features cacheadas.
cd /home/lcad/lidar_sweep_viewer/sapiens/pretrain
source /home/lcad/miniconda3/etc/profile.d/conda.sh
conda activate sapiens_gpu

# 0.99 y no 1.0: el constructor satura gate_init a ±0.99, así la etiqueta no miente
ARCHS="gatefix0.0 gatefix0.05 gatefix0.1 gatefix0.2 gatefix0.5 gatefix0.99"

run_fold () {
    F=$1; OUT=$2
    echo "######## FOLD $F — inicio $(date '+%d/%m %H:%M') ########"
    python -u horizon_sweep.py \
        --enc work_dirs/rv_rect_fold${F}/epoch_1000.pth \
        --folds $F --seeds 0 1 2 3 4 5 6 7 --horizons 3s \
        --archs $ARCHS \
        --cache work_dirs/cache_fold${F}_domain \
        --out $OUT --epochs 100 \
        || echo "!!! FOLD $F: falló — sigo"
    echo "######## FOLD $F — fin $(date '+%d/%m %H:%M') ########"
}

run_fold 0 work_dirs/horizon_domain
run_fold 3 work_dirs/horizon_fold3
echo "=== BARRIDO DE GATE COMPLETO ==="
