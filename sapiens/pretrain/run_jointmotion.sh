#!/bin/bash
# run_jointmotion.sh — ¿el problema era congelar el encoder?
#
# JointMotion (Wagner 2024, arXiv:2403.05489) dice en su seccion de fine-tuning:
# "We initialize the modality-specific encoders with the learned weights from
# pre-training and DO NOT FREEZE any weights during fine-tuning." Nosotros lo
# congelabamos (302.6M, 0 entrenables): el pre-entrenamiento era un extractor
# fijo, no una inicializacion. Ellos obtienen -3% a -12% de FDE; nosotros +5.9%.
#
# POR QUE PARCIAL Y NO TODO: descongelar los 302M da OOM en 8 GB con lote 16, y
# bajar el lote a 4 degrada el modelo POR SI SOLO (medido el 28/08: ADE 4.84 ->
# 8.29, y el gate colapsa a ~0). Descongelar solo la cola entra con lote 16 y
# deja comparable el resultado contra TODO lo medido esta semana.
#
# Tres variantes x 8 semillas, TODO lo demas identico (fold 0, encoder
# geometrico, lote 16, epoca fija 100, test de 319):
#   ft0 — congelado (replica del exp.17, control)
#   ft2 — ultimos 2 bloques descongelados (25.2M entrenables)
#   ft4 — ultimos 4 bloques descongelados (50.4M)
# ANTECEDENTE: en Fase 2 el fine-tuning parcial dio resultado MIXTO (mejoraba el
# ADE pero colapsaba la precision de validez). Mirar tambien la columna acc.
cd /home/lcad/lidar_sweep_viewer/sapiens/pretrain
source /home/lcad/miniconda3/etc/profile.d/conda.sh; conda activate sapiens_gpu

# Evalúa solo si esa combinación no está YA en el CSV. Antes la evaluación estaba
# fuera del guard de reanudación y `eval_fase1_seeds.py` appendea sin comprobar:
# relanzar una corrida cortada —lo que estos scripts dicen soportar— duplicaba
# filas, y una sola fila duplicada mueve la media ponderada ~19%. Condicionarlo a
# NUEVO=1 no alcanzaba: si el corte cae entre entrenar y evaluar, el checkpoint
# existe y la fila nunca se escribiría. La fuente de verdad es el CSV.
ya_evaluado() {   # $1=csv  $2=fold  $3=variante  $4=semilla  $5=nº de escenas esperadas
    # Exige que estén TODAS las filas, no una. `eval_fase1_seeds.py` escribe una
    # fila POR ESCENA: si una evaluación previa murió después de la primera, dar
    # la combinación por hecha dejaría el fold con medio resultado, en silencio.
    [ -f "$1" ] || return 1
    # Si el nº esperado de escenas es 0 —p.ej. porque VAL quedó vacío— NO se puede
    # concluir "ya evaluado": `[ n -ge 0 ]` es cierto siempre y saltearíamos la
    # evaluación de un fold entero en silencio, tras horas de entrenamiento.
    [ "${5:-1}" -gt 0 ] || return 1
    local n; n=$(grep -c "^$2,$3,$4," "$1")
    [ "$n" -ge "$5" ]
}
D=configs/sapiens_mae/lidar; VAL="7e2f727866c69ea0 82f90331a1dfe968"
mkdir -p work_dirs/jm
for S in 0 1 2 3 4 5 6 7; do
  for NB in 0 2 4; do
    V=ft$NB; WD=work_dirs/jm/${V}_f0s${S}
    [ -f "$WD/epoch_100.pth" ] || python -u tools/train.py $D/geo_dec_fold0.py --work-dir $WD \
        --cfg-options randomness.seed=$S model.gate_init=0.5 model.finetune_blocks=$NB \
        > $WD.log 2>&1 || { echo "!!! fallo $V s$S"; continue; }
    ya_evaluado work_dirs/jm/jm_results.csv 0 $V $S 2 || python -u eval_fase1_seeds.py --cfg $D/geo_dec_fold0.py --ckpt $WD/epoch_100.pth \
        --variant $V --seed $S --fold 0 --val-scenes $VAL --eval-windows 7 --sin-clip \
        --out work_dirs/jm/jm_results.csv 2>&1 | grep "^\[eval\]"
  done
  echo "----- semilla $S lista ($(date '+%H:%M')) -----"
done
echo "=== JOINTMOTION COMPLETO ==="
