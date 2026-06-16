#!/bin/bash
# run_next_session.sh — Pipeline FASE 1 del protocolo de Claudine (waymo_10).
#
# 10 escenas limpias (8 train / 2 val). Re-pretrena el MAE en las 8 de train,
# extrae el encoder, entrena baseline + trajectory (gated + nogate) con ese
# encoder, y evalúa ADE/FDE a 3s. Siguiente fase: 100 escenas, luego 1000.
#
# Tiempo estimado: ~1.5 horas (MAE ~40min con 8 escenas + 3 trajectory + eval).
# Correr con la máquina PRENDIDA y SIN SUSPENDER.
#
# Lanzar:
#   cd /home/lcad/lidar_sweep_viewer/sapiens/pretrain
#   bash run_next_session.sh
#
# Log en /tmp/next_session.log. Cada paso verifica el anterior (set -e).
set -e

cd /home/lcad/lidar_sweep_viewer/sapiens/pretrain
CONDA="conda run -n sapiens_gpu"
LOG=/tmp/next_session.log
: > "$LOG"

echo "========== PASO 1/4: Re-pretraining MAE (8 escenas train, 1000 ep) ==========" | tee -a "$LOG"
date | tee -a "$LOG"
rm -rf work_dirs/mae_clean10
$CONDA python tools/train.py configs/sapiens_mae/lidar/mae_clean10_pretrain.py >> "$LOG" 2>&1
test -f work_dirs/mae_clean10/epoch_1000.pth || { echo "FALLO: no se generó el checkpoint MAE" | tee -a "$LOG"; exit 1; }

echo "========== PASO 2/4: Extraer encoder ==========" | tee -a "$LOG"
date | tee -a "$LOG"
$CONDA python extract_mae_encoder.py \
    work_dirs/mae_clean10/epoch_1000.pth \
    work_dirs/mae_encoder_clean10.pth >> "$LOG" 2>&1
test -f work_dirs/mae_encoder_clean10.pth || { echo "FALLO: no se extrajo el encoder" | tee -a "$LOG"; exit 1; }

echo "========== PASO 3/4: Entrenar baseline + trajectory (gated + nogate) ==========" | tee -a "$LOG"
date | tee -a "$LOG"
rm -rf work_dirs/clean10_baseline work_dirs/clean10_gated_newmae work_dirs/clean10_nogate_newmae
$CONDA python tools/train.py configs/sapiens_mae/lidar/clean10_baseline.py       >> "$LOG" 2>&1
$CONDA python tools/train.py configs/sapiens_mae/lidar/clean10_gated_newmae.py    >> "$LOG" 2>&1
$CONDA python tools/train.py configs/sapiens_mae/lidar/clean10_nogate_newmae.py   >> "$LOG" 2>&1

echo "========== PASO 4/4: Evaluación ADE/FDE (10 escenas, horizonte 3s) ==========" | tee -a "$LOG"
date | tee -a "$LOG"
$CONDA python evaluate_clean10_newmae.py 2>&1 | \
    grep -vE "DeprecationWarning|FutureWarning|torch.load|from torch.distributed|checkpoint = torch|descartados por salto" | tee -a "$LOG"

echo "========== FASE 1 (waymo_10) COMPLETA ==========" | tee -a "$LOG"
date | tee -a "$LOG"
echo "Si la escena (nogate) supera al baseline en val -> escalar a 100 escenas." | tee -a "$LOG"
