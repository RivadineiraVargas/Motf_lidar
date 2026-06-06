#!/bin/bash
# Entrenamiento: trajectory_attn_frozen_encoder (500 épocas, ~12h)
# Ejecutar a las 22:00 del 2026-06-03

set -e

LOG_FILE="$HOME/lidar_sweep_viewer/training_frozen_$(date +%Y%m%d_%H%M%S).txt"
WORKDIR="/home/lcad/lidar_sweep_viewer/sapiens/pretrain"
CONFIG="configs/sapiens_mae/lidar/trajectory_attn_overfit.py"

echo "Iniciando entrenamiento: $(date)" | tee "$LOG_FILE"
echo "Config: $CONFIG" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "---" | tee -a "$LOG_FILE"

cd "$WORKDIR"

source /home/lcad/miniconda3/etc/profile.d/conda.sh
conda activate sapiens_final

python tools/train.py "$CONFIG" 2>&1 | tee -a "$LOG_FILE"

echo "---" | tee -a "$LOG_FILE"
echo "Entrenamiento finalizado: $(date)" | tee -a "$LOG_FILE"
