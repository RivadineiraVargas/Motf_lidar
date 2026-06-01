#!/bin/bash
# Script de entrenamiento nocturno — MAE LiDAR 4D pre-entrenamiento completo

TARGET_HOUR=22
TARGET_MIN=0
LOG_FILE="/home/lcad/lidar_sweep_viewer/training_log_$(date +%Y%m%d_%H%M%S).txt"
TRAIN_DIR="/home/lcad/lidar_sweep_viewer/sapiens/pretrain"
CONFIG="configs/sapiens_mae/lidar/mae_lidar_10_overfit.py"

echo "[$(date)] Script iniciado. Esperando hasta las ${TARGET_HOUR}:${TARGET_MIN}..." | tee "$LOG_FILE"

# Calcular segundos hasta las 22:00
NOW=$(date +%s)
TARGET=$(date -d "today ${TARGET_HOUR}:${TARGET_MIN}:00" +%s)

# Si ya pasaron las 22:00, esperar hasta mañana
if [ "$TARGET" -le "$NOW" ]; then
    TARGET=$(date -d "tomorrow ${TARGET_HOUR}:${TARGET_MIN}:00" +%s)
    echo "[$(date)] Ya pasaron las ${TARGET_HOUR}:00 — agendado para mañana." | tee -a "$LOG_FILE"
fi

WAIT=$((TARGET - NOW))
echo "[$(date)] Esperando ${WAIT} segundos (hasta las $(date -d "@$TARGET"))..." | tee -a "$LOG_FILE"
sleep "$WAIT"

echo "[$(date)] Iniciando entrenamiento MAE LiDAR (4000 épocas)..." | tee -a "$LOG_FILE"
cd "$TRAIN_DIR" || { echo "ERROR: no se encontró $TRAIN_DIR"; exit 1; }

conda run -n sapiens_final python tools/train.py "$CONFIG" 2>&1 | tee -a "$LOG_FILE"

echo "[$(date)] Entrenamiento finalizado." | tee -a "$LOG_FILE"
