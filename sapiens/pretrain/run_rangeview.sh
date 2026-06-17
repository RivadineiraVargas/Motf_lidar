#!/bin/bash
# run_rangeview.sh — Pipeline del track RANGE-VIEW (alternativa a vóxels).
#
# 10 cenas (8 treino / 2 val). Pré-treina o MAE sobre a range-view, extrai o
# encoder, treina o trajectory gated (gate_init=0.5) e avalia vs baseline.
# Compara com o melhor do track de vóxels (Val ADE 1.303m, -35%).
#
# Tempo ~1.5h. Máquina ligada e sem suspender.
#   cd /home/lcad/lidar_sweep_viewer/sapiens/pretrain
#   bash run_rangeview.sh
set -e
cd /home/lcad/lidar_sweep_viewer/sapiens/pretrain
CONDA="conda run -n sapiens_gpu"
CFG=configs/sapiens_mae/lidar
LOG=/tmp/rangeview.log
: > "$LOG"

echo "===== PASO 1/4: Pré-treino MAE range-view (8 cenas, 1000 ep) =====" | tee -a "$LOG"; date | tee -a "$LOG"
rm -rf work_dirs/mae_rangeview
$CONDA python tools/train.py "$CFG/mae_rangeview_pretrain.py" >> "$LOG" 2>&1
test -f work_dirs/mae_rangeview/epoch_1000.pth || { echo "FALLO MAE" | tee -a "$LOG"; exit 1; }

echo "===== PASO 2/4: Extrair encoder =====" | tee -a "$LOG"; date | tee -a "$LOG"
$CONDA python extract_mae_encoder.py \
    work_dirs/mae_rangeview/epoch_1000.pth work_dirs/mae_encoder_rangeview.pth >> "$LOG" 2>&1
test -f work_dirs/mae_encoder_rangeview.pth || { echo "FALLO extract" | tee -a "$LOG"; exit 1; }

echo "===== PASO 3/4: Treinar trajectory range-view gated =====" | tee -a "$LOG"; date | tee -a "$LOG"
rm -rf work_dirs/clean10_rv_gated_init
$CONDA python tools/train.py "$CFG/clean10_rv_gated_init.py" >> "$LOG" 2>&1

echo "===== PASO 4/4: Avaliação (range-view vs baseline vs vóxel) =====" | tee -a "$LOG"; date | tee -a "$LOG"
$CONDA python evaluate_rangeview.py 2>&1 | \
  grep -vE "DeprecationWarning|FutureWarning|torch.load|from torch.distributed|checkpoint = torch|descartados" | tee -a "$LOG"

echo "===== TRACK RANGE-VIEW COMPLETO =====" | tee -a "$LOG"; date | tee -a "$LOG"
