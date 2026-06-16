#!/bin/bash
# multi_horizon.sh — Prioridad 2: curva del beneficio de la escena vs horizonte.
# Entrena baseline + gated_init (gate_init=0.5, encoder limpio) a pred_len
# 10/20/50 (1s/2s/5s). El 3s (pred_len=30) ya está hecho (clean10_baseline +
# clean10_gated_init). Luego eval_multi_horizon.py arma la tabla + gráfico.
set -e
cd /home/lcad/lidar_sweep_viewer/sapiens/pretrain
CONDA="conda run -n sapiens_gpu"
CFG=configs/sapiens_mae/lidar
LOG=/tmp/multi_horizon.log
: > "$LOG"

for P in 10 20 50; do
  echo "========== HORIZONTE pred_len=$P ($(echo "scale=1;$P/10"|bc)s) ==========" | tee -a "$LOG"
  date | tee -a "$LOG"

  # Generar configs por horizonte (cambiar pred_len + work_dir)
  sed -e "s/^pred_len *= *30/pred_len    = $P/" \
      -e "s|work_dirs/clean10_baseline|work_dirs/clean10_baseline_p$P|" \
      "$CFG/clean10_baseline.py"   > "$CFG/clean10_baseline_p$P.py"
  sed -e "s/^pred_len *= *30/pred_len      = $P/" \
      -e "s|work_dirs/clean10_gated_init|work_dirs/clean10_gated_init_p$P|" \
      "$CFG/clean10_gated_init.py" > "$CFG/clean10_gated_init_p$P.py"

  rm -rf work_dirs/clean10_baseline_p$P work_dirs/clean10_gated_init_p$P
  $CONDA python tools/train.py "$CFG/clean10_baseline_p$P.py"   >> "$LOG" 2>&1
  $CONDA python tools/train.py "$CFG/clean10_gated_init_p$P.py" >> "$LOG" 2>&1
done

echo "========== CURVA MULTI-HORIZONTE: ENTRENAMIENTOS LISTOS ==========" | tee -a "$LOG"
date | tee -a "$LOG"
$CONDA python eval_multi_horizon.py 2>&1 | \
  grep -vE "DeprecationWarning|FutureWarning|torch.load|from torch.distributed|checkpoint = torch|descartados" | tee -a "$LOG"
echo "========== FIN ==========" | tee -a "$LOG"
