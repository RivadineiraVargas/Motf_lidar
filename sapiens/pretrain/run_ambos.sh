#!/bin/bash
# run_ambos.sh — relanza los experimentos 16 y 17 CON el encoder efectivamente
# cargado. El bug: tools/train.py:111 pone cfg.load_from=None cuando se pasa
# --resume, asi que ambos corrieron con encoder ALEATORIO y dieron resultados
# identicos. --resume ya se quito de las corridas de decoder.
cd /home/lcad/lidar_sweep_viewer/sapiens/pretrain
bash run_noclip.sh
bash run_geo.sh
echo "=== AMBOS EXPERIMENTOS COMPLETOS ==="
