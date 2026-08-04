#!/bin/bash
# run_domain_encoder_experiment.sh — Experimento ②: ¿un encoder MAE adaptado
# al dominio (re-pre-entrenado en las escenas de train del fold 0) da mejores
# features para el decoder que el encoder genérico congelado?
#
# Paso 1: re-pre-entrena el encoder MAE SOLO en las 20 escenas de train del
#         fold 0 (sin fuga de las 5 retenidas).
# Paso 2: corre el decoder wayformer (3 semillas) con ese encoder nuevo,
#         evaluando en las 5 escenas retenidas del fold 0.
#
# Comparación (todo en fold 0):
#   - wayformer + encoder DOMINIO (nuevo, este experimento)  -> work_dirs/cv_domain
#   - wayformer + encoder GENÉRICO (ya medido)               -> work_dirs/cv (fold0: 2.65/3.02/2.18)
#   - baseline sin escena (encoder-independiente, ya medido) -> work_dirs/cv
set -e
cd /home/lcad/lidar_sweep_viewer/sapiens/pretrain
source /home/lcad/miniconda3/etc/profile.d/conda.sh
conda activate sapiens_gpu

echo "=== PASO 1: re-pre-entrenar encoder en fold-0 train (20 escenas) ==="
python tools/train.py configs/sapiens_mae/lidar/config_rangeview_rect_fold0.py

CKPT=$(cat work_dirs/rv_rect_fold0/last_checkpoint)
echo "=== encoder listo: $CKPT ==="

echo "=== PASO 2: decoder wayformer con encoder de dominio (fold 0, 3 semillas) ==="
python cross_validate_decoder.py \
    --enc "$CKPT" \
    --folds 0 --archs wayformer \
    --out work_dirs/cv_domain \
    --cache work_dirs/cache_fold0_domain \
    --epochs 100

echo "=== FIN. Resultados en work_dirs/cv_domain/cv_results.csv ==="
echo "Comparar contra work_dirs/cv/cv_results.csv (fold 0, generic encoder + baseline)"
