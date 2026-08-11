#!/bin/bash
# run_fold3_resume.sh — retoma SOLO el fold 3.
#
# Folds 0, 1 y 2 ya están completos (encoder epoch_1000 + decoder 8 semillas),
# por eso no se usa run_folds_123.sh: volvería a correr los decoders de 1 y 2
# (~2h) para reescribir resultados idénticos.
#
# El 08/08 el encoder del fold 3 se cortó en la época 403 (máquina suspendida,
# GPU en Xid 154 -> reboot). --resume arranca del último checkpoint del
# work_dir; el config ahora guarda cada 50 épocas en vez de cada 250.
cd /home/lcad/lidar_sweep_viewer/sapiens/pretrain
source /home/lcad/miniconda3/etc/profile.d/conda.sh
conda activate sapiens_gpu

F=3
CKPT=work_dirs/rv_rect_fold${F}/epoch_1000.pth

echo "############ FOLD $F — reanudo $(date '+%d/%m %H:%M') ############"
python -u tools/train.py configs/sapiens_mae/lidar/config_rangeview_rect_fold${F}.py --resume
if [ ! -f "$CKPT" ]; then
    echo "!!! FOLD $F FALLO: no se generó $CKPT"
    exit 1
fi
echo "[fold $F] encoder listo: $CKPT"

python -u horizon_sweep.py \
    --enc "$CKPT" \
    --folds $F --seeds 0 1 2 3 4 5 6 7 --horizons 3s \
    --archs wayformer baseline \
    --cache work_dirs/cache_fold${F}_domain \
    --out work_dirs/horizon_fold${F} \
    --epochs 100
echo "############ FOLD $F — fin $(date '+%d/%m %H:%M') ############"
