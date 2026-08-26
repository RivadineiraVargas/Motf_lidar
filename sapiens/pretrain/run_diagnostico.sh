#!/bin/bash
# run_diagnostico.sh — 15 min: ¿cual normalizacion del objetivo entrena sano?
# A = historico (actual, sin recorte)  -> sospechada de inestable
# B = escala fija 10 m                 -> alternativa
# Se miran dos cosas: si la perdida BAJA de forma sostenida, y la velocidad
# por paso (con el modo A daba 4.9 s/paso, 11x mas lento que con recorte).
cd /home/lcad/lidar_sweep_viewer/sapiens/pretrain
source /home/lcad/miniconda3/etc/profile.d/conda.sh; conda activate sapiens_gpu
D=configs/sapiens_mae/lidar
for MODO in A B; do
  [ $MODO = A ] && EXTRA="" || EXTRA="train_dataloader.dataset.norm_scale=10.0"
  echo "########## MODO $MODO ##########"
  timeout 900 python -u tools/train.py $D/noclip_dec_fold0.py \
     --work-dir work_dirs/diag_$MODO --cfg-options randomness.seed=0 \
     model.gate_init=0.0 model.freeze_gate=True train_cfg.max_epochs=20 $EXTRA \
     2>&1 | grep -oE "\[[0-9]+\]\[20/20\].*loss: [0-9.]+" | sed 's/lr:[^ ]* //' | awk 'NR%4==1'
done
echo "=== DIAGNOSTICO COMPLETO ==="
