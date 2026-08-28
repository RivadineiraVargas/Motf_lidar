#!/bin/bash
# run_jointmotion.sh — replica la diferencia metodologica central de JointMotion
# (Wagner et al. 2024, arXiv:2403.05489): NO congelar el encoder.
#
# Cita del paper (seccion Fine-tuning): "We initialize the modality-specific
# encoders with the learned weights from pre-training and DO NOT FREEZE any
# weights during fine-tuning."
# Nosotros veniamos congelandolo (302.6M params, 0 entrenables): el
# pre-entrenamiento era un extractor fijo de features, no una inicializacion.
# Ellos reportan -3% a -12% de FDE con auto-supervision; nosotros +5.9%.
#
# El encoder completo no entra en 8 GB con lote 16 (OOM), asi que se baja a
# lote 4. Eso cambia la dinamica de optimizacion, de modo que se corre TAMBIEN
# un 'gated' CONGELADO con lote 4 como control: sin el, no se podria atribuir la
# diferencia al descongelamiento en vez de al lote.
#
# gate0 y baseline no se repiten: gate0 tiene la escena en cero (el gradiente
# nunca llega al encoder, descongelarlo no cambia nada) y baseline ni lo usa.
# Se comparan contra los ya medidos en work_dirs/geo.
cd /home/lcad/lidar_sweep_viewer/sapiens/pretrain
source /home/lcad/miniconda3/etc/profile.d/conda.sh; conda activate sapiens_gpu
D=configs/sapiens_mae/lidar; VAL="7e2f727866c69ea0 82f90331a1dfe968"
mkdir -p work_dirs/jm
for S in 0 1 2 3 4 5 6 7; do
  for V in gated_frozen_b4 gated_finetune_b4; do
    [ "$V" = gated_frozen_b4 ] && FR=True || FR=False
    WD=work_dirs/jm/${V}_f0s${S}
    [ -f "$WD/epoch_100.pth" ] || python -u tools/train.py $D/geo_dec_fold0.py --work-dir $WD \
        --cfg-options randomness.seed=$S model.gate_init=0.5 model.freeze_encoder=$FR \
          train_dataloader.batch_size=4 > $WD.log 2>&1 || { echo "!!! fallo $V s$S"; continue; }
    python -u eval_fase1_seeds.py --cfg $D/geo_dec_fold0.py --ckpt $WD/epoch_100.pth \
        --variant $V --seed $S --fold 0 --val-scenes $VAL --eval-windows 7 --sin-clip \
        --out work_dirs/jm/jm_results.csv 2>&1 | grep "^\[eval\]"
  done
  echo "----- semilla $S lista ($(date '+%H:%M')) -----"
done
echo "=== JOINTMOTION COMPLETO ==="
