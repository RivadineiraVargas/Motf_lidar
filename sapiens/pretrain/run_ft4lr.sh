#!/usr/bin/env bash
# run_ft4lr.sh — el hueco del exp. 18: descongelar CON la tasa de aprendizaje correcta.
#
# EL HALLAZGO QUE LO ORIGINA (10/09). El exp. 18 concluyo que descongelar el
# encoder no tiene efecto (ft0 5,22 / ft2 5,22 / ft4 5,17, semillas 4/8 en cada
# direccion) y dio por descartada la hipotesis del congelamiento. Al verificar CON
# QUE TASA se entrenaron esos bloques:
#
#   geo_dec_fold0.py:80-82   AdamW, lr=1e-3, SIN paramwise_cfg
#   finetune_blocks solo cambia requires_grad; no crea grupos de parametros
#
# Verificado construyendo el optimizador real:
#   finetune_blocks=0 -> 4,9 M entrenables, 1 grupo, lr=[0.001]
#   finetune_blocks=4 -> 55,3 M entrenables, 1 grupo, lr=[0.001]
#
# Es decir: los 50,4 M de pesos PRE-ENTRENADOS corrieron a 1e-3, cien veces el
# --enc-lr por defecto (1e-5). Eso no es fine-tuning: es destruir el
# pre-entrenamiento a la tasa del decoder, que arranca aleatorio. Que ft2 y ft4
# dieran identico a ft0 es consistente con eso — a 1e-3 durante 100 epocas, un
# encoder pre-entrenado y uno aleatorio convergen al mismo sitio.
#
# El mecanismo de LR separado existe (--enc-lr, train_decoder_mini.py:510) pero
# SOLO en el track decoder_mini, que esta congelado. La ruta de Fase 1
# (tools/train.py + configs) nunca lo tuvo.
#
# POR QUE NO EL DESCONGELAMIENTO TOTAL. Medido hoy en esta maquina:
#   freeze_encoder=False, lote 16, entrada real (16,300,5) -> OOM, pico 6,89 GB
#   de 7,62 GB usables en la RTX 4060 Laptop (8188 MiB, la unica disponible).
#   MAEViT4D hereda de VisionTransformer de mmpretrain, que NO acepta with_cp,
#   asi que tampoco hay gradient checkpointing sin escribirlo.
# Bajar el lote esta prohibido: confunde el resultado con el efecto del lote ya
# medido (ADE 4,84 -> 8,29). Queda pendiente para una GPU mayor.
#   finetune_blocks=4 con lr_mult -> pico 3,10 GB de 7,62. Entra holgado.
#
# PRE-REGISTRO, escrito antes de ver ningun numero:
#
#   PRINCIPAL   ft4lr - ft0 (ADE de moviles, ponderado por objetos).
#               ¿Descongelar BIEN ayuda? Es la pregunta de la tesis.
#     H_a  efecto negativo consistente (>=5/8 semillas) -> descongelar con LR
#          apropiado SI importa, y el exp. 18 se quedo corto por dos motivos
#          (memoria y tasa).
#     H_b  nulo, semillas repartidas -> se confirma y cierra la hipotesis del
#          congelamiento.
#     EXPECTATIVA PREVIA DECLARADA: H_b.
#
#   SECUNDARIA  ft4lr - ft4. ¿La tasa era el problema? Es la que rescata o
#               entierra el exp. 18.
#     Si |ft4lr - ft4| es grande -> el exp. 18 no midio lo que dice medir y su
#          conclusion se retracta por escrito.
#     Si ~ 0 -> la tasa daba igual y la conclusion del exp. 18 sobrevive.
#
#   TERCIARIA   el gate final. Si abre por encima de ~0,1 es senal de que la
#               escena empieza a aportar; si sigue en ~0,07 como ft0/ft2/ft4, no.
#
#   LO QUE NO SE VA A PODER REPORTAR: la precision de validez. Esa columna no
#   existe en los CSV de Fase 1 (11 y 20 columnas, ninguna de validez); es de
#   Fase 2 / decoder_mini. Se dice y no se inventa.
#
# LIMITE DE ALCANCE, dicho de antemano: esto es 1 FOLD x 8 semillas, no el
# protocolo de 5 folds. Se hace asi para ser comparable con el exp. 18, pero por
# la regla 2 del proyecto la conclusion vale para el fold 0 y no mas. El ruido
# entre folds es ~3x el de semillas.
#
# PROTOCOLO identico al exp. 18: fold 0, semillas 0-7, geo_dec_fold0.py, encoder
# geometrico work_dirs/geo/mae_encoder_fold0.pth, gate_init=0.5, epoca fija 100,
# lote 16, eval con --eval-windows 7 --sin-clip sobre 7e2f727866c69ea0 y
# 82f90331a1dfe968, promediado PONDERADO por objetos (200 y 119).
#
# CSV APARTE a proposito: jm_results.csv tiene 11 columnas y eval_fase1_seeds.py
# hoy escribe 20. Mezclarlos corromperia el CSV del exp. 18.
#
# 8 corridas de ~22,6 min + 1 de sanidad = ~3,4 h.
cd /home/lcad/lidar_sweep_viewer/sapiens/pretrain
source /home/lcad/miniconda3/etc/profile.d/conda.sh
conda activate sapiens_gpu

ya_evaluado() { [ -f "$1" ] || return 1; local n; n=$(grep -c "^$2,$3,$4," "$1"); [ "$n" -ge "$5" ]; }

D=configs/sapiens_mae/lidar
CSV=work_dirs/ft4lr/ft4lr_results.csv
VAL="7e2f727866c69ea0 82f90331a1dfe968"
PW=optim_wrapper.paramwise_cfg.custom_keys.encoder.lr_mult=0.01
mkdir -p work_dirs/ft4lr

correr() {   # $1=variante  $2=semilla  $3..=opciones extra
    local V=$1 S=$2; shift 2
    local WD=work_dirs/ft4lr/${V}_f0s${S}
    [ -f "$WD/epoch_100.pth" ] || \
        python -u tools/train.py $D/geo_dec_fold0.py --work-dir $WD \
            --cfg-options randomness.seed=$S model.gate_init=0.5 "$@" \
            > $WD.log 2>&1 || { echo "!!! fallo entrenando $V s$S"; return 1; }
    ya_evaluado $CSV 0 $V $S 2 || {
        python -u eval_fase1_seeds.py --cfg $D/geo_dec_fold0.py --ckpt $WD/epoch_100.pth \
            --variant $V --seed $S --fold 0 --val-scenes $VAL --eval-windows 7 --sin-clip \
            --cfg-options model.gate_init=0.5 "$@" --out $CSV > $WD.eval.log 2>&1 \
            || echo "!!! fallo evaluando $V s$S — ver $WD.eval.log"
        grep "^\[eval\]" $WD.eval.log
    }
}

echo "######## SANIDAD: ft0 reejecutado (1 semilla) debe reproducir el CSV del exp. 18 ########"
correr ft0chk 0 model.finetune_blocks=0

echo "######## ft4lr — 8 semillas ($(date '+%d/%m %H:%M')) ########"
for S in 0 1 2 3 4 5 6 7; do
    correr ft4lr $S model.finetune_blocks=4 $PW
    echo "----- semilla $S lista ($(date '+%d/%m %H:%M')) -----"
done

echo "=== CONTROLES OBLIGATORIOS ==="
echo "1) el encoder se carga de verdad (antecedente del bug del --resume):"
grep -l "Load checkpoint from" work_dirs/ft4lr/*.log | wc -l | xargs echo "   logs con 'Load checkpoint from':"
grep -h "Load checkpoint from" work_dirs/ft4lr/ft4lr_f0s0.log | head -1
echo "2) grupos de LR reales en el log de entrenamiento:"
grep -c "lr=1e-05" work_dirs/ft4lr/ft4lr_f0s0.log | xargs echo "   parametros con lr=1e-05:"
echo "3) sanidad ft0chk vs jm_results.csv:"
echo "   python3 -c \"import csv;[print(r) for r in csv.reader(open('$CSV')) if r[1]=='ft0chk']\""
echo "   comparar contra: grep '^0,ft0,0,' work_dirs/jm/jm_results.csv"
