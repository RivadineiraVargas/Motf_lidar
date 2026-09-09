#!/usr/bin/env bash
# simular_fold.sh — de un experimento terminado al simulador, en un comando.
#
# Regenera predictions_global.txt (el archivo que lee el viewer C++) a partir de
# un fold ya entrenado. Todo lo que se puede deducir, se deduce:
#
#   - la SEMILLA: por default la MEJOR del fold segun ade_moving del CSV, asi no
#     hay que elegirla a ojo ni mirar los numeros antes de tiempo
#   - pad_dim y num_modes: los lee export_fase1_global.py de los tensores del
#     checkpoint. Pasarlos a mano es la forma mas facil de dibujar basura
#     convincente (mmengine carga con strict=False por default)
#   - las escenas: las de VALIDACION del fold, nunca vistas en entrenamiento
#
# Uso:
#   ./simular_fold.sh 0                        # mejor semilla del fold 0, exp padk6
#   ./simular_fold.sh 2 --semilla 5            # una semilla concreta
#   ./simular_fold.sh 0 --variante pad0_k6     # el brazo de control
#   ./simular_fold.sh 0 --exp initscale --variante base_pad64
#   ./simular_fold.sh 0 --un-modo              # solo el mas probable (k>1)
#
# Despues: ./show_point_cloud ...   y en el visor  n/m  para pasar de auto en auto.
set -euo pipefail
cd /home/lcad/lidar_sweep_viewer/sapiens/pretrain

FOLD=${1:-0}; shift || true
EXP=padk6; VARIANTE=pad64_k6; SEMILLA=mejor; MODOS=todos
CFG_BASE=configs/sapiens_mae/lidar/noclip_base_fold
TXT=/home/lcad/lidar_sweep_viewer/predictions_global.txt

while [ $# -gt 0 ]; do
    case $1 in
        --exp)      EXP=$2;      shift 2 ;;
        --variante) VARIANTE=$2; shift 2 ;;
        --semilla)  SEMILLA=$2;  shift 2 ;;
        --cfg-base) CFG_BASE=$2; shift 2 ;;
        --txt)      TXT=$2;      shift 2 ;;
        --un-modo)  MODOS=mas-probable; shift ;;
        *) echo "opcion desconocida: $1"; exit 1 ;;
    esac
done

CSV=work_dirs/$EXP/${EXP}_results.csv

if [ "$SEMILLA" = "mejor" ]; then
    [ -f "$CSV" ] || { echo "!!! no existe $CSV — pasa --semilla N a mano"; exit 1; }
    SEMILLA=$(python3 - "$CSV" "$FOLD" "$VARIANTE" <<'PY'
import csv, sys, collections
csv_path, fold, variante = sys.argv[1], int(sys.argv[2]), sys.argv[3]
# ade_moving ponderado por objeto sobre las escenas de validacion del fold
agg = collections.defaultdict(lambda: [0.0, 0])
for r in csv.DictReader(open(csv_path)):
    if int(r['fold']) != fold or r['variant'] != variante:
        continue
    n = int(r['n_moving'])
    agg[int(r['seed'])][0] += float(r['ade_moving']) * n
    agg[int(r['seed'])][1] += n
if not agg:
    sys.exit(f'!!! el CSV no tiene fold {fold} variante {variante}')
mejor = min(agg, key=lambda s: agg[s][0] / agg[s][1])
print(mejor)
PY
)
    echo "[semilla] mejor del fold $FOLD para $VARIANTE: $SEMILLA"
fi

CKPT=work_dirs/$EXP/${VARIANTE}_f${FOLD}s${SEMILLA}/epoch_100.pth
[ -f "$CKPT" ] || { echo "!!! no existe $CKPT (¿esa corrida termino?)"; exit 1; }
echo "[ckpt] $CKPT"

source /home/lcad/miniconda3/etc/profile.d/conda.sh
conda activate sapiens_gpu
python -u export_fase1_global.py \
    --cfg ${CFG_BASE}${FOLD}.py --ckpt "$CKPT" --fold "$FOLD" \
    --modos "$MODOS" --txt "$TXT" 2>&1 | grep -vE "ventanas descartadas"

echo
echo "Listo. En el visor:  n/m = cambiar de auto,  t = alternar predicciones."
echo "  rojo vivo = modo mas probable   azul tenue = modos alternativos"
