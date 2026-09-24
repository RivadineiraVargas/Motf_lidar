#!/usr/bin/env bash
# ver_escena.sh — abre el visor en UNA escena, opcionalmente con un auto en foco.
#
# POR QUE EXISTE: `./show_point_cloud --input waymo_clean_view` recorre las 25
# escenas muy rapido y solo 10 tienen prediccion, asi que es facil mirar una
# pantalla vacia y creer que algo se rompio. Esto arma un directorio con
# symlinks a una sola escena y deja el visor ahi.
#
# Uso:
#   ./ver_escena.sh --todas                 # LAS 10 escenas con prediccion, en bucle
#   ./ver_escena.sh --todas 0               # ...con el primer auto de cada una en foco
#   ./ver_escena.sh 4b60f9400a30ceaf 6      # escena + indice del auto a enfocar
#   ./ver_escena.sh 394e61f27c2a1700        # sin foco: todos los autos
#   ./ver_escena.sh --listar                # que escenas hay y cuantos objetos
#
# Dentro del visor: n/m cambia de auto, t alterna predicciones, b las cajas,
# espacio pausa, ESC sale.
set -euo pipefail
cd "$(dirname "$0")"
RAIZ=$PWD
TXT=$RAIZ/predictions_global.txt

if [ "${1:-}" = "--listar" ] || [ $# -eq 0 ]; then
    [ -f "$TXT" ] || { echo "!!! falta $TXT — corre antes: sapiens/pretrain/simular_fold.sh todos"; exit 1; }
    echo "escena            objetos   (el indice del foco es la posicion en esta lista, ordenada por id)"
    awk '{print $1"\t"$2}' "$TXT" | sort -u | awk '{c[$1]++} END{for (s in c) printf "  %s   %3d\n", s, c[s]}' | sort
    exit 0
fi

FOCO=""
if [ "$1" = "--todas" ]; then
    # Solo las escenas que TIENEN prediccion. Enlazar las 25 de waymo_clean_view
    # hace que el visor pase 15 escenas en blanco y parezca que algo se rompio.
    ESCENAS=$(awk '{print $1}' "$TXT" | sort -u)
    FOCO=${2:-}
else
    ESCENAS=$1; FOCO=${2:-}
    [ -d "waymo_clean_view/bin_files/$1" ] || { echo "!!! la escena $1 no esta en waymo_clean_view"; exit 1; }
    grep -q "^$1 " "$TXT" 2>/dev/null || echo "aviso: $1 no tiene predicciones en $TXT"
fi

DIR=$(mktemp -d)
trap 'rm -rf "$DIR"' EXIT
for d in bin_files images objs_bbox poses; do
    mkdir -p "$DIR/$d"
    for S in $ESCENAS; do
        [ -d "$RAIZ/waymo_clean_view/$d/$S" ] && ln -s "$RAIZ/waymo_clean_view/$d/$S" "$DIR/$d/$S"
    done
done
[ -f waymo_clean_view/beam_inclinations.npy ] && ln -s "$RAIZ/waymo_clean_view/beam_inclinations.npy" "$DIR/"

N=$(echo $ESCENAS | wc -w)
echo "$N escena(s)${FOCO:+  |  foco en el objeto indice $FOCO de cada una}"
echo "  el visor las recorre en bucle; espacio pausa, n/m cambia de auto, ESC sale"
DISPLAY=${DISPLAY:-:1} ./show_point_cloud --input "$DIR" ${FOCO:+--foco $FOCO} -v 300
