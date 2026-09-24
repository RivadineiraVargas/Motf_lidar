#!/bin/bash
# preparar_auditoria.sh — genera una copia del proyecto SIN las conclusiones del
# autor, para una auditoría independiente.
#
# POR QUE NO ALCANZA CON BORRAR CLAUDE.md. Las conclusiones están en cuatro
# lugares, y el más contaminante no es el obvio:
#   1. CLAUDE.md                      — el estado y las 5 reglas
#   2. docs/EXPERIMENTOS_DECODER.md   — los 27 experimentos con sus números
#   3. docs/CODEBASE_MAP.md           — las 30 trampas
#   4. EL HISTORIAL DE GIT            — "exp27: la reconstruccion del MAE NO
#      predice el ADE (r=+0,34)". Ocho lineas de `git log` entregan casi todo, y
#      los cuerpos de los commits traen las tablas enteras.
#
# Por eso la copia se hace SIN .git y se inicializa un repositorio nuevo con un
# solo commit neutro.
#
# QUE SI VIAJA, Y POR QUE. El código completo, con sus comentarios. No se pueden
# quitar sin mutilar la auditoría: el auditor tiene que poder revisar el código
# que generó los CSV. Pero varios comentarios —sobre todo las cabeceras de los
# run_*.sh— contienen afirmaciones del autor con números. El README lo advierte y
# pide que se auditen como todo lo demás, en vez de tomarlos por ciertos.
#
# Uso:  ./preparar_auditoria.sh [destino]     (default: ../motf_auditoria)
set -e
ORIG=/home/lcad/lidar_sweep_viewer
DEST=${1:-/home/lcad/motf_auditoria}

[ -e "$DEST" ] && { echo "!!! $DEST ya existe — borrar o elegir otro destino"; exit 1; }
mkdir -p "$DEST/sapiens/pretrain"

# Codigo y datos de resultados; NADA de docs, CLAUDE.md ni .git
# --exclude='.gitignore': el del repo original ignora work_dirs/, y en esta copia
# los CSV de work_dirs son justamente el objeto de la auditoria.
rsync -a --exclude='.git' --exclude='.gitignore' --exclude='docs' --exclude='CLAUDE.md' \
      --exclude='*.pth' --exclude='*.pt' --exclude='*.pkl' \
      --exclude='__pycache__' --exclude='*.pyc' \
      --exclude='waymo_*' --exclude='lidar_hendrix*' --exclude='papers' \
      --exclude='*.pdf' --exclude='*.zip' --exclude='*.jpeg' \
      --exclude='explicacion' \
      "$ORIG/sapiens/pretrain/" "$DEST/sapiens/pretrain/"
rsync -a "$ORIG/utilities/" "$DEST/utilities/" 2>/dev/null || true

# Los CSV de resultados: son el objeto de la auditoria
find "$ORIG/sapiens/pretrain/work_dirs" -name "*results*.csv" -o -name "curva_*.csv" \
     -o -name "recon_*.csv" 2>/dev/null | while read f; do
    rel=${f#$ORIG/}
    mkdir -p "$DEST/$(dirname $rel)"
    cp "$f" "$DEST/$rel"
done

cat > "$DEST/README.md" <<'EOF'
# MOTF — copia para auditoría independiente

Predicción de trayectorias de agentes de tráfico a partir de nubes de puntos LiDAR
de Waymo. Un encoder ViT pre-entrenado con MAE auto-supervisado produce rasgos de
la escena; un decoder los combina con la historia de posiciones del objeto y
predice su trayectoria futura.

**Esta copia se generó a propósito SIN la documentación del autor y SIN historial
de git**, para que la auditoría no herede sus conclusiones.

## Qué hay

| ruta | qué es |
|---|---|
| `sapiens/pretrain/work_dirs/*/*results*.csv` | los resultados crudos |
| `sapiens/pretrain/agregar_resultados.py` | el agregador; su cabecera fija la convención de promediado |
| `sapiens/pretrain/eval_fase1_seeds.py` | el evaluador que escribe los CSV |
| `sapiens/pretrain/mmpretrain/models/trajectory_pred/` | los modelos |
| `sapiens/pretrain/mmpretrain/datasets/` | la carga de datos y la voxelización |
| `sapiens/pretrain/run_*.sh` | los scripts que corrieron cada experimento |
| `sapiens/pretrain/configs/sapiens_mae/lidar/` | los configs |

## Cómo leer los CSV

Una fila por **(fold, variante, semilla, escena)**. Columnas principales:

- `ade_all` / `fde_all` — error medio y final sobre todos los objetos
- `ade_moving` / `fde_moving` — solo objetos en movimiento
- `minade_all` / `minfde_all` — el mejor de K hipótesis (solo en modelos multimodales)
- `n_obj` / `n_moving` — cuántos objetos promedia esa fila
- `gate` — un escalar aprendido que pondera cuánto entra la escena

Validación cruzada dejando 2 escenas afuera por fold, 5 folds, varias semillas.

## Advertencia sobre el código

El código conserva sus comentarios, porque sin ellos no se puede auditar. Varias
cabeceras de `run_*.sh` contienen afirmaciones del autor **con números**. No las
tomes como establecidas: son parte de lo que hay que auditar. Si un comentario dice
que algo se midió, verificá que el CSV lo sostenga.

## Lo que NO está

Los datos crudos de Waymo (8,3 GB) y los checkpoints entrenados. Se puede auditar
la **interpretación** de los resultados y el **código** que los generó, pero no
re-correr los experimentos.
EOF

cd "$DEST"
printf '__pycache__/\n*.pyc\n*.pth\n*.pt\n' > .gitignore
git init -q
git add -A
git -c user.name=auditoria -c user.email=auditoria@local \
    commit -q -m "MOTF: codigo y resultados para auditoria independiente"
echo "listo: $DEST"
echo "  archivos: $(git ls-files | wc -l)"
echo "  CSV: $(git ls-files '*results*.csv' '*curva*.csv' '*recon*.csv' | wc -l)"
echo "  historial: $(git log --oneline | wc -l) commit"
echo "  docs/ presente: $([ -d docs ] && echo 'SI — REVISAR' || echo 'no')"
echo "  CLAUDE.md presente: $([ -f CLAUDE.md ] && echo 'SI — REVISAR' || echo 'no')"
