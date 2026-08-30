#!/usr/bin/env python3
"""Hook Stop — versiona los CSV de resultados que estén sin versionar o modificados.

POR QUÉ EXISTE. `work_dirs/` está en .gitignore, así que cada CSV de resultados
necesita `git add -f` a mano. El 30/08 se descubrió que `jm_results.csv` —la
evidencia cruda del experimento 18— nunca se había agregado: existía solo en el
disco de una máquina que ya se suspendió dos veces por errores Xid de NVIDIA.

Son 56 KB los 17 CSV juntos. No hay ninguna razón para no versionarlos, y la
regla no puede depender de que alguien se acuerde: por eso es un hook.

Solo hace `git add -f` (nunca commit). Los archivos quedan preparados y entran
en el próximo commit. Es idempotente: si no hay nada nuevo, no hace nada ni
imprime nada.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path('/home/lcad/lidar_sweep_viewer')
PATRON = 'sapiens/pretrain/work_dirs/**/*results*.csv'


def git(*args):
    return subprocess.run(['git', '-C', str(REPO), *args],
                          capture_output=True, text=True, timeout=30)


def main():
    if not (REPO / '.git').exists():
        return

    nuevos, modificados = [], []
    for f in sorted(REPO.glob(PATRON)):
        rel = f.relative_to(REPO).as_posix()
        seguido = git('ls-files', '--error-unmatch', rel).returncode == 0
        if not seguido:
            nuevos.append(rel)
        # `git diff HEAD` cubre tanto el árbol de trabajo como lo ya preparado.
        elif git('diff', '--quiet', 'HEAD', '--', rel).returncode != 0:
            modificados.append(rel)

    porAgregar = nuevos + modificados
    if not porAgregar:
        return

    r = git('add', '-f', *porAgregar)
    if r.returncode != 0:
        print(json.dumps({'systemMessage':
                          f'hook: no pude versionar los CSV — {r.stderr.strip()[:200]}'}))
        return

    partes = []
    if nuevos:
        partes.append(f'{len(nuevos)} sin versionar ('
                      + ', '.join(Path(p).name for p in nuevos[:3])
                      + (', …' if len(nuevos) > 3 else '') + ')')
    if modificados:
        partes.append(f'{len(modificados)} modificado(s)')
    print(json.dumps({'systemMessage':
                      'CSV de resultados preparados para el próximo commit: '
                      + ' y '.join(partes)}))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:      # un hook nunca debe tumbar la sesión
        print(json.dumps({'systemMessage': f'hook de versionado falló: {e}'}))
