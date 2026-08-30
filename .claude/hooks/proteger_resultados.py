#!/usr/bin/env python3
"""Hook PreToolUse — bloquea Write/Edit sobre resultados y checkpoints.

POR QUÉ EXISTE. Cada `*_results.csv` de `work_dirs/` es la evidencia cruda de un
experimento y cada `.pth` cuesta entre 19 minutos y 20 horas de GPU. Ninguno de
los dos se produce con las herramientas Write o Edit: los escriben los scripts de
entrenamiento y evaluación desde Python. Así que un Write o un Edit apuntando ahí
solo puede ser un error, y conviene que falle antes y no después.

No estorba el pipeline: los scripts escriben por Bash, que este hook no toca.
"""
import json
import sys

EXT_PROTEGIDAS = ('.csv', '.pth', '.pt')


def main():
    try:
        datos = json.load(sys.stdin)
    except Exception:
        return                      # sin payload legible, no opinamos

    ruta = (datos.get('tool_input') or {}).get('file_path') or ''
    if not ruta:
        return

    norm = ruta.replace('\\', '/')
    if 'work_dirs/' not in norm or not norm.endswith(EXT_PROTEGIDAS):
        return

    que = 'un checkpoint' if norm.endswith(('.pth', '.pt')) else 'un CSV de resultados'
    print(json.dumps({
        'hookSpecificOutput': {
            'hookEventName': 'PreToolUse',
            'permissionDecision': 'deny',
            'permissionDecisionReason': (
                f'{ruta} es {que} dentro de work_dirs/. Esos archivos los generan '
                'los scripts de entrenamiento y evaluación, no se editan a mano: '
                'un CSV vale la evidencia de un experimento y un .pth vale entre 19 '
                'minutos y 20 horas de GPU. Si de verdad hay que corregir un '
                'resultado, hacelo con un script que deje registro, o pedíselo al '
                'usuario explícitamente.'),
        }
    }))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        pass                        # ante la duda, no bloquear
