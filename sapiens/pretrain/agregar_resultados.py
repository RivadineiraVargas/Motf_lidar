#!/usr/bin/env python3
"""agregar_resultados.py — el paso que faltaba entre los CSV y las tablas.

`eval_fase1_seeds.py` escribe una fila por (fold, variante, semilla, escena) y se
detiene ahí a propósito: promediar todas las muestras juntas como si fueran
independientes fabrica significancia falsa. Correcto. El problema es que hasta ahora
NADA agregaba esas filas después — todos los números publicados del proyecto se
calcularon a mano, sin que quedara registrado el método. Dos agregaciones ad hoc
distintas dieron 4,836 y 5,217 para la MISMA corrida (7% de diferencia).

Este script fija la convención y la imprime ANTES de cada tabla.

LAS TRES DECISIONES DE AGREGACIÓN, explícitas:

  1. DENTRO de una escena — ya la tomó el evaluador: promedia por objeto, con peso
     igual entre objetos de la misma escena.

  2. ENTRE escenas de validación — es la que estaba sin declarar. Las dos escenas
     del fold 0 tienen 200 y 119 objetos:
       --peso objetos (default) : pondera por n_obj. Cada OBJETO pesa igual.
       --peso escena            : media simple. Cada ESCENA pesa igual.
     La ponderada es la defendible: la media simple le da el mismo peso a la escena
     fácil que a la difícil. Los números de la reunión del 26/08 usan `escena`, así
     que para reproducirlos hay que pedirlo explícitamente.

  3. ENTRE folds — nunca promediar sin mirar. La varianza ENTRE folds es ~3x la de
     semillas (sd 0,29 vs 0,098): con 25 escenas domina QUÉ escenas caen de cada
     lado del corte. Si el CSV trae más de un fold, este script muestra el efecto
     POR FOLD y recién después el resumen entre folds, con n=folds, no n=corridas.

Las comparaciones son PAREADAS por (fold, semilla): la misma semilla en el mismo
fold es la misma inicialización, así que la diferencia cancela el ruido de GPU.

USO
    python agregar_resultados.py work_dirs/jm/jm_results.csv
    python agregar_resultados.py work_dirs/geo/geo_results.csv --peso escena
    python agregar_resultados.py work_dirs/f1cv/f1cv_results.csv --por-fold
    python agregar_resultados.py work_dirs/jm/jm_results.csv --comparar ft2:ft0 ft4:ft0
    python agregar_resultados.py work_dirs/*/­*_results.csv --poblacion moviles --metrica fde
"""
import argparse
import csv
import glob
import math
import statistics as st
from collections import defaultdict


# ---------------------------------------------------------------- estadística
def _betacf(a, b, x, itmax=200, eps=3e-16):
    """Fracción continua de la beta incompleta (Numerical Recipes, 6.4)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    """Beta incompleta regularizada I_x(a,b). Sin dependencias externas."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1.0 - x) / b


def p_dos_colas(t, gl):
    """p de dos colas para un t de Student con `gl` grados de libertad."""
    if gl <= 0:
        return float('nan')
    return _betai(0.5 * gl, 0.5, gl / (gl + t * t))


def t_pareado(a, b):
    """t de Student pareado sobre a−b. Devuelve (media, sd, t, gl, p, n_favor, n)."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n < 2:
        return (d[0] if n else float('nan'), float('nan'), float('nan'), 0,
                float('nan'), sum(1 for x in d if x < 0), n)
    m, sd = st.mean(d), st.stdev(d)
    if sd == 0:
        t = float('inf') if m else 0.0
    else:
        t = m / (sd / math.sqrt(n))
    gl = n - 1
    p = p_dos_colas(t, gl) if math.isfinite(t) else 0.0
    return m, sd, t, gl, p, sum(1 for x in d if x < 0), n


# ------------------------------------------------------------------- lectura
def leer(paths, metrica, poblacion):
    """-> ({(fold, variante, semilla): [(valor, n, escena)]}, gates, escenas, avisos).

    Tres defensas que la auditoría del 30/08 encontró que faltaban:

    (a) DUPLICADOS. `eval_fase1_seeds.py` abre el CSV en modo 'a' y no comprueba si
        ya existe la fila; los scripts `run_*.sh` llaman a la evaluación FUERA del
        guard `[ -f epoch_100.pth ]`. O sea que relanzar una corrida cortada —el
        escenario que los scripts dicen soportar— reevalúa lo ya hecho y duplica
        filas. Una sola fila duplicada mueve la media ponderada ~19%. Acá se
        deduplica por (fold, variante, semilla, escena) quedándose con la ÚLTIMA
        aparición, y se avisa fuerte.

    (b) COBERTURA DESPAREJA. Si a una corrida le falta una escena (crash a mitad de
        evaluación), promediarla contra corridas completas compara peras con
        manzanas. Se verifica que toda corrida de un fold cubra las mismas escenas.

    (c) CELDAS VACÍAS. El evaluador escribe '' en `ade_moving` cuando la escena no
        tiene ningún objeto que se desplace más de 1 m — y documenta que el 60-75%
        no se desplaza. Con --poblacion moviles eso reventaba con un ValueError
        opaco. Ahora se saltea la escena avisando.
    """
    col = f'{metrica}_{"moving" if poblacion == "moviles" else "all"}'
    col_n = 'n_moving' if poblacion == 'moviles' else 'n_obj'
    ultima = {}          # (fold, variante, semilla, escena) -> (valor, n, gate)
    orden = []
    dups = 0
    vacias = []
    for path in paths:
        with open(path) as fh:
            rdr = csv.DictReader(fh)
            cols = rdr.fieldnames or []
            if col not in cols:
                raise SystemExit(
                    f'{path}: no tiene la columna {col}. ¿Es un CSV de '
                    f'eval_fase1_seeds.py? Columnas: {", ".join(cols)}')
            # Los CSV viejos (fase1_results.csv, exp. 15) no traen `fold`: es el 0.
            hay_fold = 'fold' in cols
            for r in rdr:
                clave = (int(r['fold']) if hay_fold else 0,
                         r['variant'], int(r['seed']), r['scene'])
                if r[col] == '' or r[col_n] in ('', '0'):
                    vacias.append(clave)
                    continue
                if clave in ultima:
                    dups += 1
                else:
                    orden.append(clave)
                try:
                    g = float(r['gate'])
                except (ValueError, KeyError):
                    g = float('nan')
                ultima[clave] = (float(r[col]), int(r[col_n]), g)

    avisos = []
    if dups:
        avisos.append(
            f'!! {dups} fila(s) DUPLICADA(S) por (fold, variante, semilla, escena). '
            'Se usó la última de cada una. Causa casi segura: se relanzó una corrida '
            'cortada y la evaluación volvió a correr sobre checkpoints ya existentes.')
    if vacias:
        ej = ', '.join(f'{f}/{v}/s{s}/{e[:8]}' for f, v, s, e in vacias[:3])
        avisos.append(
            f'!! {len(vacias)} escena(s) sin dato en `{col}` — se excluyen. Ej: {ej}'
            + ('  (con --poblacion moviles esto pasa cuando ninguna trayectoria de la '
               'escena se desplaza más de 1 m)' if poblacion == 'moviles' else ''))

    crudo = defaultdict(list)
    gates = {}
    escenas = set()
    for clave in orden:
        if clave not in ultima:
            continue
        f, v, s, e = clave
        val, n, g = ultima[clave]
        crudo[(f, v, s)].append((val, n, e))
        gates[(f, v, s)] = g
        escenas.add(e)

    # (b) cobertura pareja: dentro de cada fold, toda corrida debe cubrir lo mismo.
    por_fold = defaultdict(set)
    for (f, _, _), filas in crudo.items():
        por_fold[f] |= {e for _, _, e in filas}
    for (f, v, s), filas in sorted(crudo.items()):
        faltan = por_fold[f] - {e for _, _, e in filas}
        if faltan:
            avisos.append(
                f'!! fold {f} / {v} / semilla {s} cubre '
                f'{len(filas)} de {len(por_fold[f])} escenas — le falta(n) '
                f'{", ".join(sorted(x[:8] for x in faltan))}. NO es comparable con las '
                'corridas completas; volver a evaluar esa corrida antes de publicar.')

    return crudo, gates, escenas, avisos


def agregar(crudo, peso):
    """Colapsa las escenas de cada corrida a un escalar, con la convención pedida."""
    out = {}
    for clave, filas in crudo.items():
        if peso == 'objetos':
            total = sum(n for _, n, _ in filas)
            out[clave] = (sum(v * n for v, n, _ in filas) / total) if total else float('nan')
        else:
            out[clave] = st.mean([v for v, _, _ in filas])
    return out


# -------------------------------------------------------------------- salida
def tabla_variantes(vals, gates, folds, variantes):
    print(f'  {"variante":<12} {"n":>3}  {"media":>8}  {"sd":>7}   gate')
    print(f'  {"-" * 12} {"-" * 3}  {"-" * 8}  {"-" * 7}   {"-" * 6}')
    for v in variantes:
        xs = [vals[(f, v, s)] for f, s in folds if (f, v, s) in vals]
        gs = [gates[(f, v, s)] for f, s in folds
              if (f, v, s) in gates and not math.isnan(gates[(f, v, s)])]
        sd = st.stdev(xs) if len(xs) > 1 else float('nan')
        g = f'{st.mean(gs):.4f}' if gs else '  —'
        print(f'  {v:<12} {len(xs):>3}  {st.mean(xs):8.3f}  {sd:7.3f}   {g}')


def comparar(vals, celdas, pares, etiqueta):
    print(f'\n  comparaciones pareadas por (fold, semilla) — {etiqueta}')
    print(f'  {"":<22} {"efecto":>16}  {"rel":>7}  {"t":>7}  {"p":>8}  {"a favor":>8}')
    print(f'  {"-" * 22} {"-" * 16}  {"-" * 7}  {"-" * 7}  {"-" * 8}  {"-" * 8}')
    for a, b in pares:
        comunes = [c for c in celdas if (c[0], a, c[1]) in vals and (c[0], b, c[1]) in vals]
        if len(comunes) < 2:
            print(f'  {a} vs {b:<10} — menos de 2 pares en común, no se compara')
            continue
        xa = [vals[(f, a, s)] for f, s in comunes]
        xb = [vals[(f, b, s)] for f, s in comunes]
        m, sd, t, gl, p, favor, n = t_pareado(xa, xb)
        rel = 100 * m / st.mean(xb) if st.mean(xb) else float('nan')
        sig = '*' if p < 0.05 else ' '
        print(f'  {a} − {b:<12} {m:+8.3f} ± {sd:5.3f}  {rel:+6.1f}%  '
              f'{t:+7.2f}  {p:8.4f}{sig} {favor:>4}/{n}')
    print('    (efecto negativo = la primera variante tiene MENOS error; * = p<0,05)')


def main():
    ap = argparse.ArgumentParser(
        description='Agrega los CSV de eval_fase1_seeds.py con una convención declarada.')
    ap.add_argument('csv', nargs='+', help='uno o más *_results.csv (acepta comodines)')
    ap.add_argument('--peso', choices=['objetos', 'escena'], default='objetos',
                    help='cómo promediar ENTRE escenas de validación (default: objetos)')
    ap.add_argument('--poblacion', choices=['todos', 'moviles'], default='todos',
                    help='objetos a incluir (default: todos)')
    ap.add_argument('--metrica', choices=['ade', 'fde'], default='ade')
    ap.add_argument('--comparar', nargs='*', metavar='A:B',
                    help='pares a comparar; sin esto compara todos contra todos')
    ap.add_argument('--por-fold', action='store_true',
                    help='desglosa el efecto fold por fold antes del resumen')
    args = ap.parse_args()

    paths = sorted({p for patron in args.csv for p in glob.glob(patron)} or set(args.csv))
    crudo, gates, escenas, avisos = leer(paths, args.metrica, args.poblacion)
    if not crudo:
        raise SystemExit('no se leyó ninguna fila')
    vals = agregar(crudo, args.peso)

    folds = sorted({f for f, _, _ in vals})
    variantes = sorted({v for _, v, _ in vals})
    semillas = sorted({s for _, _, s in vals})
    celdas = sorted({(f, s) for f, _, s in vals})
    n_esc = max(len(f) for f in crudo.values())

    # La convención va ANTES de la tabla, siempre. Es la regla de este proyecto.
    print('=' * 78)
    print(f'  archivos    : {", ".join(paths)}')
    print(f'  métrica     : {args.metrica.upper()} sobre objetos '
          f'{"MÓVILES" if args.poblacion == "moviles" else "TODOS (móviles + parados)"}')
    print(f'  dentro de escena : media por objeto (la calcula el evaluador)')
    print(f'  entre escenas    : ' + (
        'PONDERADA por número de objetos — cada objeto pesa igual'
        if args.peso == 'objetos' else
        'MEDIA SIMPLE — cada escena pesa igual'))
    print(f'  muestreo    : {len(folds)} fold(s) {folds} × {len(semillas)} semillas '
          f'× {n_esc} escena(s) de validación')
    print(f'  escenas     : {", ".join(sorted(escenas))}')
    for a in avisos:
        for linea in [a[i:i + 74] for i in range(0, len(a), 74)]:
            print(f'  {linea}')
    if avisos:
        print('  ' + '-' * 74)
    if len(folds) == 1:
        print('  AVISO: UN SOLO FOLD. La varianza entre folds es ~3x la de semillas.')
        print('         Un t grande acá NO autoriza a concluir. Ver')
        print('         docs/EXPERIMENTOS_DECODER.md, experimento 11.')
    print('=' * 78)

    print(f'\n  {args.metrica.upper()} por variante')
    tabla_variantes(vals, gates, celdas, variantes)

    if args.comparar:
        pares = [tuple(p.split(':', 1)) for p in args.comparar]
    else:
        pares = [(a, b) for i, a in enumerate(variantes) for b in variantes[i + 1:]]

    if args.por_fold and len(folds) > 1:
        for f in folds:
            celdas_f = [c for c in celdas if c[0] == f]
            comparar(vals, celdas_f, pares, f'fold {f}')
        print('\n  RESUMEN ENTRE FOLDS — n = folds, no corridas.')
        print('  Se promedia el efecto de cada fold y se testea sobre esos promedios.')
        for a, b in pares:
            efectos = []
            for f in folds:
                cf = [c for c in celdas if c[0] == f
                      and (f, a, c[1]) in vals and (f, b, c[1]) in vals]
                if cf:
                    efectos.append(st.mean([vals[(f, a, s)] - vals[(f, b, s)] for _, s in cf]))
            if len(efectos) < 2:
                continue
            m, sd = st.mean(efectos), st.stdev(efectos)
            # Mismo criterio que t_pareado: sd=0 con m=0 es un empate exacto
            # (t=0, p=1), no un efecto infinitamente significativo.
            t = (m / (sd / math.sqrt(len(efectos))) if sd
                 else (float('inf') if m else 0.0))
            p = p_dos_colas(t, len(efectos) - 1) if math.isfinite(t) else 0.0
            print(f'  {a} − {b:<12} {m:+8.3f} ± {sd:5.3f}  t={t:+6.2f}  '
                  f'p={p:.4f}  {sum(1 for e in efectos if e < 0)}/{len(efectos)} folds')
    else:
        comparar(vals, celdas, pares, f'{len(celdas)} pares')

    print()


if __name__ == '__main__':
    main()
