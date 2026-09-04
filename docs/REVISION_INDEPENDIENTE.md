# Revisión independiente del proyecto

Cómo pedirle a otra sesión —otra cuenta, otro modelo— que audite este proyecto sin
heredar las conclusiones de quien lo escribió.

## Por qué en dos etapas

El repo contiene las conclusiones: `CLAUDE.md`, `docs/EXPERIMENTOS_DECODER.md` y las
30 trampas del `CODEBASE_MAP.md`. Un revisor que los lea primero deja de auditar los
datos y pasa a verificar un relato. Los once errores documentados de este proyecto
fueron de **interpretación**, no de cómputo — así que anclarlo al relato desactiva
justo lo que se quiere medir.

Por eso: primero a ciegas sobre los CSV, después el contraste.

---

## Etapa 1 — a ciegas (pegar tal cual)

```
Este repositorio es una tesis de maestría sobre predicción de trayectorias a partir
de LiDAR. Quiero una auditoría INDEPENDIENTE de sus resultados.

REGLA IMPORTANTE: en esta etapa NO leas `docs/`, `CLAUDE.md` ni ningún README.
Contienen las conclusiones del autor y quiero las tuyas, no una verificación de las
suyas. Limitate a:
  - los CSV en `sapiens/pretrain/work_dirs/*/*results*.csv`
  - el agregador `sapiens/pretrain/agregar_resultados.py` (leé su cabecera: fija la
    convención de promediado)
  - el código en `sapiens/pretrain/*.py` y `sapiens/pretrain/mmpretrain/`

Cada CSV tiene una fila por (fold, variante, semilla, escena). Contestá:

1. ¿Qué experimentos se corrieron y qué compara cada uno? Deducilo de los CSV.
2. Para cada comparación: ¿cuál es el efecto, y sobre CUÁNTOS folds y CUÁNTAS
   semillas? Decí siempre el n. La varianza entre folds acá es ~3x la de semillas,
   así que el n que vale es el de folds.
3. ¿Qué conclusiones se sostienen y cuáles no? Marcá explícitamente los números que
   NO alcanzan para afirmar nada.
4. ¿Ves algún problema metodológico en cómo se midió? Por ejemplo comparaciones
   entre poblaciones distintas, métricas no comparables entre sí, o resultados que
   dependen de una sola semilla o un solo fold.
5. Auditá el código que genera esos CSV. ¿Hay algún bug que produciría números
   plausibles pero incorrectos?

No me digas que algo "parece bien". Recalculá y mostrame los comandos.
```

## Etapa 2 — el contraste (después de la etapa 1)

```
Ahora sí leé `docs/EXPERIMENTOS_DECODER.md`, `docs/CODEBASE_MAP.md` (sección
Trampas) y `CLAUDE.md`. La rama `resultados/validados` tiene además
`docs/RESULTADOS_VALIDADOS.md`, un registro de lo que el autor considera defendible.

Compará contra TU análisis de la etapa 1 y decime:

1. ¿Dónde las conclusiones del autor van MÁS LEJOS de lo que los datos sostienen?
2. ¿Hay algún número citado en los docs que no puedas reproducir desde los CSV?
3. ¿Alguna de las 30 trampas documentadas está mal descrita, o hay alguna que el
   autor no vio?
4. ¿Coincidís con el diagnóstico general del estado del proyecto?

Sé específico y contradecime donde corresponda. El autor tuvo once retractaciones
documentadas; la número doce me interesa más que un visto bueno.
```

---

## Qué necesita el revisor, y qué no

| para... | alcanza con | falta |
|---|---|---|
| recalcular todos los números | clonar el repo | nada |
| auditar el código | clonar el repo | nada |
| verificar docs contra CSV | clonar el repo | nada |
| **re-correr un experimento** | — | los datos (ver abajo) y una GPU |

Los 33 CSV de resultados **sí** están versionados a propósito. Los datos
(`waymo_clean`, 8,3 GB) y los checkpoints (`*.pth`) no: están en `.gitignore`.

Eso significa que el revisor puede auditar **la interpretación** —que es donde
estuvieron todos los errores históricos— pero no **la generación** de los números.
Para lo segundo hace falta subir datos.

## Si además se quieren reproducir corridas

Tres cosas, por orden de valor sobre peso:

1. **Los logs de entrenamiento** — 1.175 archivos, **61 MB**. Livianos y muy
   auditables: traen la configuración efectiva de cada corrida, las curvas de
   pérdida y el gate aprendido época a época. Permiten detectar que un arreglo no
   llegó al config sin re-correr nada. Van directo al repo.
2. **El subset de las 10 escenas de Fase 1** — **366 MB** (`bin_files`,
   `objs_bbox`, `poses` de esas escenas). Con eso se reproduce cualquier corrida de
   los experimentos 15-27. Conviene como *GitHub Release* (admite hasta 2 GB por
   archivo) y no como commit, para no inflar el historial:
   `gh release create datos-fase1 waymo_clean_fase1.tar.gz`
3. **Los encoders extraídos** — ~1,2 GB cada uno. Solo si se quiere reproducir el
   decoder sin re-pre-entrenar el MAE (20 min por fold en una RTX 4060). En general
   no vale la pena: es más barato regenerarlos.

## Advertencia sobre el estado

Antes de clonar, verificar que no haya un experimento a mitad de camino: un CSV
parcial lleva a conclusiones que cambian al día siguiente. `git log -1` y el estado
de `work_dirs/*/` lo dicen.
