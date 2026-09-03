# Resultados validados — el libro mayor

**Qué es esto.** Un índice de los resultados que **se pueden defender**, cada uno
con el número, sobre cuántas semillas y folds está promediado, y **exactamente qué
lo produjo**: config, script, CSV crudo y commit. Es lo que hay que abrir cuando
alguien pregunta "¿de dónde sale ese número?" — en una reunión, en la escritura de
la tesis, o en seis meses.

**Qué NO es.** No es un resumen de todo lo corrido. `work_dirs/` tiene **19 CSV de
resultados** de tres tracks distintos y varios protocolos incompatibles (esquemas
de 9, 10 y 11 columnas). La mayoría **no son comparables entre sí**. Los que no
están acá, no están por algo: protocolo viejo, un solo fold, o una sola semilla.

**Regla que este documento hace cumplir:** ningún número se cita de memoria.
Todos los de acá se recalculan con `agregar_resultados.py` desde el CSV que se
indica. El comando está en cada fila.

---

## Advertencia sobre "el mejor resultado"

El ADE más bajo del proyecto lo tiene **`gate0`: 2,781 m**. Y `gate0` es el
**control de arquitectura** — el modelo completo con la rama de escena congelada
en cero. O sea: **nuestro mejor modelo es el que no usa la escena LiDAR.**

| variante | ADE | n |
|---|---|---|
| **`gate0`** (modelo completo, escena apagada) | **2,781 ± 0,993** | 40 corridas |
| `baseline` (MLP cinemático) | 2,988 ± 1,051 | 40 corridas |
| `gated` (escena activa, `gate_init=0,5`) | 3,504 ± 1,280 | 40 corridas |

Recalcular con:
```bash
cd sapiens/pretrain && python agregar_resultados.py \
    work_dirs/noclipcv/noclipcv_results.csv --por-fold
```

---

## 1. Los tres efectos de Fase 1 (el resultado central de la tesis)

**Población:** 5 folds × 8 semillas × 2 escenas de validación = 40 corridas por
variante. Época fija 100 (sin sesgo de selección). Evaluación `--sin-clip
--eval-windows 7`. **n = 5 folds** (el test se hace sobre los promedios por fold,
no sobre las 40 corridas: la varianza entre folds es ~3× la de semillas).

| efecto | valor | p | folds a favor | lectura |
|---|---|---|---|---|
| capacidad (`gate0 − baseline`) | **−0,207 ± 0,219** | 0,102 | **5/5** | dirección consistente, potencia insuficiente |
| escena, `gate_init=0,5` (`gated − gate0`) | +0,723 ± 0,529 | 0,038 | 0/5 | **contaminado** — ver §2 |
| escena, `gate_init=0,05` (`gated005 − gate0`) | +0,276 ± 0,335 | 0,139 | 0/5 | la escena **no aporta** |

*Efecto positivo = MÁS error.* El gate aprendido converge a **0,0042**: el modelo
cierra la rama de escena por su cuenta.

- **Configs:** `noclip_{base,dec}_fold{0..4}.py`
- **Scripts:** `run_noclip_cv.sh` (baseline, gate0, gated) · `run_gateinit.sh` (gated005)
- **CSV:** `work_dirs/noclipcv/noclipcv_results.csv` · `work_dirs/gateinit/gateinit_results.csv`
- **Commits:** `efd801a` (exp. 19), `df33f99` (exp. 20)
- **Documentado en:** `docs/EXPERIMENTOS_DECODER.md`, experimentos 19 y 20

```bash
python agregar_resultados.py work_dirs/noclipcv/noclipcv_results.csv \
    work_dirs/gateinit/gateinit_results.csv --por-fold \
    --comparar gated:gate0 gated005:gate0 gate0:baseline
```

---

## 2. El hallazgo metodológico (vale por sí solo)

**La inicialización del gate residual le costaba a la variante experimental más de
la mitad de su entrenamiento efectivo.** Con `gate_init=0,5` la rama de escena
entra a media amplitud desde el primer paso, sin haber aprendido nada: el decoder
gasta épocas aprendiendo a callarla. Medido en el fold 0: `gated` pasa sus primeras
~10 épocas con la pérdida clavada en 0,254 mientras `gate0` ya baja a 0,062, y la
pérdida que alcanza en la época 100 su control ya la tenía en la **41**.

Corregir eso movió el efecto de **+0,723 a +0,276** y un **p=0,038 a p=0,139**.

> Un resultado significativo se evaporó al corregir un detalle de inicialización.
> Es la décima retractación del proyecto, y la única encontrada por medición propia
> en menos de 24 h.

Base teórica: **ReZero** (Bachlechner 2020, `papers/`) — un escalar residual
iniciado en cero **sí** recibe gradiente. De ahí que 0,05 sea la salida conservadora
y no 0.

---

## 3. El encoder MAE generaliza (experimento 21)

**Población:** 5 folds, 8 máscaras pareadas por muestra. Pérdida MSE de
reconstrucción.

| población | entrenado | sin entrenar | trivial (predecir 0) | vs trivial |
|---|---|---|---|---|
| 8 ventanas vistas | 0,0691 ± 0,0224 | 1,158 | 0,341 | 79,7 % mejor |
| 48 ventanas nuevas, escenas vistas | 0,1170 ± 0,0082 | 1,163 | 0,338 | 65,4 % mejor |
| **14 ventanas, escenas RETENIDAS** | **0,1913 ± 0,0441** | 1,163 | 0,338 | **43,5 % mejor** |

**Los encoders no memorizaron**, pese a haberse pre-entrenado con 8 muestras. Eso
sostiene el resultado de §1: el gate cierra sobre una escena que **sí significa
algo**, no sobre ruido.

**Lo accionable:** la brecha está en cruzar entre **escenas** (0,117 → 0,191), no
entre ventanas (0,069 → 0,117).

- **Script:** `diagnostico_encoder_mae.py` · **Commit:** `8ca9c4d`

---

## 4. El encoder de 10 sweeps: +31,8 %, no +3,7 % (experimento 23)

Peldaño de 10 sweeps del protocolo de Claudine, **range-view**. No se cambió ni la
arquitectura ni los datos: se arregló **cómo se mide**.

| medición | mejora sobre el modelo sin entrenar |
|---|---|
| ítem 11 del checklist (n=1, época 6000) | +3,7 % |
| **época 50, retenido de 55 imágenes de 5 escenas** | **+31,8 %** |

El óptimo es una **meseta entre las épocas 25 y 50 de 6000**, con caída abrupta
después: 55/55 imágenes a favor de la 50 contra la 400 (t=+10,96).

**Cuidado con la lectura.** Esto dice que el encoder **reconstruye** mejor de lo
registrado. **No** dice que la escena ayude a predecir trayectorias — son cosas
distintas, y §1 sigue en pie. La adenda del mismo experimento lo comprobó: los
encoders de vóxeles **no** están degradados (t=−0,43 entre las épocas 600 y 1000),
así que el resultado de la escena no se cae por ahí.

- **Config:** `rect_overfit10_val.py` · **Script:** `curva_overfit10.py`
- **CSV:** `work_dirs/rect_ov10_fino/curva_overfit10.csv` · **Commits:** `0399972`, `ef394ac`

---

## 5. La historia completa no mejora (experimento 22)

**Población:** 5 folds × 8 semillas × 2 brazos, con `--poblacion-hist 11` en los dos
para que midan los mismos objetos prediciendo el mismo futuro.

```
base11 (1,1 s)   3,832 ± 1,286
base5  (0,5 s)   3,043 ± 1,133
efecto  +0,788 ± 0,951   t=+1,85   p=0,137   1/5 folds
```

**Mecanismo medido:** `base11` alcanza MENOS pérdida de entrenamiento que `base5`
(0,0410 vs 0,0471, n=40 cada uno) y generaliza peor. Sobreajuste.

- **Script:** `run_hist11.sh` · **CSV:** `work_dirs/hist11/hist11_results.csv` · **Commit:** `19ae361`

---

## Lo que NO está validado y no debe citarse

| | por qué |
|---|---|
| todo lo anterior al **30/08/2026** | la escena estaba desalineada en el tiempo para el 43 % de los objetos, y el pos-embed del decoder MAE quedaba en ceros |
| experimentos **15 a 18** | un solo fold (el 0). La varianza entre folds es ~3× la de semillas |
| `work_dirs/{cv,cv_domain,horizon_*,angular,fase1_seeds}` | track `decoder_mini`, congelado, o protocolo viejo sin `--sin-clip` |
| experimento **17** (objetivo GeoMAE) | el encoder **nunca convergió**: condición *sin probar*, no refutada |
| `predictions_global.txt` (lo que muestra el visor) | es del **18/08**, del track decoder_mini — **no es el modelo del que hablan estos resultados** |

---

## Las tres conclusiones que se sostienen

1. **La escena LiDAR auto-supervisada no aporta** a esta escala. Cinco folds, ocho
   semillas, escena alineada, control de arquitectura y sin el artefacto del gate:
   **0/5 folds**, y el gate aprendido cierra a 0,004.
2. **No se puede afirmar que perjudique.** Con el arranque corregido, p=0,139.
3. **El cuello es de datos, no de arquitectura.** Tres negativos independientes
   —escena, capacidad, contexto temporal— apuntan al mismo lugar: 236 ventanas de
   entrenamiento desde 8 escenas, y un MAE pre-entrenado con 8 muestras.
