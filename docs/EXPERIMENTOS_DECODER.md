# Registro de experimentos — decoder mini (Wayformer condicionado vs baseline)

Registro corrido de las pruebas hechas para responder la pregunta central:
**¿la escena LiDAR (vía el encoder MAE) aporta a la predicción de trayectorias
sobre un baseline puramente cinemático (historia del objeto, sin escena)?**

Protocolo común salvo que se indique lo contrario: encoder MAE congelado
(`rv_rect_overfit100/epoch_3000.pth`, 100 sweeps), decoder residual sobre
velocidad constante (`gt - cv`), 25 escenas de `waymo_clean`, horizonte 8s
(16 waypoints a 2Hz), historia del objeto 1.0s (10 frames a 10Hz).

## Resumen ejecutivo (actualizar al final de cada tanda)

| # | Experimento | ADE8 no-visto | vs baseline | Veredicto |
|---|---|---|---|---|
| 1 | Baseline (sin escena, historia sola) | **4.65 ± 1.52** (n=15) | — | referencia |
| 2 | Wayformer, atención cruda (~6784 tok) | 4.97 ± 1.67 (n=15) | pierde, t=-3.07 | ❌ |
| 3 | Wayformer, pooling 16 latentes | 5.00 ± 1.67 (n=15) | pierde, t=-4.17 | ❌ |
| 4 | Wayformer, fine-tune último bloque encoder (20 ép) | 2.84 ± 0.29 (n=3, solo fold 0) | pierde, t=-2.63 | ❌ |
| 5 | Wayformer, fine-tune último bloque, 60 ép, 1 semilla | 2.51 (mejor, ép.20) | **gana** vs wayformer congelado (2.65) | ⚠️ mixto — ver nota |
| 6 | Barrido de horizonte 1s/3s/5s/8s (5 folds, 1 semilla) | ver curva abajo | pierde en TODOS los horizontes | ❌ el "punto dulce" de Fase 1 NO reaparece |

**Conclusión provisoria:** ningún cambio de PUENTE (crudo, pooling) extrae
señal generalizable de la escena con 20 escenas de train. El fine-tuning
parcial del encoder muestra la primera señal de mejora en ADE8 (exp. 5,
época 20) pero con una regresión seria y simultánea en la accuracy de
validez (1.00 → 0.54) — no es una mejora limpia. Además, ese mismo punto
(fold 0, semilla 0, época 20) dio un número distinto en el experimento 4
(2.79) que en el 5 (2.51) con configuración nominalmente idéntica —
indicio de no-determinismo de GPU en las operaciones de atención, que hay
que tener en cuenta antes de sacar conclusiones de una sola semilla.

---

## Experimento 1-2: Validación cruzada baseline vs Wayformer (crudo)

**Fecha:** 2026-07-29. **Commit:** `01dd12a`.

**Diseño:** 5 folds (25 escenas / 5, partición alfabética fija) × 3 semillas
[0,1,2] = 15 mediciones por arquitectura. Comparación PAREADA (mismo fold +
semilla en ambos modelos). Reemplaza una medición única anterior (escena
`82f9`, ADE8 7.19 Wayformer vs 7.85 baseline) que sugería lo contrario y
resultó no ser representativa.

**Resultado:**
```
baseline     ADE8 = 4.65 ± 1.52  (min 2.30, max 6.89, n=15)
wayformer    ADE8 = 4.97 ± 1.67  (min 2.18, max 7.24, n=15)
baseline - wayformer: diff -0.32 ± 0.40  (baseline gana 12/15)  t=-3.07
```

**Diagnóstico:** en 9/15 corridas de Wayformer el mejor checkpoint fue la
época 1 (early-stop descartó todo el entrenamiento posterior por no
generalizar) — la corrección condicionada en escena no aporta.

**Reproducir:**
```
conda run -n sapiens_gpu python cross_validate_decoder.py \
    --enc work_dirs/rv_rect_overfit100/epoch_3000.pth \
    --epochs 100 --archs wayformer baseline
```

---

## Experimento 3: Puente con pooling aprendido (Perceiver-style)

**Fecha:** 2026-07-29. **Commit:** `d24ae24`.

**Hipótesis:** la atención cruda sobre ~6784 tokens crudos del encoder da
demasiada capacidad de sobreajuste con solo 20 escenas de train; resumir cada
sweep a 16 latentes aprendidos ANTES de la atención final debería reducir el
sobreajuste y mejorar la generalización.

**Arquitectura:** `MiniWayformerPooled` — 16 latentes compartidos entre
sweeps hacen `MultiheadAttention` sobre los tokens crudos → produce 16 tokens
resumen por sweep → esos alimentan el `TransformerDecoder` de slots (igual
que antes, pero sobre 16 tokens en vez de ~6784).

**Resultado:**
```
wayformer_pooled  ADE8 = 5.00 ± 1.67  (n=15)
wayformer - wayformer_pooled: diff -0.03 ± 0.29  (empatan, t=-0.36)
baseline - wayformer_pooled: diff -0.34 ± 0.32  (baseline gana 13/15)  t=-4.17
```

**Diagnóstico:** el pooling SÍ ayuda a optimizar (8/15 corridas superan el
piso CV en época >1, vs 6/15 del diseño crudo) pero lo aprendido NO
generaliza mejor — resultado final estadísticamente idéntico al crudo.
**Descarta el diseño del puente como causa principal** (2 diseños distintos,
mismo resultado).

**Reproducir:**
```
conda run -n sapiens_gpu python cross_validate_decoder.py \
    --archs wayformer_pooled --epochs 100
```

---

## Experimento 4: Fine-tuning parcial del encoder (piloto, 20 épocas)

**Fecha:** 2026-07-29. **Commit:** `2d4da08`.

**Hipótesis:** si ni el puente crudo ni el pooling ayudan, quizás el
problema es que el encoder está 100% congelado — sus features fueron
entrenadas solo para reconstruir píxeles de la range-view, sin ninguna
señal relacionada con movimiento. Descongelar la última capa podría
permitir una adaptación mínima a la tarea sin destruir lo aprendido.

**Diseño:** descongelado el último de los 6 bloques transformer del encoder
(1.77M de 13.7M parámetros) + norma final, con lr propio 100x menor que el
decoder (1e-5 vs default). Piloto acotado: solo fold 0 (el mismo split de
escenas de los experimentos 1-3), 3 semillas, 20 épocas — comparación
directa contra las filas de Wayformer-crudo de ese mismo fold antes de
invertir en las 5 folds completas.

**Resultado (comparación directa, mismo fold, mismas semillas):**

| Semilla | Wayformer congelado (exp. 2) | Wayformer + fine-tune último bloque |
|---|---|---|
| 0 | 2.65 | 2.79 |
| 1 | 3.02 | 3.16 |
| 2 | 2.18 | 2.58 |

```
wayformer_ft1  ADE8 = 2.84 ± 0.29  (n=3, solo fold 0)
wayformer - wayformer_ft1: diff -0.22 ± 0.15  (wayformer congelado gana 3/3)  t=-2.56
baseline - wayformer_ft1: diff -0.50 ± 0.33  (baseline gana 3/3)  t=-2.63
```

**Diagnóstico:** el fine-tuning empeoró el resultado de forma consistente en
las 3 semillas frente al mismo encoder congelado. Dado el patrón claro y
consistente con un piloto barato, se decide NO escalar esta configuración
a las 5 folds completas.

**Salvedad metodológica:** con `eval_every=20` y `epochs=20` solo hay 2
puntos de medición (época 1 y época 20) — no se puede saber si el modelo
venía mejorando o empeorando en el camino. El experimento 5 corrige esto.

**Reproducir:**
```
conda run -n sapiens_gpu python cross_validate_decoder.py \
    --enc work_dirs/rv_rect_overfit100/epoch_3000.pth \
    --epochs 20 --folds 0 --finetune-blocks 1 --enc-lr 1e-5 --archs wayformer
```

---

## Experimento 5: Fine-tuning parcial, trayectoria completa (60 épocas, 1 semilla)

**Fecha:** 2026-07-29 (tarde/noche). **Estado:** en curso.

**Motivo:** corregir la salvedad metodológica del experimento 4 — con
evaluación cada 10 épocas (en vez de 2 puntos) se puede ver si el
desempeño en no-visto mejora, se estanca o empeora con más tiempo de
entrenamiento, antes de descartar definitivamente la idea del fine-tuning.

**Diseño:** fold 0, semilla 0 (mismo split), 60 épocas, `eval_every=10`
(7 puntos de medición: ép. 1, 10, 20, 30, 40, 50, 60).

**Resultado (fold 0, semilla 0):**

| Época | ADE8 no-visto | ADE5 | FDE | Accuracy validez |
|---|---|---|---|---|
| 1 | 3.27 | 2.13 | 5.48 | 1.00 |
| 10 | 2.60 | 1.51 | 4.48 | 0.54 |
| **20** (mejor) | **2.51** | 1.70 | 3.98 | 0.54 |
| 30 | 2.87 | 1.73 | 5.11 | 0.59 |
| 40 | 2.87 | 1.78 | 5.03 | 0.54 |
| 50 | 2.81 | 1.70 | 5.01 | 0.54 |
| 60 | 2.57 | 1.56 | 4.52 | 0.54 |

**Diagnóstico — resultado MIXTO, no una mejora limpia:**

1. **Confirma sobreajuste más allá de ép.20**: mejora rápida ép.1→20, luego
   empeora (ép.30-40) y nunca vuelve a bajar del óptimo de ép.20. La curva
   confirma que 20 épocas (experimento 4) no era "cortar muy pronto".
2. **En su mejor punto, SÍ supera al wayformer congelado** en el mismo
   fold/semilla: 2.51 vs 2.65 (experimento 1-2, fold 0 semilla 0).
3. **Pero con una regresión seria simultánea**: la accuracy de validez de
   objetos cae de 1.00 a 0.54 en el mismo checkpoint — el modelo mejora la
   trayectoria pero empeora mucho la clasificación de "¿este slot es un
   objeto real?" (casi al azar). No es una mejora limpia.
4. **Discrepancia de no-determinismo**: el mismo punto (fold 0, semilla 0,
   época 20) dio 2.79 en el experimento 4 y 2.51 acá, con configuración
   nominalmente idéntica (la única diferencia es `eval_every`, que no
   debería alterar los pesos entrenados). Indica no-determinismo de GPU
   en las operaciones de atención — una sola semilla no es enteramente
   reproducible en este pipeline. Refuerza la necesidad de validar con
   múltiples semillas antes de confiar en cualquier número puntual.

**Próximo paso sugerido (no ejecutado aún):** si se quiere seguir esta
línea, habría que (a) investigar por qué cae la accuracy de validez
(¿ponderar más la pérdida BCE? ¿fine-tunear el encoder recién después de
que las cabezas se estabilicen?), y (b) repetir con 3 semillas × early
stopping cada 10 épocas antes de considerar esto una mejora real.

**Reproducir:**
```
conda run -n sapiens_gpu python train_decoder_mini.py \
    --scenes 2e41fe6faf5cd2ea 367b072edc9822ea 394e61f27c2a1700 4014ae5bcda2726f \
             4a2ef30000d19d90 4b60f9400a30ceaf 7e2f727866c69ea0 82f90331a1dfe968 \
             92ab54c34f237728 9e897ff552287bea 9ea216a54ee07b49 9fffe68876965f2e \
             aaccfa0a1132fb83 adce80bac21c1895 ae3d6f946b8e7871 d2399ea6a028ecb2 \
             e52c6a9366981ad e75176fd226ea04a f2ca03b1434a27e4 f7cc90b8f4611d4d \
    --unseen 2a81f5233075e987 41692b0ec7ff4123 8e0342468563ae5e a20f67087b9a288 \
             db4edc9bd0c9d18c \
    --enc work_dirs/rv_rect_overfit100/epoch_3000.pth --arch wayformer \
    --epochs 60 --eval-every 10 --seed 0 --finetune-blocks 1 --enc-lr 1e-5 \
    --out work_dirs/ft_trajectory
```

---

## Contexto y referencias

- El resultado positivo inicial (`7.19` vs `7.85` en la escena `82f9` sola,
  commit `0144623`) quedó **revertido** por el experimento 1-2. Se mantiene
  en el historial de memoria del proyecto como ejemplo documentado de por
  qué la validación cruzada es necesaria antes de reportar una conclusión.
- Paper relacionado: *WOMD-LiDAR: Raw Sensor Dataset Benchmark for Motion
  Forecasting* (Chen et al., Waymo) — su propio baseline supervisado
  (SWFormer + Wayformer) también reportó mejora marginal en ADE al agregar
  LiDAR, incluso con ~100k escenas (4000x más que las 25 disponibles acá).
- Infraestructura: `sapiens/pretrain/train_decoder_mini.py` (función
  `train_decoder()`, única fuente de verdad del loop de entrenamiento) y
  `sapiens/pretrain/cross_validate_decoder.py` (driver de validación
  cruzada, extensible a nuevas arquitecturas vía `--archs`).

---

## Experimento 6: Barrido de horizonte de predicción (1s / 3s / 5s / 8s)

**Fecha:** 2026-08-02. **Script:** `horizon_sweep.py`.

**Hipótesis:** en la Fase 1 (pipeline viejo: encoder de vóxeles + gate) la
escena LiDAR mostró un "punto dulce" a 3s (+25% de mejora), degradándose a
1s (neutral) y 5s (neutral). Todos los experimentos del decoder MAE (1-5)
fueron a 8s — más lejos que ese pico. Quizás la escena SÍ ayuda a un
horizonte menor y estábamos midiendo en el punto equivocado.

**Diseño:** por cada horizonte (2/6/10/16 waypoints = 1/3/5/8s),
cross-validation 5 folds × 1 semilla × 2 archs (wayformer vs baseline).
Screening (1 semilla): el objetivo es la TENDENCIA del beneficio de la
escena a lo largo del horizonte. Encoder 100sw congelado, features
cacheadas (independientes del horizonte).

**Resultado:**

| Horizonte | Wayformer (con escena) | Baseline (sin escena) | Diff pareada (way−base) | Señal |
|---|---|---|---|---|
| 1s | 0.52 ± 0.20 | 0.44 ± 0.15 | +0.07 ± 0.22 | baseline (2/5) |
| 3s | 1.42 ± 0.51 | 1.31 ± 0.33 | +0.11 ± 0.41 | baseline (3/5) |
| 5s | 2.65 ± 0.85 | 2.49 ± 0.72 | +0.16 ± 0.22 | baseline (1/5) |
| 8s | 5.03 ± 1.78 | 4.65 ± 1.62 | +0.38 ± 0.27 | baseline (0/5) |

*(diff = ADE_wayformer − ADE_baseline en metros; negativo = la escena ayuda)*

**Diagnóstico — el "punto dulce" de 3s NO se reproduce con el encoder MAE:**

1. La escena **no ayuda a ningún horizonte** — la diff es positiva (baseline
   gana) en los 4.
2. **El daño crece con el horizonte** (+0.07 → +0.11 → +0.16 → +0.38), lo
   contrario de un pico en 3s.
3. **Matiz honesto**: a 1s y 3s la diff es chica y con desvío mayor que la
   media (±0.22 y ±0.41) → ahí es prácticamente un empate/ruido, no un
   daño claro. El daño solo es nítido a 8s (baseline gana 5/5). O sea: a
   horizonte corto la escena es *neutral* (no ayuda ni molesta); a horizonte
   largo *molesta*. Nunca ayuda.
4. **Por qué difiere de Fase 1**: aquel +25% a 3s usaba encoder de vóxeles
   **re-pre-entrenado en las escenas de train** + un mecanismo de *gate*
   aprendible — un pipeline distinto. Ese resultado NO transfiere al encoder
   MAE range-view congelado. Sugiere que la diferencia estaba en el
   encoder/gate, no en el horizonte.

**Conclusión:** descarta la hipótesis del horizonte. Junto con los
experimentos 2-5 (puente crudo, pooling, fine-tuning), son cuatro ángulos
distintos que confirman lo mismo: **con el encoder MAE congelado y 20
escenas de train, la escena LiDAR no aporta señal predictiva sobre la
cinemática + histórico del objeto, a ningún horizonte.** Consistente con el
paper WOMD-LiDAR (mejora marginal de ADE incluso con features supervisadas
y ~100k escenas).

**Reproducir:**
```
conda run -n sapiens_gpu python horizon_sweep.py \
    --enc work_dirs/rv_rect_overfit100/epoch_3000.pth --epochs 100
```
