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
| 5 | Wayformer, fine-tune último bloque, 60 ép | *en curso* | — | ⏳ |

**Conclusión provisoria (hasta el experimento 4):** ningún cambio arquitectónico
probado extrae señal generalizable de la escena LiDAR con 20 escenas de
entrenamiento. El baseline puramente cinemático gana de forma consistente y
estadísticamente significativa en las tres variantes con escena.

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

**Reproducir:**
```
conda run -n sapiens_gpu python train_decoder_mini.py \
    --scenes <20 escenas de train del fold 0> \
    --unseen <5 escenas held-out del fold 0> \
    --enc work_dirs/rv_rect_overfit100/epoch_3000.pth --arch wayformer \
    --epochs 60 --eval-every 10 --seed 0 --finetune-blocks 1 --enc-lr 1e-5 \
    --out work_dirs/ft_trajectory
```

*(resultados pendientes de completar)*

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
