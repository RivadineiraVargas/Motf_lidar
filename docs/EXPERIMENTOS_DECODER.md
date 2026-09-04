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
| 6 | Barrido de horizonte 1s/3s/5s/8s (5 folds, 1 semilla) | ver curva abajo | pierde en TODOS los horizontes | ❌ el "punto dulce" de Fase 1 NO reaparece — **revertido por el exp. 8** |
| 7 | Encoder MAE **adaptado al dominio** (fold 0, 3 semillas, 8s) | 2.287 ± 0.052 (n=3) | gana, t=−2.10 (3/3, no signif.) | ⚠️ primera señal a favor |
| 8 | Barrido de horizonte con **encoder de dominio** (fold 0) | **0.726 ± 0.079 @3s** (n=8) | **gana, t=−5.94, p=0.0006, −20.4%** | ✅ **el pico de 3s reaparece** |
| 9 | Gate aprendible sobre la escena (fold 0, 3s, 8 semillas) | 0.759 ± 0.047 (n=8) | gana a baseline (t=−8.80) pero **empata con exp. 8** (t=+1.16) | ➖ no aporta; el valor aprendido sí informa |
| 10 | **Réplica en el fold 4** del exp. 8 (3s, 8 semillas) | 1.792 ± 0.116 (n=8) | empata, t=−0.59 (4/8), −1.3% | ❌ **el efecto del fold 0 NO se replica** |

**OJO CON LA ESCALA — no comparar entre bloques.** Los experimentos 1-3 y 6
son promedios de **5 folds**; los 4, 5, 7, 8 y 9 son **solo fold 0** y el 10
es **solo fold 4**. La dificultad cambia mucho entre folds (baseline 8s: 4.65
a 5 folds vs 2.34 en el fold 0; baseline 3s: 0.912 en el fold 0 vs 1.816 en el
fold 4). Sólo son comparables entre sí las filas del mismo fold y el mismo
horizonte — por eso todas las comparaciones se hacen PAREADAS contra el
baseline medido en ese mismo fold/semilla.

**Conclusión al 2026-08-07 — el efecto es REAL pero DEPENDE DEL SPLIT.**
En el fold 0, con encoder adaptado al dominio y midiendo a 3s, la escena
aporta de forma contundente: **−20.4% de ADE (t=−5.94, p=0.0006, 8/8
semillas)**. En el fold 4, mismo protocolo y mismas 8 semillas: **−1.3%,
t=−0.59, 4/8 — un nulo**. Media de los dos folds: −0.105, con sd ENTRE
folds de 0.115. **No corresponde afirmar "la escena ayuda" sin más folds.**

> ⚠️ Una versión previa de esta sección (escrita el 06/08, antes del exp. 10)
> decía que la hipótesis quedaba "sostenida bajo dos condiciones
> identificadas". Eso se escribió con un solo fold medido y quedó
> sobrevendido; el exp. 10 lo corrige.

**Lo que sí quedó establecido:**

1. Los seis experimentos negativos (1-6) compartían dos defectos que ninguno
   controlaba a la vez: encoder MAE **genérico** y horizonte **8s**.
   Corrigiendo ambos aparece señal real, al menos en algunos splits.
2. De los 3 ingredientes de la Fase 1, quedan replicados 2 (encoder de
   dominio + horizonte 3s); el 3ro, el gate, **no aporta** (exp. 9) — pero
   converge a 0.092 ± 0.005, evidencia independiente de que la señal de
   escena es real y chica.
3. **La variable que más manda es el SPLIT, no la arquitectura.** Con el
   encoder genérico a 3s, la diff por fold ya iba de −0.153 (fold 1) a
   +0.815 (fold 3); el "+0.109 promedio" del exp. 6 lo empujaba casi
   entero un solo fold. Con 25 escenas, cada fold retiene 5 y basta una con
   maniobras atípicas para mover la media.

**Lo que sigue sin estar probado:** faltan los folds 1, 2 y 3 (un encoder de
dominio por fold, ~12.5h c/u ≈ 37h de GPU) para tener una respuesta de 5
folds. Previsión con los datos actuales: media entre −0.05 y −0.10,
negativa pero probablemente no significativa, dominada por el fold 3.
Sería consistente con el paper WOMD-LiDAR (mejora marginal aun con ~100k
escenas y features supervisadas).

**Advertencia metodológica vigente (de los exp. 4-5):** el mismo punto
(fold 0, semilla 0, ép. 20) dio 2.79 en el exp. 4 y 2.51 en el 5 con
configuración nominalmente idéntica — no-determinismo de GPU en las
operaciones de atención. Se confirmó otra vez en el exp. 8: el renglón de
8s reprodujo la MEDIA del exp. 7 (2.29 en ambos) con semillas individuales
muy distintas (sd 0.24 vs 0.05). **Fiarse de medias sobre varias semillas,
nunca de una corrida suelta.**

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

---

## Experimento 7: Encoder MAE adaptado al dominio

**Fecha:** 2026-08-04. **Commit:** `b57a986`.
**Script:** `sapiens/pretrain/run_domain_encoder_experiment.sh`.

**Hipótesis:** los experimentos 1-6 usaron siempre el encoder MAE *genérico*
(`rv_rect_overfit100`, 100 sweeps de 24 escenas). En la Fase 1, el +25% a 3s
vino de un encoder de vóxeles **re-pre-entrenado en las escenas de train**.
¿Y si el cuello de botella no era el puente ni el horizonte, sino que las
features del encoder genérico no separan información de movimiento?

**Diseño:** encoder MAE re-pre-entrenado desde cero SOLO en las 20 escenas de
train del fold 0 (`config_rangeview_rect_fold0.py`, 1000 ép, loss 2.15→0.40,
~12.5h), sin fuga de las 5 retenidas. Después, decoder wayformer con ese
encoder **congelado**, 3 semillas, fold 0. Aísla UNA variable: el encoder.

**Resultado (fold 0, 3 semillas, ADE8):**

| Configuración | ADE8 |
|---|---|
| Wayformer + encoder **dominio** | **2.287 ± 0.052** |
| Baseline sin escena | 2.342 ± 0.038 |
| Wayformer + encoder genérico (exp. 2) | 2.620 ± 0.421 |

```
dominio - baseline: -0.055 ± 0.045  (a favor de la escena 3/3)  t=-2.10
   -> t_crit(gl=2) = 4.303  =>  NO significativo (p ~ 0.052)
```

**Diagnóstico:** primera vez en 6 experimentos que la escena no daña. Señales
secundarias: `best_ep` 40/40/20 (aprende mucho más allá del piso CV) contra
`best_ep`=1 en 2/3 con el genérico, y la sd entre semillas colapsa de 0.421 a
0.052 — el encoder de dominio destraba la optimización.

**Refuerzo (importante):** el encoder genérico se entrenó sobre 24 escenas
excluyendo sólo `82f9…` (ver `utilities/make_rect_png_100.py`), o sea que
**vio en auto-supervisado las 5 escenas retenidas del fold 0**. Tenía ventaja
de fuga y aun así perdió contra el de dominio.

**Reproducir:**
```
bash sapiens/pretrain/run_domain_encoder_experiment.sh
```

---

## Experimento 8: Barrido de horizonte CON el encoder de dominio

**Fecha:** 2026-08-06. **Script:** `horizon_sweep.py` (ahora con
`--folds`, `--seeds`, `--horizons`, `--archs`).

**Hipótesis:** el experimento 6 concluyó "no hay punto dulce a 3s", pero midió
el encoder **genérico** — el que el experimento 7 identificó como cuello de
botella. La pregunta del horizonte quedaba sin responder para el encoder
adaptado al dominio.

**Diseño:** fold 0 (obligatorio: usar el encoder de dominio en otro fold sería
FUGA), horizontes 1s/3s/5s/8s. 3 semillas; a 3s se ampliaron a **8 semillas**
al ver la señal. Features ya cacheadas (`cache_fold0_domain`) → ~1h total.

**Resultado (diff pareada = wayformer − baseline; negativo = la escena ayuda):**

| Horizonte | n | Diff (encoder DOMINIO) | t | Relativo | Diff (encoder genérico, exp. 6) |
|---|---|---|---|---|---|
| 1s | 3 | +0.001 ± 0.051 | 0.05 | +0.4% | +0.07 |
| **3s** | **8** | **−0.186 ± 0.089** | **−5.94** | **−20.4%** | +0.11 |
| 5s | 3 | −0.182 ± 0.213 | −1.48 | −11.7% | +0.16 |
| 8s | 3 | −0.053 ± 0.206 | −0.44 | −2.3% | +0.38 |

A 3s, con 8 semillas: **p = 0.00058**, IC95% `[-0.247, -0.124]` (no incluye el
cero), **8/8 semillas a favor de la escena**.

**Diagnóstico:**

1. **El signo se invierte en 3s y 5s** respecto del encoder genérico. El
   experimento 6 medía el encoder equivocado; el punto dulce existe y depende
   de que el encoder esté adaptado al dominio.
2. **La forma de la curva replica la Fase 1**: neutral a 1s, pico a 3s, decae
   después. El −20.4% es del mismo orden que el +25% de Fase 1 con vóxeles.
3. **Mecanismo coherente:** a 1s la velocidad constante ya es casi perfecta
   (0.32 m) y no hay nada que aportar; a 3s el baseline se estanca (`best_ep`=1
   en 2/3, piso CV) y la escena corrige de verdad; a 8s la señal se diluye en
   la incertidumbre acumulada.
4. **Las 5 semillas extra bajaron la media de −0.210 a −0.186** (regresión a la
   media leve) pero la sd NO se disparó (0.089) → no era ruido de semillas.

**Reproducir:**
```
conda run -n sapiens_gpu python horizon_sweep.py \
    --enc work_dirs/rv_rect_fold0/epoch_1000.pth \
    --folds 0 --seeds 0 1 2 3 4 5 6 7 --horizons 3s \
    --cache work_dirs/cache_fold0_domain --out work_dirs/horizon_domain \
    --epochs 100
```

---

## Experimento 9: Gate aprendible sobre la rama de escena

**Fecha:** 2026-08-06. **Arquitectura:** `MiniWayformerGated` +
`GatedDecoderLayer` en `train_decoder_mini.py` (arch `wayformer_gated`).

**Hipótesis:** tercer y último ingrediente de la Fase 1. Allá un escalar
aprendido `tanh(scene_gate)` (init 0.5) escalaba la rama de escena, dejando
que el modelo aprendiera *cuánto* condicionar en ella. El diagnóstico viejo
(escena `9e89`) era que los modelos sobre-corrigen justo donde la velocidad
constante ya es perfecta.

**Implementación:** hubo que escribir la capa de decoder a mano — en
`nn.TransformerDecoderLayer` la cross-attention está fusionada con la
self-attention y el FFN, y el gate tiene que escalar **sólo** la rama de
escena (`x = n2(x + g * cross_attn(x, mem, mem))`). Escalar la memoria de
entrada no equivale: la softmax normaliza sobre las claves. Un único gate
compartido entre las 2 capas (como el escalar de Fase 1), por eso el
parámetro vive en el módulo padre y entra por `forward`. `gate_init=0.5` y no
0: arrancar en 0 anula el gradiente de toda la rama y el gate no abre nunca
(candado documentado en Fase 1).

**Resultado (fold 0, 3s, 8 semillas, ADE@3s):**

| Modelo | ADE@3s |
|---|---|
| Wayformer **gated** | 0.759 ± 0.047 |
| Wayformer sin gate (exp. 8) | 0.726 ± 0.079 |
| Baseline sin escena | 0.912 ± 0.022 |

```
gated  - baseline: -0.152 ± 0.049  (8/8)  t=-8.80   -> la escena ayuda
gated  - ungated : +0.033 ± 0.082  (4/8)  t=+1.16   -> EMPATAN
```

**Diagnóstico — el gate NO aporta, pero lo que aprende sí informa:**

1. **Empata con el wayformer común** (t=1.16, 4/8). El tercer ingrediente de
   Fase 1 no transfiere. Explicación: allá el decoder era un MLP con la escena
   *concatenada*, donde sin gate la rama entraba siempre a fuerza completa;
   acá la cross-attention con conexión residual ya puede atenuar la escena por
   su cuenta, así que el gate es redundante.
2. **El valor aprendido es el hallazgo real.** Las 8 semillas convergen a
   `tanh(scene_gate)` = **0.0917 ± 0.0051** partiendo de 0.500 — el modelo
   decide solo, con altísima reproducibilidad, que la escena debe entrar al
   **~9% de su fuerza**. Y con esa atenuación igual le gana al baseline por
   0.152 (t=−8.80). Corrobora desde un ángulo independiente el diagnóstico del
   experimento 3 (pooling): **la señal de escena es real pero chica, y demasiada
   capacidad cruda la ahoga.** El pooling la atacó reduciendo tokens; el gate,
   reduciendo amplitud; ambos apuntan a lo mismo.
3. **`best_ep`=1 en 8/8**: aprende la corrección útil en UNA época y de ahí en
   adelante sólo sobreajusta (el ungated mejora hasta ép. 20-80 en 5/8).
4. **Nota de método:** con 1 sola semilla este experimento parecía un claro
   empeoramiento (0.81 vs 0.675, `best_ep` 1 vs 20). Con 8 semillas, empatan.
   Otro caso del no-determinismo advertido en el resumen ejecutivo.

**Reproducir:**
```
conda run -n sapiens_gpu python horizon_sweep.py \
    --enc work_dirs/rv_rect_fold0/epoch_1000.pth \
    --folds 0 --seeds 0 1 2 3 4 5 6 7 --horizons 3s --archs wayformer_gated \
    --cache work_dirs/cache_fold0_domain --out work_dirs/horizon_domain \
    --epochs 100
```

---

## Experimento 10: Réplica en un segundo fold (¿generaliza el −20.4%?)

**Fecha:** 2026-08-06/07. **Script:** `sapiens/pretrain/run_fold4_experiment.sh`
(encadena encoder → decoder). **Config:** `config_rangeview_rect_fold4.py`.

**Hipótesis:** todo el bloque 7-9 sale del **fold 0**. La varianza ENTRE folds
era la fuente dominante de ruido (sd 0.326 a 8s, contra 0.089 entre semillas),
así que un solo split no puede sostener el hallazgo. ¿El −20.4% a 3s aparece
también en otro fold?

**Diseño:** encoder MAE re-pre-entrenado desde cero SOLO en las 20 escenas de
train del **fold 4** (1000 ép, loss 2.15→0.3907, ~13h; el del fold 0 cerró en
0.4007 — trayectorias casi calcadas). Después, decoder a 3s con **8 semillas**,
wayformer y baseline, evaluando en las 5 escenas retenidas del fold 4. Mismo
protocolo que el exp. 8, cambiando sólo el split.

**Resultado (ADE@3s):**

| Fold | Wayformer | Baseline | Diff pareada | t | A favor | Relativo |
|---|---|---|---|---|---|---|
| 0 (exp. 8) | 0.726 ± 0.079 | 0.912 ± 0.022 | **−0.186 ± 0.089** | **−5.94** | 8/8 | −20.4% |
| **4 (este)** | 1.792 ± 0.116 | 1.816 ± 0.007 | **−0.024 ± 0.115** | −0.59 | 4/8 | −1.3% |

```
media de los 2 folds: -0.105   |   sd ENTRE folds: 0.115
sd entre semillas dentro de cada fold: 0.089 (f0) / 0.115 (f4)
```

**Diagnóstico:**

1. **El efecto no se replica.** En el fold 4 es un nulo limpio (4/8 semillas,
   t=−0.59). El fold 0 no era representativo del conjunto.
2. **El fold 4 es un split mucho más difícil**: baseline 1.816 contra 0.912 del
   fold 0 (2x). Y su baseline es extraordinariamente estable (±0.007, `best_ep`=1
   casi siempre) — o sea que ahí la velocidad constante es difícil de mejorar y
   el margen donde la escena podría aportar es más chico.
3. **ERROR DE SELECCIÓN, documentado a propósito:** el fold 4 se eligió como
   "caso adversarial" citando diff **+0.834** con encoder genérico... pero ese
   número es **a 8s**. A 3s — el horizonte que se iba a medir — el ranking por
   fold con encoder genérico era otro:

   | fold | 0 | 1 | 2 | 3 | 4 |
   |---|---|---|---|---|---|
   | diff @3s (genérico, 1 semilla) | −0.066 | −0.153 | +0.100 | **+0.815** | −0.151 |

   El fold adversarial a 3s era el **3**, no el 4. Se arrastró un ranking de un
   horizonte a otro sin verificarlo. El fold 4 resultó un split neutro: la
   prueba fue menos exigente de lo previsto, aunque tampoco sesgada a favor.
4. **Hallazgo colateral que reencuadra los exp. 1-6:** esa misma tabla muestra
   que con el encoder genérico a 3s los folds 0, 1 y 4 YA daban negativo. El
   "+0.109 promedio" del exp. 6 estaba dominado por el fold 3 (+0.815). La
   conclusión "a 3s la escena no ayuda" nunca fue pareja entre splits.

**Conclusión:** con 2 folds medidos, la hipótesis queda **sostenida en un split
y ausente en otro**. Faltan los folds 1, 2 y 3 (~37h de GPU) para una respuesta
de 5 folds. La variable dominante del proyecto no es la arquitectura ni el
horizonte: es **qué escenas caen en el split**, con sólo 25 escenas disponibles.

**Reproducir:**
```
# 1) encoder de dominio del fold (12.5h)
conda run -n sapiens_gpu python tools/train.py \
    configs/sapiens_mae/lidar/config_rangeview_rect_fold4.py
# 2) decoder, 3s, 8 semillas  (o directamente: bash run_fold4_experiment.sh)
conda run -n sapiens_gpu python horizon_sweep.py \
    --enc work_dirs/rv_rect_fold4/epoch_1000.pth \
    --folds 4 --seeds 0 1 2 3 4 5 6 7 --horizons 3s \
    --archs wayformer baseline \
    --cache work_dirs/cache_fold4_domain --out work_dirs/horizon_fold4 \
    --epochs 100
```

---

## Experimento 11: CV completa de 5 folds — la escena NO ayuda (resultado definitivo)

**Fecha:** 2026-08-10. **Scripts:** `run_folds_123.sh`, `run_fold3_resume.sh`.

**Hipótesis:** los exp. 8-10 dejaron el efecto sostenido en el fold 0 (−20.4%,
t=−5.94) y ausente en el fold 4. Faltaban los folds 1, 2 y 3 para promediar
sobre los 5 y responder si el efecto es real o dependiente del split.

**Diseño:** un encoder MAE de dominio por fold (re-pre-entrenado desde cero solo
en las 20 escenas de train de ESE fold, 1000 ép, ~12.5h c/u), después decoder a
3s con 8 semillas, wayformer vs baseline. `--folds F` obligatorio (usar el
encoder de un fold en otro sería fuga).

**Resultado (diff pareada way−base; negativo = la escena ayuda):**

| fold | baseline | wayformer | diff | t | a favor | relativo |
|---|---|---|---|---|---|---|
| 0 | 0.912 | 0.726 | **−0.186 ± 0.089** | −5.94 | 8/8 | −20.4% |
| 1 | 1.252 | 1.190 | −0.061 ± 0.074 | −2.35 | 6/8 | −4.9% |
| 2 | 1.082 | 1.168 | +0.086 ± 0.084 | +2.89 | 1/8 | +7.9% |
| 3 | 1.424 | 1.993 | **+0.570 ± 0.130** | +12.40 | 0/8 | +40.0% |
| 4 | 1.816 | 1.792 | −0.024 ± 0.115 | −0.59 | 4/8 | −1.3% |

```
ENTRE FOLDS (n=5): +0.077 ± 0.292   t=0.589  gl=4   NO SIGNIFICATIVO
IC95% [-0.286, +0.439]  (incluye el 0)     3/5 folds a favor
sd ENTRE folds 0.292  vs  sd entre semillas 0.098   ->  3x
```

**Diagnóstico:**

1. **El efecto no sobrevive.** La media entre folds ni siquiera mantiene el
   signo: queda a favor del baseline. El −20.4% del fold 0 era una medición de
   un solo split.
2. **Validez del outlier verificada.** El fold 3 (+40%) es justo el encoder que
   se cortó el 08/08 (máquina suspendida, Xid 154) y se retomó con `--resume`.
   Terminó bien: loss final 0.3991 contra 0.389-0.401 de los otros cuatro. El
   outlier no es un encoder roto. Además el exp. 10 ya lo había marcado como el
   split adversarial a 3s (+0.815 con encoder genérico) — es un split
   consistentemente hostil, no ruido.
3. **LECCIÓN METODOLÓGICA (para el informe).** Muestrear bien la dimensión
   SEMILLA no protege de nada si no se muestrea el SPLIT. En el fold 0 había
   8 semillas, t=−5.94, p=0.0006, 8/8 a favor — y aun así el efecto era del
   split. Es la SEGUNDA vez que pasa: el 18/07 el 7.19 vs 7.85 de una escena se
   evaporó con la CV del 29/07. Con 25 escenas, la varianza dominante es qué
   escenas caen en cada lado del corte.

---

## Experimento 12: ¿El gate rescata los splits donde la escena daña?

**Fecha:** 2026-08-10. **Script:** `run_gated_folds_1234.sh`.

**Hipótesis:** el fracaso del fold 3 no es "la escena no ayuda" sino algo más
específico: el wayformer quedó 40% PEOR que el baseline, o sea que el decoder no
logró IGNORAR la escena cuando no servía. El gate (escalar aprendible
`tanh(scene_gate)` sobre la rama de cross-attn) es una válvula de amplitud que
puede cerrarse hasta 0 y degradar con gracia al baseline. En el fold 0 ya se
sabía que empata con el ungated (+0.033, t=1.16) => no costaría nada donde sí
hay señal.

**Diseño:** solo `wayformer_gated`, 3s, 8 semillas, folds 1-4 (el 0 ya estaba del
exp. 9). Features cacheadas => ~25 min por fold, ~2h total. Los baselines ya
estaban en los CSV y `horizon_sweep.py` aparea contra ellos.

**Resultado (ADE@3s):**

| fold | baseline | wayformer | gated | gate−base | gate−way |
|---|---|---|---|---|---|
| 0 | 0.912 | 0.726 | 0.759 | −0.152 | +0.033 |
| 1 | 1.252 | 1.190 | 1.244 | −0.008 | +0.053 |
| 2 | 1.082 | 1.168 | 1.245 | +0.163 | +0.077 |
| 3 | 1.424 | 1.993 | **2.122** | +0.699 (+49.1%) | +0.129 |
| 4 | 1.816 | 1.792 | 1.773 | −0.043 | −0.019 |

```
ENTRE FOLDS (n=5):
  gated - baseline : +0.132 ± 0.336  t=+0.87  no significativo  (3/5)
  gated - wayformer: +0.055 ± 0.055  t=+2.24  no significativo  (1/5)
```

**Diagnóstico:**

1. **La hipótesis se refuta.** El gate no rescata el fold 3: lo empeora, de
   +40.0% a +49.1%. Contra el ungated pierde en 4/5 folds. Ninguna variante con
   escena le gana al baseline promediando folds.
2. **MECANISMO DEL FALLO (lo valioso de este experimento).** `best_ep`=1 en 6/8
   semillas del fold 3, y en la época 1 el gate todavía vale ~0.497 — casi sin
   moverse de su init de 0.5. El early-stop congela el modelo ANTES de que la
   válvula se cierre, así que el checkpoint que se evalúa tiene la escena
   entrando a fuerza casi completa, justo en el split donde la escena es veneno.
   El gate aprende a cerrarse, pero demasiado tarde para que el early-stop lo
   aproveche. Explica por qué el gate sí servía en Fase 1: ahí el decoder era un
   MLP que entrenaba muchas más épocas.
   OJO al leer checkpoints: el `scene_gate` guardado es el del MEJOR checkpoint,
   NO el convergido. El valor interpretable es `gate_final`, que se imprime en el
   log (`train_decoder_mini.py` lo calcula aparte justo por esto).
3. **HALLAZGO COLATERAL — lo más reproducible del proyecto.** El gate converge
   al mismo valor en los 5 splits, desde 40 inicializaciones en 0.5:

   | fold | gate_final |
   |---|---|
   | 0 | 0.0917 ± 0.0051 |
   | 1 | 0.1059 ± 0.0045 |
   | 2 | 0.1026 ± 0.0075 |
   | 3 | **0.0772 ± 0.0079** |
   | 4 | 0.1016 ± 0.0087 |

   folds 1-4 juntos (n=32): **0.0968 ± 0.0135**. Mientras el ADE salta de −20% a
   +40% según el split, el peso óptimo aprendido para la escena replica en
   ~0.10 con dispersión de ±0.01. "El modelo decide solo que la escena debe
   entrar al ~10% de su fuerza" es la afirmación cuantitativa más sólida que
   produjo esta línea de trabajo.
4. **Correlación gate vs beneficio: NO establecida.** r=−0.734 entre `gate_final`
   y `way−base` sobre los 5 folds (el fold 3 cierra más el gate y es donde la
   escena más daña), pero con n=5 el |r| crítico al 5% es 0.878 y la relación la
   sostiene ese único punto: sacando el fold 3 se desarma. Se registra como
   observación, no como resultado.

**Reproducir:**
```
bash run_gated_folds_1234.sh
# o un fold suelto:
conda run -n sapiens_gpu python horizon_sweep.py \
    --enc work_dirs/rv_rect_fold3/epoch_1000.pth \
    --folds 3 --seeds 0 1 2 3 4 5 6 7 --horizons 3s --archs wayformer_gated \
    --cache work_dirs/cache_fold3_domain --out work_dirs/horizon_fold3 --epochs 100
```

---

## Experimento 13: ¿En qué se equivoca? Dirección vs magnitud, por fold y semilla

**Fecha:** 2026-08-18. **Script:** `sapiens/pretrain/angular_error_analysis.py`.
**Datos:** `work_dirs/angular/angular_results.csv`.

**Hipótesis:** los exp. 11-12 establecieron QUE el beneficio de la escena depende
del split, pero no POR QUÉ. El ADE mezcla dos errores distintos: apuntar mal
(dirección) y estimar mal cuánto avanza (magnitud). Separarlos debería
identificar el modo de fallo del fold 3 (+40%).

**Diseño:** por fold, se codifican una vez las 5 escenas retenidas con el encoder
de dominio de ESE fold y se reusan las features para las 8 semillas del decoder
(el costo lo domina el encoder). Métricas sobre objetos **móviles** (|despl. GT|
>= 1 m): en los parados la dirección no está definida, y son el 72-75% del total.
Se mide al último waypoint (3 s).

**Contexto de los splits:** el fold 3 tiene objetos que se desplazan el DOBLE que
los del fold 0 (5.55 m vs 2.72 m de media a 3 s; p90 19.2 vs 10.9), con la misma
proporción de parados. Es un split de autopista.

**Resultado — errores gruesos de dirección (>45°), 8 semillas por fold:**

| fold | baseline | wayformer | diff pareada | t | semillas peor |
|---|---|---|---|---|---|
| 0 | 11.7 ± 1.3 % | 11.1 ± 1.0 % | −0.7 ± 1.6 | −1.23 | 2/8 |
| 1 | 13.3 ± 1.3 % | 16.2 ± 1.8 % | **+2.9 ± 2.7** | 3.03 | 6/8 |
| 2 | 13.2 ± 0.3 % | 9.5 ± 4.5 % | **−3.7 ± 4.6** | −2.31 | 2/8 |
| 3 | 8.8 ± 1.4 % | 16.1 ± 3.3 % | **+7.3 ± 4.0** | **5.17** | **8/8** |
| 4 | 14.8 ± 0.3 % | 12.3 ± 2.5 % | **−2.5 ± 2.3** | −3.01 | 2/8 |

Mediana del error angular (caso típico): las diferencias son de ±1-2° y no
siguen el patrón de los errores gruesos — el efecto está en la COLA, no en el
caso típico.

Sesgo de magnitud (m, negativo = se queda corto), objetos móviles:

| fold | baseline | wayformer |
|---|---|---|
| 0 | −3.43 ± 0.65 | −3.19 ± 0.31 |
| 1 | −3.86 ± 0.49 | −4.67 ± 0.52 |
| 2 | −8.15 ± 0.03 | −7.42 ± 0.62 |
| 3 | −2.95 ± 0.26 | **−4.15 ± 0.96** |
| 4 | −4.66 ± 0.02 | −3.94 ± 0.78 |

**Diagnóstico:**

1. **Resultado sólido y ACOTADO:** en el fold 3 la rama de escena casi duplica
   los fallos direccionales gruesos (16.1% vs 8.8%), con 8/8 semillas y t=5.17.
   Combinado con que ahí los objetos se mueven el doble, cada fallo cuesta el
   doble de metros — consistente con el +40% de ADE del exp. 11.
2. **El patrón NO es general.** En los folds 2 y 4 la escena REDUCE los errores
   gruesos de forma significativa (t=−2.31 y −3.01). "La escena agrega riesgo de
   cola" no es una propiedad del método: depende del split, igual que el ADE.
3. **La explicación NO queda demostrada.** Correlación entre el exceso de errores
   gruesos y el daño en ADE: r=+0.687 sobre 5 folds; con n=5 el |r| crítico al 5%
   es 0.878. No significativa, y la sostiene sobre todo el fold 3. Se registra
   como observación, no como mecanismo probado.
4. **AVISO DE MÉTODO (tercera vez en el proyecto).** Este análisis se corrió
   primero con UNA semilla por fold y produjo dos afirmaciones que las 8 semillas
   desmintieron: (a) que en el fold 0 el wayformer tenía menos errores gruesos
   (11.8% vs 12.7% con 1 semilla -> empate, t=−1.23, con 8); (b) que calibraba
   mejor la magnitud (era un artefacto de promediar TODOS los objetos, con el
   ~73% parados aplastando la media; entre móviles el fold 3 va al revés).
   Ninguna medición de este pipeline es confiable con una sola semilla.

**Reproducir:**
```
conda run -n sapiens_gpu python angular_error_analysis.py             # 5 folds x 8 semillas
conda run -n sapiens_gpu python angular_error_analysis.py --folds 3 --seeds 0 1 2
```

---

## Simulación con el pipeline de dominio (visualización)

`export_decoder_mini_global.py` quedó parametrizado (`--enc-cfg`, `--enc-ckpt`,
`--dec`, `--dec-baseline`, `--n-wp`); los defaults conservan el comportamiento
viejo (encoder 10sw, 8 s). El horizonte se fija en el global `N_WP` del módulo
ANTES de construir samples y modelos, igual que hace `train_decoder()`.

Cobertura SIN FUGA de las 25 escenas: cada escena la predice el modelo del fold
que la retuvo (los 5 folds parten las 25 en grupos disjuntos), usando en cada
fold la semilla cuyo ADE cae más cerca de la media de las 8 — fold 0 s5, fold 1
s0, fold 2 s6, fold 3 s2, fold 4 s7. Genera `predictions_global_cv25.txt`
(10.966 puntos, 25/25 escenas) para el viewer C++.

```
./show_point_cloud --input waymo_clean_view      # OJO: _view, no waymo_clean
```
(`waymo_clean` tiene bins dispersos que rompen el `reshape(64,2650)` de la vista
superior; ver exp. de contrato de datos en CHECKLIST_CLAUDINE.md. El viewer lee
`predictions_global.txt` con nombre fijo desde el cwd.)

GIFs por escena en `work_dirs/sim_dominio_fold0_3s/` (split donde la escena
ayudaba) y `work_dirs/sim_dominio_fold3_3s/` (split adversarial).

---

## Experimento 14: barrido de gate CONGELADO — el control que faltaba desde el principio

**Fecha:** 2026-08-18/19. **Script:** `run_gate_sweep.sh` (arch `gatefix<v>` en
`train_decoder_mini.py`: `MiniWayformerGated` con `scene_gate` congelado en v).

**Hipótesis:** el exp. 12 mostró que el gate aprendido converge a 0.0968 ± 0.0135
en los 5 folds, pero eso solo dice DÓNDE aterriza el modelo, no si ese punto es
bueno. Congelando el gate en valores fijos se construye la curva ADE vs
cantidad-de-escena, que responde la pregunta de la tesis como curva y no como
sí/no. `gatefix0.0` (escena anulada) debía reproducir el baseline: control interno.

**Diseño:** 6 valores (0.0 / 0.05 / 0.1 / 0.2 / 0.5 / 0.99) × 8 semillas × folds
0 y 3 (los dos extremos: la escena "ayudaba" −20.4% / "dañaba" +40.0%). 96
corridas, ~10 h. Features cacheadas.

**Resultado 1 — la curva es PLANA:**

| gate | fold 0 ADE | vs gate=0 | fold 3 ADE | vs gate=0 |
|---|---|---|---|---|
| 0.0 | 0.782 ± 0.027 | — | 2.001 ± 0.156 | — |
| 0.05 | 0.773 ± 0.038 | −0.009 (t=−1.58) | 2.134 ± 0.037 | +0.133 (t=+2.13) |
| 0.1 | 0.776 ± 0.031 | −0.007 (t=−1.25) | 2.019 ± 0.181 | +0.018 (t=+0.23) |
| 0.2 | 0.775 ± 0.032 | −0.007 (t=−1.32) | 2.120 ± 0.115 | +0.119 (t=+1.89) |
| 0.5 | 0.747 ± 0.045 | −0.035 (t=−2.23) | 2.165 ± 0.020 | +0.163 (t=+2.76) |
| 0.99 | 0.761 ± 0.053 | −0.021 (t=−1.42) | 2.048 ± 0.142 | +0.046 (t=+0.53) |

De 0% a 99% de escena el ADE no se mueve fuera del ruido. Única celda
significativa: fold 3 gate 0.5 (t=+2.76) — pero (a) va en contra de la escena,
(b) es 1 de 10 comparaciones, exactamente lo que el azar predice al 5%, y (c) no
es monótona (0.5 significativo y 0.99 no). **No hay relación dosis-respuesta.**

**Resultado 2 — EL CONTROL FALLA, y ahí está el hallazgo:**

`gatefix0.0` NO reproduce el baseline. Y no por un bug: `MiniBaseline` procesa
cada objeto con un MLP **independiente**, mientras que el modelo gated conserva
**self-attention ENTRE objetos**, 2 capas y FFN. La comparación
"wayformer vs baseline" que sostuvo el proyecto entero nunca midió la escena:
medía escena + capacidad del decoder + interacción entre agentes, todo junto.

Descomposición (8 semillas):

| componente | fold 0 | fold 3 |
|---|---|---|
| **arquitectura** (gatefix0.0 vs baseline, SIN escena) | **−0.129** t=−9.19 8/8 | **+0.578** t=+9.15 0/8 |
| **escena** (gate aprendido vs gate 0, misma arq.) | −0.023 t=−1.80 (ns) | +0.121 t=+1.91 (ns) |
| total reportado históricamente | −0.186 t=−5.94 | +0.570 t=+12.40 |

La arquitectura explica el **69%** del efecto en el fold 0 y el **101%** en el
fold 3. Lo que queda para la escena no es significativo en ninguno de los dos.

**Diagnóstico:**

1. **La escena LiDAR no aporta nada**, con ningún encoder, ningún puente, ningún
   horizonte y ahora tampoco con ninguna DOSIS. Medido con el control correcto
   (misma arquitectura, escena apagada) y a lo largo de toda la curva.
2. **Lo que dependía del split nunca fue la escena: era la ARQUITECTURA.** Un
   decoder transformer de 2 capas con atención entre objetos, entrenado con 20
   escenas, gana 14% en un split y pierde 39% en otro. Eso explica de una vez el
   enigma que arrastraba el proyecto desde el exp. 8, y es coherente con que en
   el fold 3 `best_ep`=1 casi siempre: el modelo grande sobreajusta desde la
   primera época.
3. **Reencuadre de la tesis.** La pregunta deja de ser "¿la escena ayuda?" (no) y
   pasa a ser una crítica metodológica con evidencia: *con 25 escenas la
   capacidad del decoder domina cualquier efecto de las features de escena, y la
   comparación estándar "con LiDAR vs baseline simple" está confundida con
   capacidad del modelo*. Aplica a cualquier trabajo que compare así sin
   controlar arquitectura.
4. **Retractación:** el 18/08 se propuso el −0.129 del fold 0 como resultado
   positivo ("la atención entre objetos ayuda un 14%"). El fold 3 lo revierte
   (+0.578). También depende del split; no es un hallazgo.

**Reproducir:**
```
bash run_gate_sweep.sh
```

---

## Latencia de inferencia (puente a la etapa 2: el vehículo del LCAD)

**Script:** `latency_benchmark.py`. GPU: RTX 4060 Laptop.

| etapa | fp32 | autocast fp16 |
|---|---|---|
| encoder MAE (forward, 1 sweep) | 139.0 ms | **26.4 ms** (×5.3) |
| decoder (K slots) | 2.6 ms | — |
| **total cómputo** | **141.6 ms → 7.1 Hz** | **~29 ms → ~34 Hz** |

(+31.9 ms de lectura del .npy, que en el vehículo se reemplaza por la extracción
real del sweep.) A 10 Hz el presupuesto es 100 ms: en fp32 **no entra**, con
precisión mixta entra con 3× de margen y el error numérico es 0.046%.
El encoder es el **98%** del cómputo: cualquier optimización va ahí.

**TRAMPA ENCONTRADA AL VALIDAR fp16:** el encoder NO es determinista — dos
llamadas fp32 con la misma entrada dan 69.3% de error relativo elementwise. Con
`torch.manual_seed` fijado da 0.000%. Es una PERMUTACIÓN del mismo conjunto de
tokens (máx. 1.5e-5 en las sumas por token ordenadas), inocua porque la
cross-attention es invariante al orden de la memoria. Pero invalida cualquier
comparación elementwise de salidas del encoder sin fijar semilla: sin ese
control, fp16 parecía romper el modelo (70% de error) cuando en realidad su
error es 0.046%.

---

## Experimento 15: vuelta a Fase 1 (10 escenas) + dos bugs de datos

**Fechas:** 2026-08-23/26. **Scripts:** `run_fase1_seeds.sh`, `run_fase1_cv.sh`,
`run_rv_fold0.sh`, `run_rv_aug_fold0.sh`, `run_reeval_windows.sh`,
`run_reeval_sinclip.sh`, `run_noclip.sh`, `run_diagnostico.sh`.

**Contexto:** los exp. 1-14 usaron el pipeline de range-view a 25 escenas, donde
la escena es UN SOLO barrido. Fase 1 (10 escenas) tiene escena TEMPORAL —5
barridos— en las dos representaciones. Se volvió a esa escala con el control de
arquitectura del exp. 14 (`gate0`: mismo modelo, gate congelado en 0) y
evaluando en ÉPOCA FIJA (sin el sesgo de selección H1 de la auditoría).

### Resultado 1 — el efecto depende del tamaño del test

| representación | test | escena (gated − gate0) | t | |
|---|---|---|---|---|
| vóxeles | 51 | −0.170 (−9.5%) | −2.91 | sig |
| vóxeles | 319 | −0.049 (−3.2%) | −1.18 | ns |
| range-view | 51 | −0.273 (−14.2%) | −2.51 | sig |
| range-view | 319 | −0.060 (−3.7%) | −0.72 | ns |

Las dos representaciones convergen a **~−3%** al ampliar el test de 1 a 7
ventanas temporales por objeto. Que dos pipelines independientes aterricen en el
mismo valor es la señal más fuerte de que ése es el efecto real.

### Resultado 2 — vóxeles y range-view EMPATAN

ADE del mejor modelo: 1.45 vs 1.56 (con recorte), 13.19 vs 13.29 (sin recorte).
La ventaja histórica de los vóxeles (1.303 vs 1.685 en `RESULTADOS_ADE_FDE.md`)
era **augmentación de datos**, no representación: el config de range-view no
tenía `augment=True`. Al igualarla, la brecha desaparece. **Responde la Sec. 6
del plan de Claudine**, que pedía comparar representaciones.

### BUG A — el objetivo estaba recortado (crítico)

`trajectory_dataset.py` y `range_view.py` (cada uno con su propio `__getitem__`)
normalizan con media y desvío del **histórico** (5 puntos, ~0.5 s) y aplican ese
desvío también al **futuro** (3 s, decenas de metros), con clip a ±5 → ≈±2.5 m.

- **32%** de los valores del futuro se recortaban; del histórico, **0%**.
- Verificado: el objetivo real supera 5 en el **92%** de las muestras, pero el
  modelo predice >5 en solo el **27%** → subpredice el movimiento por ~4x, y no
  puede evitarlo: nunca vio un ejemplo mayor.
- **Todos los ADE de la documentación (~1.4 m) son ~10x optimistas.** Sin recorte
  son ~13 m.
- Las COMPARACIONES entre modelos siguen válidas: los tres comparten el objetivo.

Sin recorte, la escena pasa a ser significativa en ambas representaciones
(−0.417 t=−2.94 y −0.481 t=−2.92, 7/8 semillas) — pero el efecto RELATIVO sigue
siendo ~3%. No cambió el efecto, cambió la potencia para detectarlo.
El "castigo por capacidad" (gate0 peor que baseline) **desaparece** sin recorte:
era un artefacto del truncamiento.

### BUG B — la normalización es la causa raíz

Quitar el clip sin más deja valores de hasta 28 y el entrenamiento se vuelve
**inestable** (pérdida 36.8 → 10.9 → 16.8) y **11x más lento** (4.92 s/paso
contra 0.45). Probado con clip=50, que no recorta nada real: igual de lento → no
es el clip, es la magnitud de los valores.

FIX disponible: parámetros `clip_norm` (None desactiva) y `norm_scale` (escala
fija en metros) en ambos datasets. **El default preserva el comportamiento
anterior**, así que los checkpoints existentes siguen siendo válidos.
`run_diagnostico.sh` compara la normalización por histórico contra escala fija.

### Coordenadas: los datos están en el marco del EGO

`trajectory_dataset.py:124` transforma cada centro con `inv(pose)` de SU propio
frame. Un objeto parado "se mueve" a la velocidad del ego (58-66 km/h en las 2
escenas de validación). PERO medido: el ego aporta solo **3-22%** del
desplazamiento; en coordenadas del mundo los objetos igual se desplazan 26-30 m
en 3 s. La hipótesis de que "casi todo el movimiento es del ego" quedó
**refutada**: la tarea es genuinamente difícil. Ego vs mundo queda como decisión
de diseño abierta.

### Nota operativa — la GPU se cuelga al suspender

El 25/08 a las 18:31 la máquina se suspendió: `Xid 31` + `Xid 154` → *Node Reboot
Required*. El proceso quedó vivo pero congelado en la época 7, sin escribir al
log durante 14 h. Es la **segunda vez** (la primera, 08/08). Si un entrenamiento
largo deja de escribir, revisar `dmesg | grep -i xid` antes de asumir que avanza.

---

## Experimento 16: reentrenado SIN recorte y con escala fija — resultado final de Fase 1

**Fecha:** 2026-08-26. **Script:** `run_noclip.sh`, `run_diagnostico.sh`.
**Datos:** `work_dirs/noclip/noclip_results.csv`.

**Qué cambia respecto del exp. 15:** el objetivo ya no se recorta (`clip_norm=None`)
y se normaliza con **escala fija de 10 m** (`norm_scale=10.0`) en vez del desvío
del histórico. Es la primera vez que el modelo aprende la tarea real. Todo lo
demás idéntico: fold 0, 3 variantes, 8 semillas, época fija 100, test de 319.

**Resultado (8 semillas):**

| variante | ADE | FDE |
|---|---|---|
| baseline | 5.07 ± 1.15 m | 10.05 m |
| **gate0** (arquitectura, sin escena) | **4.57 ± 1.00 m** | **8.99 m** |
| gated (con escena) | 5.02 ± 0.76 m | 9.51 m |

| comparación | efecto | t | semillas | |
|---|---|---|---|---|
| **CAPACIDAD** (gate0 − baseline) | **−10.0%** | −5.35 | **8/8** | **significativo** |
| ESCENA (gated − gate0) | +9.8% | +1.60 | 1/8 | no significativo |

**Diagnóstico:**

1. **La atención entre objetos es el único componente con efecto demostrado:**
   −10%, unánime, t=−5.35. Es el resultado más firme del proyecto.
2. **La escena no aporta.** Apunta a perjudicar pero no alcanza significancia
   (1/8). Con 5 semillas daba +0.644 (t=+3.94); con 8, +0.450 (t=+1.60) y el
   desvío se duplicó. **Octava instancia del patrón** de este proyecto: un efecto
   que parece firme con pocas semillas y se desinfla al completarlas.
3. **El ADE cae de ~13.5 m a ~4.6 m.** Confirma que el techo estructural del
   recorte (el modelo no podía predecir >±2.5 m) explicaba la mayor parte del
   error medido en el exp. 15.
4. **La lentitud 11x del exp. 15 NO era numérica: era la GPU degradada.** Tras
   reiniciar, la misma config corre a 0.45 s/paso. El diagnóstico anterior estaba
   equivocado.
5. **El diagnóstico de normalización (`run_diagnostico.sh`)** comparó normalizar
   por histórico contra escala fija, 20 épocas: ambas estables, la fija reduce la
   pérdida 81% contra 69%. Se eligió la fija por eso y porque elimina el
   desajuste de calibrar con 0.5 s y aplicar a 3 s.

**CORRECCIÓN IMPORTANTE sobre el número de tokens.** Los experimentos de Fase 1
NO tienen 6785 tokens de escena: el camino de vóxeles usa **300** y el de
range-view **128**. Los 6785 son de la Fase 2 (`rv_rect_*`, 25 escenas). Por lo
tanto **la hipótesis "son demasiados tokens" no explica estos resultados**: las
representaciones ya son compactas y aun así la escena no aporta. Queda como única
explicación en pie el **objetivo del pre-entrenamiento** (reconstruir píxeles no
obliga a codificar movimiento ni geometría útil).

**Comparación con la literatura:** ver `papers/BEVTraj_Kong2025_arXiv-2509.10080.pdf`.
BEVTraj resuelve el mismo problema sin mapa e iguala o supera a métodos con mapa
HD (minADE₁₀ 0.905 vs 0.988 de Wayformer). Sus features BEV vienen de **BEVFusion,
supervisado por detección**, no de un MAE de reconstrucción — que es exactamente
la diferencia que nuestros experimentos señalan como determinante.

---

## Experimento 17: objetivo geométrico tipo GeoMAE — el encoder NO convergió

**Fecha:** 2026-08-27. **Script:** `run_geo.sh`. **Configs:** `geo_{mae,dec,base}_fold0.py`.
**Datos:** `work_dirs/geo/geo_results.csv`, log del encoder en `work_dirs/geo/mae.log`.

**Qué cambia respecto del exp. 16:** el encoder MAE se pre-entrena con objetivo
`centroide` (predecir el centroide de los puntos dentro de cada vóxel, normalizado
a [-1,1], NaN en los vacíos) en vez de `ocupacion`, y con 7 ventanas por escena
(`max_windows=7`, 8 → 56 muestras). La pérdida usa las tres ideas de
`pointmap_l1_loss.py` de Sapiens: máscara de validez, normalización por magnitud
media y L1 en vez de MSE. Motivación: GeoMAE (Tian 2023) critica explícitamente a
quienes "adoptan MAE directamente y solo predicen coordenadas u ocupación" y
reporta **+2,7 AP** solo por cambiar el objetivo.

**HALLAZGO PRINCIPAL — el pre-entrenamiento geométrico nunca aprendió.**

| encoder | pérdida inicial | pérdida final |
|---|---|---|
| ocupación (exp. 16) | 1,946 | **0,221** |
| **geométrico (exp. 17)** | 0,599 | **0,437**, oscilando |

El de ocupación baja un orden de magnitud; el geométrico se mueve dentro del ruido
a lo largo de 5000 pasos registrados. **Por lo tanto este experimento NO prueba la
idea de GeoMAE**: mide un decoder alimentado por un encoder cuyo pre-entrenamiento
falló en converger. La hipótesis del propio log — vóxel de 2 m demasiado grueso
para pedir precisión sub-vóxel, cuando GeoMAE usa vóxeles mucho más finos — sigue
siendo la explicación más plausible y **no está descartada**.

**Resultado del decoder (fold 0, 8 semillas, época fija 100, test de 319):**

| variante | ADE | FDE |
|---|---|---|
| baseline | 5,57 ± 1,43 m | 11,16 m |
| **gate0** (arquitectura, sin escena) | **4,92 ± 1,22 m** | **9,85 m** |
| gated (con escena, encoder geométrico) | 5,22 ± 0,80 m | 10,39 m |

| comparación | efecto | t | semillas | |
|---|---|---|---|---|
| **CAPACIDAD** (baseline − gate0) | **+13,1%** | +5,64 | **8/8** | **significativo** |
| ESCENA (gate0 − gated) | −5,6% | −1,05 | 7/8 a favor de gate0 | no significativo |
| geométrico vs ocupación (solo `gated`) | −2,6% | −1,38 | 6/8 | no significativo |

**Nota de consistencia (verificada, no es un bug):** las filas `baseline` y `gate0`
de `geo_results.csv` son **idénticas** a las de `noclip_results.csv`. Es correcto
por construcción: con el gate congelado en 0 la rama de escena se multiplica por
cero, así que el encoder no puede influir en el resultado. Sirve como comprobación
cruzada de que el montaje es determinista. **No** es una repetición del bug del
`--resume` (exp. 16): se verificó en los logs que el checkpoint geométrico
`work_dirs/geo/mae_encoder_fold0.pth` sí se carga.

**Lectura honesta:** la mejora del 2,6% no es significativa y viene de un encoder
roto. El objetivo geométrico **queda como condición NO probada**, no como
condición refutada.

---

## Experimento 18: descongelar el encoder (réplica de JointMotion) — sin efecto

**Fecha:** 2026-08-28/29. **Rama:** `encoder/jointmotion-finetune`.
**Script:** `run_jointmotion.sh`. **Datos:** `work_dirs/jm/jm_results.csv`.

**Motivación.** JointMotion (Wagner 2024) dice en su sección de fine-tuning:
*"We initialize the modality-specific encoders with the learned weights from
pre-training and do not freeze any weights during fine-tuning."* Nosotros
congelábamos los 302,6 M (0 parámetros entrenables): el pre-entrenamiento era un
**extractor fijo**, no una **inicialización**. Ellos obtienen −3% a −12% de FDE.
Era la última diferencia metodológica sin probar.

**Por qué parcial y no total.** Descongelar los 302 M enteros da OOM en 8 GB con
lote 16, y bajar el lote a 4 **degrada el modelo por sí solo** (medido el 28/08:
ADE 4,84 → 8,29 y el gate colapsa a ~0). Descongelar solo la cola entra en memoria
con lote 16 y deja comparable el resultado contra todo lo demás. Implementado con
el parámetro `finetune_blocks` en `TrajectoryModelWithAttention`.

**Diseño:** 3 variantes × 8 semillas, fold 0, encoder geométrico del exp. 17,
lote 16, época fija 100, test de 319. Todo idéntico salvo cuántos bloques se
descongelan.

| variante | entrenables en el encoder | ADE | FDE | gate final |
|---|---|---|---|---|
| ft0 — congelado (control) | 0 | 5,22 ± 0,80 m | 10,39 m | 0,072 |
| ft2 — últimos 2 bloques | 25,2 M | 5,22 ± 0,78 m | 10,28 m | 0,068 |
| ft4 — últimos 4 bloques | 50,4 M | 5,17 ± 0,88 m | 10,28 m | 0,077 |

| comparación | efecto | t | semillas |
|---|---|---|---|
| ft2 − ft0 | +0,003 ± 0,152 (+0,1%) | +0,05 | 4/8 |
| ft4 − ft0 | −0,052 ± 0,283 (−1,0%) | −0,52 | 4/8 |

**Diagnóstico:**

1. **Descongelar no tiene ningún efecto.** Diferencias de milésimas, sin dirección
   consistente, y las semillas se reparten mitad y mitad — la firma de un efecto
   nulo, no de uno pequeño.
2. **El control valida el montaje:** `ft0` reproduce el `gated` del exp. 17 exacto
   (mismas corridas reutilizadas), y el encoder geométrico se carga de verdad
   (`Load checkpoint from ./work_dirs/geo/mae_encoder_fold0.pth` presente en los
   24 logs). Verificado explícitamente por el antecedente del bug del `--resume`.
3. **Queda descartada la hipótesis del congelamiento** como explicación del
   resultado negativo del proyecto.

---

## Convención de promediado — LEER ANTES DE CITAR UN ADE

Las dos escenas de validación del fold 0 tienen **200 y 119 objetos**. Promediarlas
con peso igual o ponderadas por número de objetos da números distintos:

| variante | media simple de escenas | ponderada por objetos |
|---|---|---|
| ft0 | 4,836 | **5,217** |
| ft2 | 4,845 | **5,220** |
| ft4 | 4,804 | **5,165** |

Un 7% de diferencia. Los números reportados en la reunión del 26/08 y en los
experimentos 15-17 usan la **media simple**, que le da el mismo peso a la escena
fácil que a la difícil. La **ponderada es la defendible** en la disertación.
Las comparaciones entre modelos no cambian (todos comparten el promedio), pero el
ADE absoluto sí. **Fijar una convención y declararla antes de cada tabla.**

---

## Estado al 2026-08-30 — las condiciones de la literatura, probadas

Cuatro condiciones separan nuestro montaje de los trabajos que sí obtienen ganancia
con auto-supervisión. Estado de cada una:

| condición | qué probamos | resultado |
|---|---|---|
| escala del pre-entrenamiento | 8 → 56 muestras (`max_windows`) | sin cambio |
| representación | vóxeles vs range-view | empatan (exp. 15) |
| encoder descongelado | 2 y 4 bloques, 8 semillas | **sin efecto** (exp. 18) |
| **objetivo geométrico** | centroide — **el encoder no convergió** | **NO PROBADA** (exp. 17) |

**Lo que queda en pie:** la atención entre objetos aporta **−10% a −13% de ADE,
8/8 semillas, t≈−5,4**, replicado en los experimentos 16 y 17. Es el resultado más
firme del proyecto. La escena LiDAR auto-supervisada no aporta a esta escala.

**La única vía sin agotar** es hacer que el objetivo geométrico converja de verdad
(vóxel más fino), y el objetivo CME de JointMotion, que conecta explícitamente el
movimiento con el entorno.


## Estado al 2026-08-27 — tras la reunión con Claudine

**La reunión salió bien.** Los resultados fueron aceptados. Surgieron tres
preguntas, respondidas abajo, y quedó un bug reportado.

### Respuestas a las preguntas de la reunión

**¿Entrenamos el decoder?** Sí, es lo único que se entrena. Verificado:
encoder 302,6 M parámetros con **0 entrenables** (`freeze_encoder=True`);
decoder 4,89 M, todos entrenables. El encoder se pre-entrena aparte y se congela.

**¿Por qué las escenas son tan cortas?** Límite del dataset, no decisión nuestra:
11 frames de LiDAR (1,1 s) contra 91 de etiquetas. Cita del paper:
*"We only release the first 1 second LiDAR data for each scene. This helps reduce
the 87.9% size of the raw LiDAR data."* **Escenas más largas requieren cambiar de
dataset** — nuScenes (20 s con LiDAR continuo, el que usa BEVTraj) o Argoverse 2.

**Visualización de un solo objeto:** `viz_un_auto.py` genera `viz_tres_autos.png`
(tres objetos por percentil de desplazamiento) y `viz_un_auto.png` (caso extremo).
Muestran que las trayectorias son casi rectas y que **ambos modelos se quedan
cortos**, cada vez más cuanto más rápido va el objeto.

### BUG PENDIENTE (reportado 27/08, sin diagnosticar)

En el simulador, **la imagen superior (range-view) con los objetos marcados en
rojo aparecía dañada** durante la reunión. Revisar la proyección de bboxes a la
range view en `export_decoder_mini_global.py` (método de Gabriel, calibrado con
`waymo_clean/beam_inclinations.npy`).

### Hallazgo: el encoder se pre-entrenaba con OCHO muestras

`LidarSequenceDataset.load_data_list` devolvía **un ítem por escena**: con 8
escenas de train, el MAE veía 8 muestras repetidas 1000 épocas, para un ViT de
302 M. Ver commit 99a4239.

> **CORRECCIÓN 02/09.** Este párrafo decía *"Corregido con `max_windows`
> (8 → 56)"*. Es falso para los encoders que de verdad se usaron. El arreglo
> llegó al **dataset** y a **`geo_mae_fold0.py`**, pero `max_windows` sigue en
> **1 por defecto** y ningún `f1cv_mae_fold*.py` lo declara — esos configs los
> había generado `run_fase1_cv.sh` antes del arreglo. Los cinco encoders de la
> CV de 5 folds (`work_dirs/f1cv/mae_encoder_fold*.pth`), o sea los que
> sostienen los experimentos 19 y 20, se pre-entrenaron con **8 muestras cada
> uno**. Verificable en una línea: los cinco logs terminan en `[1000][8/8]`.
>
> Un arreglo aplicado al código pero no a los configs que se corren es
> indistinguible de no haberlo aplicado. Lo que sí quedó medido está en el
> experimento 21, abajo: pese a las 8 muestras, los encoders generalizan.

### Corrección: el conteo de tokens

Los experimentos de Fase 1 **no** usan 6785 tokens de escena: vóxeles usa **300**
y range-view **128**. Los 6785 son de Fase 2 (`rv_rect_*`). Por lo tanto *"son
demasiados tokens"* no explica estos resultados — las representaciones ya son
compactas y aun así la escena no aporta.

### Sobre Sapiens (verificado)

El paper es arXiv:2408.12569 (Khirodkar et al., Meta 2024): modelo fundacional de
**visión humana**, 300 M de imágenes de personas. **No existe literatura de
Sapiens aplicado a LiDAR** (0 resultados en arXiv). Y **no usamos sus pesos** —
entrenamos desde cero. En la práctica, "adaptar Sapiens" es "usar un ViT de
302 M": el activo que hace valioso a un modelo fundacional no está en juego.
Repos completos clonados en `~/referencias/{sapiens_full,sapiens2}`.

### Papers de referencia en `papers/`

| paper | aporte |
|---|---|
| WOMD-LiDAR (Chen 2023) | Wayformer+LiDAR: minADE 1,10 → **1,09**. El +2% es en **mAP, no ADE** |
| BEVTraj (Kong 2025) | map-free iguala a métodos con mapa HD; su BEV es **supervisada por detección** |
| GeoMAE (Tian 2023) | **+2,7 AP** cambiando el objetivo a targets geométricos; funciona sin datos extra |
| JointMotion (Wagner 2024) | auto-supervisión para movimiento, pero **0 menciones de LiDAR** (polilíneas) |

### Experimento 17: objetivo geométrico — CERRADO, ver arriba

Encoder del fold 0 con objetivo `centroide` + 7 ventanas (56 muestras), pérdida
con máscara de validez, normalización por magnitud y L1 (ideas de
`pointmap_l1_loss.py` de Sapiens). Configs `geo_*_fold0.py`, script `run_geo.sh`.

**La señal de alerta se confirmó:** la pérdida del encoder nunca bajó
(0,599 → 0,437 en 5000 pasos, oscilando), contra el modo anterior que caía
1,946 → 0,221. El objetivo geométrico **no llegó a probarse**. Resultado
completo en la sección "Experimento 17" más arriba.

---

## Experimento 19: la validación cruzada de 5 folds — el primer resultado defendible

**Fechas:** lanzado 30/08 09:06, terminado 31/08 04:33 (19 h 30).
**Script:** `run_noclip_cv.sh`. **Datos:** `work_dirs/noclipcv/noclipcv_results.csv`.
**Diseño:** 5 folds × 3 variantes × 8 semillas = **120 corridas**, 240 filas, cero
duplicados. Encoder MAE re-pre-entrenado desde cero por fold (antifuga verificada).
Época fija 100, evaluación `--sin-clip --eval-windows 7`.

**Por qué se corrió.** Al estrenar `agregar_resultados.py` el 30/08, su aviso
automático de "un solo fold" saltó en TODOS los CSV del proyecto: los experimentos
15 a 18 descansaban sobre un único split de escenas. La CV de 5 folds nunca se había
corrido. Además, ese mismo día la revisión de código encontró 33 errores, entre
ellos que **la escena LiDAR estaba desalineada en el tiempo para el 43% de los
objetos** — así que ningún número anterior era comparable de todas formas.

**Resultado entre folds (n=5 folds, no corridas):**

| efecto | valor | t | p | folds a favor |
|---|---|---|---|---|
| **capacidad** (gate0 − baseline) | −0,207 ± 0,219 | −2,11 | 0,102 | **5/5** |
| **escena** (gated − gate0) | +0,723 ± 0,529 | +3,05 | **0,038** | 0/5 |

Por fold, el efecto de la escena: +39,9%, +40,6%, +11,1%, +39,0%, +3,5%.

**Este fue el primer efecto del proyecto que NO se evaporó al pasar de un fold a
cinco.** Pero ver el experimento 20: la mayor parte era un artefacto nuestro.

---

## Experimento 20: el arranque del gate — la mitad del "daño" era diseño experimental

**Fechas:** 31/08, 06:12 → 14:29 (8 h 17). **Script:** `run_gateinit.sh`.
**Datos:** `work_dirs/gateinit/gateinit_results.csv` (5 folds × 8 semillas).

**El diagnóstico.** Leyendo las curvas de pérdida de entrenamiento del fold 0:

| época | gate0 | gated (init 0,5) | gated005 (init 0,05) |
|---|---|---|---|
| 5 | 0,0685 | **0,2552** | 0,0825 |
| 10 | 0,0618 | **0,2542** | 0,0618 |
| 100 | 0,0336 | 0,0489 | 0,0361 |

`gated` pasa sus primeras ~10 épocas **paralizado**: la pérdida clavada en el valor
inicial mientras el control ya bajó un orden de magnitud. Y el efecto acumulado es
peor que un retraso: la pérdida que `gated` alcanza en la época 100, `gate0` ya la
tenía en la **época 41**. Con evaluación en época fija, la variante experimental
competía con **menos de la mitad del entrenamiento efectivo** de su control.

**La causa.** `gate_init=0.5` mete la rama de escena a media amplitud desde el primer
paso, cuando todavía no aprendió nada: el decoder recibe 15 números útiles (la
historia) más 64 de ruido, y gasta épocas aprendiendo a callarlos.

**Los tres regímenes del gate:**

| arranque | ¿la rama aprende? | ¿ahoga al decoder? |
|---|---|---|
| 0,0 | **no** — el gradiente a la rama va multiplicado por `tanh(g)=0` | no |
| **0,05** | **sí** | **no** |
| 0,5 | sí | **sí** — 59 épocas equivalentes |

Con `g=0` el gradiente al **escalar** no es cero, pero es proporcional a una
proyección que nunca aprende: ruido. El gate hace una caminata al azar alrededor del
cero — de ahí el `−0,0047` que registró el commit 8692268. `0,05` rompe la simetría
sin ahogar. Es la receta de LayerScale, y el ángulo teórico está en ReZero
(Bachlechner 2020, `papers/`), aunque su argumento no transfiere limpio: ReZero SUMA
a una identidad y nuestra rama CONCATENA.

**Resultado: el efecto de la escena, con el mismo control.**

| variante | efecto entre folds | t | p | folds a favor |
|---|---|---|---|---|
| `gated` (init 0,5) | +0,723 ± 0,529 | +3,05 | **0,038** | 0/5 |
| **`gated005`** (init 0,05) | **+0,276 ± 0,335** | **+1,84** | **0,139** | 0/5 |

Por fold: +22,4%, +13,5%, +1,2%, +10,6%, +0,1% — **baja en los cinco, sin excepción.**

**El gate converge a 0,0042.** Arrancando en 0,05, el modelo lo cierra casi del todo.
Cuando se le da un arranque justo, **decide por sí mismo que la escena no sirve**.

---

## Estado al 2026-09-01 — la conclusión que se sostiene

**Lo que se puede afirmar:**

1. **La escena LiDAR auto-supervisada NO aporta** a esta escala. Cinco folds, ocho
   semillas, escena temporalmente alineada, control de arquitectura, y sin el
   artefacto del gate: **0 de 5 folds a favor**, y el gate aprendido cierra a 0,004.
2. **No se puede afirmar que perjudique.** Con el arranque corregido, +0,276 con
   p=0,139. La formulación honesta es *"no aporta, y posiblemente estorba un poco"*.
3. **La capacidad (atención entre objetos) tampoco alcanza significancia**: −0,207,
   **5/5 folds en la misma dirección**, pero p=0,102. Dirección consistente, potencia
   insuficiente con n=5 folds.

**El hallazgo metodológico, que vale por sí solo:** la inicialización del gate
residual le costaba a la variante experimental **más de la mitad de su
entrenamiento**, inflando el efecto medido de +0,276 a +0,723 — casi el triple. Un
resultado con p=0,038 pasó a p=0,139 al corregir un detalle de inicialización.

**Retractación registrada (la décima del proyecto):** el resultado del experimento 19
—"la escena perjudica, p=0,038"— se retractó en menos de 24 horas, por medición
propia. No sobrevive al control del arranque del gate.

---

## Experimento 21: ¿los encoders memorizaron? — diagnóstico de generalización del MAE

**Fecha:** 2026-09-02 · **Script:** `sapiens/pretrain/diagnostico_encoder_mae.py`
**No entrena nada:** lee los cinco `epoch_1000.pth` existentes y hace forwards.

### Por qué

Los experimentos 19 y 20 concluyen que la escena no aporta. Esa conclusión solo
vale si la escena que ve el decoder significa algo. Y los cinco encoders se
pre-entrenaron con **8 muestras** (ver la corrección del 02/09 arriba) durante
1000 épocas, sin ningún `val_dataloader`: la caída 1,29 → 0,019-0,087 según el
fold (mínimo de cada uno de los 5 logs) es
pérdida de **entrenamiento sobre esas ocho ventanas**. Nunca se midió la
reconstrucción fuera de ellas.

Si el encoder hubiera memorizado, en validación entregaría ruido, y que el gate
cierre a 0,004 dejaría de ser *"el modelo descarta la escena"* para ser *"el
modelo descarta ruido"*: la condición quedaría **sin probar**, no refutada — el
mismo error que se corrigió en el experimento 17.

### Diseño

Tres poblaciones que separan las dos hipótesis, por fold:

| población | qué es | n por fold |
|---|---|---|
| `train_vistas` | las 8 ventanas exactas que el MAE optimizó | 8 |
| `train_nuevas` | las otras 6 ventanas/escena de las MISMAS escenas (t0≥1) | 48 |
| `val` | las 7 ventanas de cada una de las 2 escenas RETENIDAS del fold | 14 |

Y dos referencias sin las cuales los números no significan nada: el **mismo
modelo sin entrenar** (el "no aprendió nada") y la pérdida **trivial** de
predecir 0 en todo (con ocupación en {0,1}, la fracción de vóxeles ocupados).

**Máscaras pareadas.** La pérdida MAE depende de qué vóxeles se enmascaren: se
fija la semilla justo antes de cada forward, así que la máscara nº k es idéntica
en las tres poblaciones y en los dos modelos. 8 máscaras promediadas por muestra.

**Trampa encontrada al escribir el script.** `MAEViT4D.forward` solo enmascara
bajo `if self.training`; en `eval()` devuelve `mask=ceros` y la pérdida sale
`0/1e-6 = 0` para cualquier modelo. La primera corrida dio 0.0000 idéntico en
entrenado y aleatorio. Hay que medir con el modelo en `train()` — verificando
antes que no haya dropout activo ni BatchNorm, que es lo que hace
`verificar_modo_train`.

### Resultado — media sobre los 5 folds

| población | entrenado | aleatorio | trivial | vs trivial |
|---|---|---|---|---|
| `train_vistas` | 0,0691 ± 0,0224 | 1,1584 | 0,3408 | **79,7 %** mejor |
| `train_nuevas` | 0,1170 ± 0,0082 | 1,1632 | 0,3380 | **65,4 %** mejor |
| `val` | 0,1913 ± 0,0441 | 1,1625 | 0,3384 | **43,5 %** mejor |

`val` peor que `train_nuevas` en **5/5 folds**. Razones: val/train_nuevas =
1,63× · val/train_vistas = 2,77×.

### Lectura

**Los encoders NO memorizaron.** En escenas que nunca vieron reconstruyen 6×
mejor que sin entrenar y 43,5 % mejor que la predicción trivial. La hipótesis de
que el gate se cierra porque le entregan ruido **queda refutada**: el resultado
negativo de los experimentos 19 y 20 se sostiene.

**Hay sobreajuste, pero es un gradiente y no un derrumbe** (1,63×, consistente en
5/5 folds).

**Dónde está la brecha, que es lo accionable.** El salto grande no está entre las
8 ventanas vistas y las 48 nuevas de las mismas escenas (0,069 → 0,117), sino al
cruzar a **escenas** distintas (0,117 → 0,191). Consecuencia directa:

> Arreglar `max_windows=1` daría poco: añadiría ventanas de escenas que el
> encoder ya maneja bien. Lo que no generaliza es el cruce entre escenas, y eso
> solo se arregla con **más escenas**. `waymo_clean` tiene **25 escenas con
> datos de 492 directorios**: el cuello de botella es la extracción.

**Lo que este experimento NO responde.** Reconstruir ocupación bien no implica
que las features sirvan para trayectorias — es la crítica de GeoMAE
(arXiv:2305.08808) ya anotada en `mae_head_4d.py`. Que el encoder generalice
descarta la explicación *"es ruido"*, no la explicación *"el objetivo de
pre-entrenamiento es el equivocado"*.

---

## Experimento 22: la historia completa (1,1 s) — no mejora la predicción

**Fecha:** 2026-09-02 · **Script:** `sapiens/pretrain/run_hist11.sh`
**CSV:** `work_dirs/hist11/hist11_results.csv` · 5 folds × 8 semillas × 2 brazos

### Por qué

WOMD-LiDAR entrega 11 frames de LiDAR por escena = 1,1 s, que es exactamente la
ventana de historia que define el benchmark de Waymo (`docs/ESTUDIO_WAYFORMER.md`).
Todos los configs de Fase 1 usan `history_len=5`: veníamos prediciendo 3 s de
futuro con 0,5 s de pasado, tirando más de la mitad del contexto que ya estaba en
el disco.

Medido antes de correr nada: pasar de 5 a 11 cuesta el **13 %** de las ventanas de
entrenamiento (236 → 206 en el fold 0), no el 7× que uno supondría — el train ya
tomaba una sola ventana por objeto.

### El control de población, sin el cual esto no significaba nada

Con `history_len=11` solo entra la ventana `f0=0`, así que la evaluación por
defecto pasaba de 183 ventanas / 29 objetos a 24 / 24: **poblaciones distintas,
ADE incomparables**. Por eso se agregó `--poblacion-hist` a `eval_fase1_seeds.py`,
y los DOS brazos se evalúan con `--poblacion-hist 11`: se quedan las ventanas cuyo
futuro arranca en el frame absoluto 11 y cuyo objeto existe también con historia
11. El de `h=5` usa su ventana `f0=6`, el de `h=11` su `f0=0`, y **ambos predicen
los frames 11..40 de los mismos objetos**. Verificado: la población de h=11 es
subconjunto de la de h=5 en los 5 folds (24, 59, 40, 25, 82 objetos).

Por eso `base5` se RE-EVALÚA acá y no se leen las filas de `work_dirs/noclipcv`:
esas son de la población vieja. La diferencia no es cosmética — el mismo
checkpoint da 3,84 en la población nueva y 4,03 en la vieja.

Se descartó además que el cambio de origen falseara la comparación: con
`norm_scale=10` fijo, la magnitud media del objetivo normalizado es 0,7058 (h=5)
contra 0,7436 (h=11), y el máximo de h=11 es *menor*. No es la normalización.

### Resultado

| | ADE medio (n=40 corridas) |
|---|---|
| `base11` (1,1 s de historia) | 3,832 ± 1,286 |
| `base5` (0,5 s) | 3,043 ± 1,133 |

Por fold, pareado por (fold, semilla):

| fold | efecto | rel | p | a favor |
|---|---|---|---|---|
| 0 | +1,470 | +39,7 % | 0,0064* | 0/8 |
| 1 | +2,069 | +97,0 % | 0,0000* | 0/8 |
| 2 | +0,079 | +1,7 % | 0,8244 | 4/8 |
| 3 | −0,173 | −8,1 % | 0,3505 | 5/8 |
| 4 | +0,497 | +19,4 % | 0,0611 | 1/8 |

**Entre folds (n = 5 folds): +0,788 ± 0,951 · t=+1,85 · p=0,1373 · 1/5 folds.**

### Lectura

**La historia completa NO mejora la predicción.** Va peor en promedio y solo 1 de
5 folds la favorece.

**Pero no se puede afirmar que perjudique:** p=0,137 entre folds. Misma forma que
el resultado de la escena.

**Y la advertencia de siempre, otra vez:** dos folds dan p<0,01 *dentro del fold*
—el fold 1 con p=0,0000 y +97 %— y el test entre folds queda en p=0,137. Es la
undécima vez que este proyecto ve una significancia intra-fold evaporarse al
promediar. La varianza entre folds sigue dominando.

**El mecanismo, medido y no supuesto:** `base11` alcanza MENOS pérdida de
entrenamiento que `base5` (0,0410 ± 0,0149 contra 0,0471 ± 0,0110, n=40 corridas
cada uno) y generaliza peor. Es sobreajuste: con 206 ventanas de entrenamiento,
darle al MLP 33 entradas en vez de 15 le alcanza para memorizar.

### Lo que esto cierra

Tres resultados independientes apuntan al mismo lugar:

| experimento | resultado |
|---|---|
| 19-20: la escena LiDAR | no aporta (0/5 folds) |
| 19: la capacidad (atención) | −0,207, 5/5 folds, pero p=0,102 |
| **22: más contexto temporal** | **no aporta (1/5 folds), y sobreajusta** |

No falta información por muestra: **faltan muestras**. 236 ventanas de
entrenamiento desde 8 escenas, y un encoder MAE pre-entrenado con 8. El
experimento 21 ya había localizado la brecha del encoder en cruzar entre
*escenas*. El siguiente paso no es otra variante de arquitectura.

---

## Experimento 23: la prueba de 10 sweeps de Claudine, medida bien

**Fecha:** 2026-09-02 · **Config:** `configs/sapiens_mae/lidar/rect_overfit10_val.py`
**Curva:** `curva_overfit10.py` · **CSV:** `work_dirs/rect_ov10_fino/curva_overfit10.csv`

Peldaño **10 sweeps** de la escalera de Claudine (ítems 5 y 11 del checklist).
No se cambió ni la arquitectura ni los datos: se arregló **cómo se mide**.

### Tres cosas que estaban mal, todas verificadas

**1. Train y evaluación estaban en dominios distintos.** `range_png_rect/train/`
tenía las 10 imágenes en **2650×64** (filas nativas) mientras `val`, `unseen`,
`train100` y los cinco `fold*_train` están en **2650×1024**. El pipeline reescala
a 1024 con **bicubic**; `make_rect_png_scenes.py` escribe las de 1024 con
**INTER_NEAREST**. El modelo entrenaba sobre gradientes suavizados y se evaluaba
sobre bloques duros de 16 filas repetidas. Explica por qué el de 100 sweeps daba
mejor (3,16): su `train100` ya estaba en 1024.

**2. El config tomaría hoy 612 imágenes, no 10.** `data_root` apunta a la RAÍZ de
`range_png_rect/` y `CustomDataset` recorre subdirectorios. El 28/06 ahí solo
estaba `train/` (el log de la corrida original dice `[6000][5/5]` con
`batch_size=2` = 10 imágenes, correcto). Después aparecieron `train100`, `val`,
`unseen` y cinco `fold*_train`. Verificado construyendo el dataset: **612**, e
incluiría `val/` y `unseen/` — fuga directa en el split de evaluación.

**3. El retenido era n=1.** Una imagen en `val`, una en `unseen`. Se reemplazó por
`ov10_val_intra` (2a81 sweep 9, el único sweep de la escena de train que el modelo
no vio) y `ov10_val_escenas` (**5 escenas nunca vistas × 11 sweeps = 55 imágenes**).

**Identificación de las 10 imágenes.** Comparando píxel a píxel contra los `.npy`
de `range_files`: escena `2a81f5233075e987`, sweeps **{0,1,2,3,4,5,6,7,8,10}** —
diferencia media 0,20 sobre 255. Falta el 9 porque el generador ordenó los `.npy`
como **strings**: 0, 1, 10, 2, 3... El ítem 5 decía "10 sweeps de la escena 2a81",
correcto, pero no son los sweeps 0-9.

### La curva de generalización

Antes no existía: el config original tenía `max_keep_ckpts=2` con `interval=500`,
así que en disco quedaban solo `epoch_5500` y `epoch_6000`. Con `interval=25`:

| época | train | val_intra | val_escenas |
|---|---|---|---|
| sin entrenar | 3,059 | 3,043 | 2,684 |
| 25 | 0,719 | 0,805 | 1,848 |
| **50** | 0,665 | 0,749 | **1,830** |
| 75 | 0,642 | 0,719 | 1,853 |
| 100 | 0,625 | 0,706 | 1,919 |
| 250 | 0,564 | 0,646 | 1,991 |
| 400 | 0,524 | 0,646 | 2,052 |

**El óptimo está en la época 50 de 6000.** Pareado por imagen sobre las 55
retenidas, 4 máscaras:

```
ép 50 vs ép 400:  +0,2301 ± 0,1557   t=+10,96   55/55 imágenes
ép 50 vs ép  75:  +0,0272 ± 0,0279   t= +7,25   55/55 imágenes
ép 50 vs ép  25:  +0,0142 ± 0,0419   t= +2,51   30/55 imágenes
```

O sea: **una meseta entre las épocas 25 y 50, con caída abrupta después.** La 50 no
se distingue de la 25 (30/55 es cara o cruz) pero sí de la 75 y de la 400.

### Resultado

| medición | mejora sobre el modelo sin entrenar |
|---|---|
| ítem 11 del checklist (n=1, época 6000) | **+3,7 %** |
| época 2000, retenido de 55 imágenes | +13,7 % |
| **época 50, retenido de 55 imágenes** | **+31,8 %** |

**El encoder de 10 sweeps es ~9× mejor de lo que el checklist le acreditaba.** No
se mejoró el encoder: es el mismo, con los mismos datos. Lo que estaba mal era la
medición — una imagen, la época equivocada, y otro dominio.

### El hallazgo que vale por sí solo

`val_intra` y `val_escenas` **van en direcciones opuestas**. El sweep retenido de la
misma escena mejora hasta la época ~300 (0,805 → 0,640) mientras el retenido de
otras escenas empeora desde la 50. El modelo sigue aprendiendo **su escena** y eso le
cuesta la **transferencia entre escenas**.

Es la misma estructura del experimento 21 en el encoder de vóxeles
(0,069 → 0,117 → 0,191), ahora con la curva temporal que allá no teníamos.

### Lo que abre

Los cinco encoders de la CV de Fase 1 se entrenaron **1000 épocas sin
`val_dataloader`** y se usó el **último** checkpoint. Si les pasa lo mismo, están
en la zona degradada. Es comprobable barato: en disco quedaron las épocas **600 y
800** además de la 1000, y `diagnostico_encoder_mae.py` ya sabe medirlo.

### Adenda al 23: ¿les pasa lo mismo a los encoders de Fase 1? No.

La curva de range-view abrió la sospecha de que los cinco encoders de vóxeles de la
CV —entrenados 1000 épocas sin `val_dataloader`, usando el último checkpoint—
estuvieran degradados. Medido con `diagnostico_encoder_mae.py --epoch epoch_{600,800,1000}.pth`,
reconstrucción en las escenas RETENIDAS de cada fold:

| fold | ép 600 | ép 800 | ép 1000 |
|---|---|---|---|
| 0 | 0,1937 | 0,1902 | 0,1939 |
| 1 | 0,2591 | 0,2592 | **0,2445** |
| 2 | 0,1356 | 0,1313 | **0,1261** |
| 3 | 0,1726 | 0,1632 | 0,1632 |
| 4 | **0,1860** | 0,1978 | 0,2060 |
| **media** | 0,1894 | 0,1883 | **0,1867** |

```
ép600 vs ép1000:  -0,0027 ± 0,0138   t=-0,43   2/5 folds
ép800 vs ép1000:  -0,0016 ± 0,0088   t=-0,41   2/5 folds
```

**No hay degradación**, y si algo la última época es marginalmente la mejor.
**Los ADE de los experimentos 19-22 no están comprometidos por esto.**

**Lo que esta comprobación NO prueba.** Solo existen los checkpoints de las épocas
600, 800 y 1000 — el último 40 % del entrenamiento. En range-view el óptimo estaba
en la época 50 de 6000, o sea al 0,8 % de la corrida; el equivalente acá sería la
época ~8. Si el pico fuera igual de temprano, las tres mediciones caerían todas
después de la caída y una meseta baja se vería exactamente así de plana. Lo medido
es *"¿se degrada en el último 40 %?"* (no), no *"¿es la 1000 la mejor época?"*
(abierto). Distinguirlo cuesta ~1,6 h de re-pre-entrenamiento con checkpoints cada
25 épocas.

---

## Experimento 24: multimodal k=6 — la métrica de la literatura mejora 44 % mientras la predicción empeora

**Fecha:** 2026-09-03 · **Rama:** `decoder/multimodal-wta`
**Script:** `run_multimodal.sh` · **CSV:** `work_dirs/multimodal/multimodal_results.csv`
**n = 5 folds × 8 semillas × 2 escenas de validación** (160 filas, comparación
pareada por (fold, semilla); el test entre folds usa **n = 5**, no 40).

Reproducir:

```bash
python agregar_resultados.py work_dirs/multimodal/multimodal_results.csv \
    --comparar baseline_k6:baseline_k1 --por-fold                 # ADE real
python agregar_resultados.py work_dirs/multimodal/multimodal_results.csv \
    --comparar baseline_k6:baseline_k1 --metrica minade --por-fold  # minADE_6
```

### Por qué

Wayformer y MTR no predicen **una** trayectoria sino **K hipótesis**, entrenadas con
pérdida *winner-takes-all* (solo el modo más cercano al futuro real recibe gradiente
de regresión) y una cabeza de clasificación que aprende a puntuarlas. Es una de las
tres brechas metodológicas identificadas frente a esos papers. Se implementó en
`trajectory_model_attn.py` y `baseline_model.py` con `num_modes=K` y `cls_weight`.

Se midió sobre el **baseline cinemático**, no sobre el modelo con escena: la
arquitectura grande tarda 8× por corrida, y la pregunta —¿sirve la multimodalidad?—
no depende de la escena. `num_modes=1` reproduce exactamente el comportamiento
anterior, lo que mantiene válidos los checkpoints de los experimentos 15-22.

### Resultado

| métrica | k=1 | k=6 | efecto | p | folds a favor |
|---|---|---|---|---|---|
| ADE (modo más probable) | 2,988 | 3,285 | **+0,298** | 0,036 | **0/5** |
| ADE móviles | — | — | **+0,303** | 0,036 | **0/5** |
| FDE (modo más probable) | 6,433 | 6,937 | **+0,504** | 0,041 | 1/5 |
| **minADE_6** | 2,988 | 2,264 | **−0,723 (−24 %)** | 0,005 | **5/5** |
| **minFDE_6** | 6,433 | 4,479 | **−1,954 (−44 %)** | 0,006 | **5/5** |

Las dos mitades de la tabla dicen cosas **opuestas**, y las dos son correctas.

### El hallazgo que vale por sí solo

**Reportar solo minADE/minFDE —que es lo que hace la literatura— habría mostrado
una mejora del 24 % y 44 %, con p<0,01 y 5/5 folds, sobre un modelo cuya predicción
real empeoró de forma significativa en 0/5 folds.**

minADE_k toma el mejor de los K modos *sabiendo cuál fue el futuro real*. Es un
oráculo: mide si entre las hipótesis hay una buena, no si el modelo sabe elegirla.
Con K=1 coincide con el ADE; con K=6 se vuelve una métrica distinta, y compararla
contra el ADE de un modelo unimodal —como se hace en la práctica— es comparar un
oráculo contra una predicción.

Esto es un resultado metodológico, del mismo orden que el `gate_init` del
experimento 20: no cambia qué modelo es mejor, cambia qué números se pueden creer.
**Toda métrica `min*` de este proyecto se reporta junto a la métrica del modo más
probable, nunca sola.**

### La causa, leída de las curvas

De `work_dirs/multimodal/baseline_k6_f0s0.log`:

| época | `wta_reg` | `wta_cls` |
|---|---|---|
| 1 | 0,1165 | 1,5961 |
| 25 | **0,0149** | 1,2856 |
| 50 | 0,0280 | 1,6908 |
| 100 | 0,0365 | 1,4535 |

Tres cosas a la vez:

1. **`cls` es ~40× mayor que `reg`.** Con `cls_weight=1.0` el gradiente total lo
   domina la clasificación.
2. **La regresión empeora después de la época 25** (0,0149 → 0,0365): el modelo
   *desaprende a predecir* mientras persigue clasificar.
3. **El clasificador casi no aprende.** `cls` se queda en ~1,45 contra el 1,79 del
   azar puro (log 6).

El winner-takes-all **sí** especializa los modos —minADE_6 = 2,264 lo demuestra: las
hipótesis buenas están ahí—. Pero al predecir se elige por `argmax` de los logits, y
ese clasificador no distingue cuál sirve. **La brecha 3,285 vs 2,264 es exactamente
el costo de elegir mal, no de predecir mal.**

### El brazo que se canceló

El diseño original tenía un segundo brazo `gate0_k6` (modelo completo con la escena,
gate congelado en 0). Se **canceló** al terminar el baseline: eran 8,5 h de GPU para
reproducir el mismo desbalance en un modelo 8× más caro, con el diagnóstico ya
hecho. No hay números de `gate0_k6` — el CSV no los contiene y no deben citarse.

### Lo que abre

Si la brecha es de *selección* y no de *predicción*, bajar `cls_weight` debería
recuperar la regresión. Barrido en `run_clsweight.sh` (0,01 / 0,05 / 0,2 × 2 folds
× 4 semillas; el 1,0 ya medido entra de cuarto punto).

**Es un barrido para elegir un hiperparámetro, no un resultado.** El peso que gane
se valida después sobre los 5 folds completos: elegir y reportar sobre los mismos
folds sería el error de la regla 2.

Hipótesis alternativa si ningún peso alcanza: con 236 ventanas de entrenamiento
repartidas en 6 modos, cada modo ve ~39 ejemplos. Sería otra vez el cuello de datos
(experimento 21), no un problema de la pérdida.

---

## Experimento 25: `cls_weight` no salva la multimodalidad — y el barrido casi produce una conclusión falsa

**Fecha:** 2026-09-03 · **Rama:** `decoder/multimodal-wta`
**Scripts:** `run_clsweight.sh` (barrido) + `run_clsweight_val.sh` (validación)
**CSV:** `work_dirs/clsweight/clsweight_results.csv`, `work_dirs/clsweight_val/clsweight_val_results.csv`
El `baseline_k1` y el `cls_weight=1.0` se reusan de `work_dirs/multimodal/multimodal_results.csv`
(mismo config `noclip_base_fold*.py`, mismas semillas, pareo por (fold, semilla)).

### Por qué

El experimento 24 dejó un diagnóstico: el winner-takes-all especializa los modos
—minADE_6 lo prueba— pero el clasificador no sabe elegirlos, y `wta_cls` es ~40×
mayor que `wta_reg`. La hipótesis era que `cls_weight=1.0` estaba mal calibrado y
que bajarlo recuperaría la regresión.

### El barrido

3 pesos × 2 folds × 4 semillas. El `1.0` ya estaba medido y entra de cuarto punto.

| `cls_weight` | efecto vs k=1 | folds a favor | n |
|---|---|---|---|
| 0,01 | +1,122 | 0/2 | 2 folds |
| **0,05** | **−0,264** | **2/2** | 2 folds |
| 0,2 | +0,020 | 0/2 | 2 folds |
| 1,0 | +0,298 | 0/5 | 5 folds (exp. 24) |

El 0,05 parecía sólido: único que le ganaba al k=1, y con un efecto relativo casi
idéntico en los dos folds (**−7,6 %** y **−8,1 %**).

**La hipótesis monótona era falsa.** La brecha ADE − minADE en el fold 0 va
3,32 / 1,35 / 1,80 / 2,26 para 0,01 / 0,05 / 0,2 / 1,0: tiene un **mínimo**, no una
pendiente. Con 0,01 el clasificador se queda sin gradiente y elige casi al azar
entre modos muy especializados — peor que no tener modos.

### La validación, y por qué se diseñó así

Los folds 0 y 1 son donde se **eligió** el 0,05. Reportar ahí sería sesgo de
selección. `run_clsweight_val.sh` corrió el 0,05 sobre los **folds 2, 3 y 4, que
nunca participaron de la elección**, y completó las semillas 4-7 de los folds 0-1
para poder publicar la tabla de 5 folds.

**El test independiente no replica — da vuelta el signo:**

| fold | efecto vs k=1 |
|---|---|
| 2 | +0,342 |
| 3 | +0,214 |
| 4 | +0,116 |
| **entre folds (n=3)** | **+0,224 ± 0,113 · p=0,075 · 0/3 folds** |

Tabla de 5 folds (folds 0-1 **sesgados** por la elección): +0,069 ± 0,250, p=0,57,
2/5 folds. Con las 8 semillas, la ventaja del fold 1 se encogió de −0,184 a −0,014.

### Lo que se concluye

**Ningún `cls_weight` probado mejora la predicción real.** El 0,05 es, en el mejor
de los casos, indistinguible del k=1; en los folds retenidos es peor.

La multimodalidad con winner-takes-all **no aporta en este peldaño**. La explicación
más probable no es la pérdida sino los datos: 236 ventanas de entrenamiento
repartidas en 6 modos dan ~39 ejemplos por modo. Es el mismo cuello del
experimento 21.

### El hallazgo del exp. 24 se refuerza

minADE_6 con `cls_weight=0,05`, sobre los 5 folds: **−0,858 (−29 %), p=0,003,
5/5 folds** — mientras el ADE real no mejora (2/5 folds, p=0,57). El mismo patrón
que en el 24 con `cls_weight=1,0`, ahora en **folds independientes y con otro
hiperparámetro**. Ya no es un accidente de un peso: es cómo se comporta el WTA acá.

### El segundo hallazgo, metodológico

**Elegir el mejor de 3 pesos sobre 2 folds fabricó un efecto que parecía sólido y
que se dio vuelta en los folds retenidos.** No fue un número ruidoso y evidente: era
−7,6 % y −8,1 %, dos folds de acuerdo, con la consistencia que uno usa como señal de
confianza. Pasó a +0,224 y 0/3 folds.

Es la regla 2 en acción, y esta vez el diseño lo atrapó **antes** de que llegara a
ninguna conclusión — a diferencia de las once retractaciones anteriores. Lo que lo
hizo posible fue decidir la partición **antes** de mirar: los folds de validación se
fijaron al escribir el barrido, no después de ver el resultado.

**Regla que queda:** todo hiperparámetro elegido por barrido se valida en folds que
no participaron de la elección, y el número que se reporta es el de esos folds.

---

## Experimento 26: la época del encoder no cambia nada — la adenda del 23 queda cerrada

**Fecha:** 2026-09-03 · **Rama:** `decoder/multimodal-wta`
**Scripts:** `run_curva_mae.sh` + `curva_mae_voxel.py`
**CSV:** `work_dirs/f1cv_curva/curva_fold{0..4}.csv`
**n = 5 folds × 101 checkpoints** (épocas 10..1000 cada 10, más el modelo sin entrenar),
4 máscaras pareadas por ventana, población `val` = las 2 escenas RETENIDAS de cada fold.

### Por qué

La adenda del experimento 23 midió las épocas 600/800/1000 y no encontró
degradación, pero dejó dicho lo que no podía probar: el config tiene
`checkpoint=dict(interval=200, max_keep_ckpts=3)`, así que eso es **el último 40 %**
y en disco no quedaba nada anterior. En range-view el óptimo estaba en la época 50
de 6000 (0,8 % de la corrida); el equivalente acá sería la época ~8, y un pico así
de temprano se vería exactamente como la meseta plana que midió la adenda.

Importaba porque **toda la Fase 1 (exp. 19-22) usó `epoch_1000`**, y es el conjunto
que respondió que la escena no aporta (0/5 folds).

### El control de sanidad, primero

Misma semilla y mismo config, solo cambia el hook de checkpoint. La época 1000 de la
curva reproduce el valor de la adenda en los cinco folds:

| fold | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| diferencia vs adenda | −0,0000 | −0,0000 | +0,0000 | −0,0000 | +0,0025 |

La curva es comparable con la adenda tensor a tensor.

### El "mejor checkpoint" es un artefacto, y hay que decirlo

`curva_mae_voxel.py` imprime la época de menor pérdida. Esas épocas son
**530, 450, 960, 30 y 100** — dispersas por todo el rango. En el fold 0 el mínimo
cae a **−2,69 sd** de la media de los 91 checkpoints posteriores a la época 100, y
el máximo a +2,42 sd: exactamente los extremos que produce tomar el mejor de 91
sorteos con sd 0,0052.

**Ese número no se cita.** Es la trampa 29 —elegir el mejor de muchos sobre una sola
medición— en otra forma. La lectura correcta promedia el ruido en ventanas gruesas.

### Resultado 1 — ¿se degrada al final? No

Meseta (épocas 100-600) contra último tercio (700-1000), fold por fold:

| fold | meseta | último tercio | efecto |
|---|---|---|---|
| 0 | 0,1867 | 0,1933 | +3,6 % |
| 1 | 0,2273 | 0,2501 | +10,0 % |
| 2 | 0,1440 | 0,1291 | **−10,4 %** |
| 3 | 0,1630 | 0,1611 | −1,1 % |
| 4 | 0,1727 | 0,2001 | +15,9 % |

**Entre folds: +0,0080 ± 0,0175 · t=1,03 · p=0,36 · 3/5 folds.**

No hay degradación sistemática. El fold 0 solo daba +3,6 % y parecía una señal; el
fold 2 va en dirección contraria por −10,4 %. La varianza entre folds se come el
efecto — el patrón de siempre en este proyecto, y la razón de la regla 2.

### Resultado 2 — ¿hay un pico temprano como en range-view? Tendencia, no

Épocas 10-100 contra el resto (110-1000):

| fold | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| efecto | −2,8 % | −10,3 % | **+7,4 %** | −16,2 % | −11,3 % |

**Entre folds: −0,0134 ± 0,0155 · −6,6 % · t=−1,92 · p=0,127 · 4/5 folds.**

Cuatro de cinco folds prefieren las épocas tempranas, pero no llega a significancia.
Es el mismo territorio que el hallazgo de capacidad (p=0,102, 5/5 folds): una
dirección consistente que no alcanza el umbral con n=5.

### Lo que se concluye

**La elección de época del encoder no compromete los experimentos 19-22.** Toda la
caída ocurre antes de la época 10 (0,748 → 0,188 en el fold 0); después la curva es
ruido alrededor de una meseta. `epoch_1000` es defendible.

La adenda del experimento 23 queda **cerrada**, y con mucho mejor respaldo: no era
que sus tres mediciones cayeran después de un pico, es que **no hay un pico**.

### Lo que este experimento NO tocó, y es lo que importa

Todo esto mide **pérdida de reconstrucción del MAE**. El vínculo entre reconstrucción
y ADE **nunca se estableció en este proyecto**: no hay ninguna medición de que un
encoder que reconstruye mejor produzca una trayectoria mejor.

O sea que aun si el pico temprano hubiera dado significativo, no se seguía que
re-correr los exp. 19-20 con esa época mejorara nada. El seguimiento que se había
fijado de antemano —re-correr con la época 300— **no se dispara**, y de todos modos
habría sido un salto por encima de un eslabón sin medir.

Ese eslabón es medible y es barato: el fold 3 tiene un 27 % de diferencia de
reconstrucción entre la época 30 y la 1000. Correr el decoder con los dos encoders y
comparar el ADE responde si la reconstrucción predice algo del desempeño río abajo —
una pregunta más básica que cualquiera de las que veníamos haciendo.

---

## Experimento 27: la reconstrucción del MAE NO predice el ADE

**Fecha:** 2026-09-04 · **Rama:** `decoder/multimodal-wta`
**Scripts:** `recon_dos_ckpts.py` (eje x) + `run_recon_ade.sh` (eje y)
**CSV:** `work_dirs/recon_ade/recon_ade_results.csv`, `work_dirs/f1cv_curva/recon_dos_ckpts.csv`
**n = 5 folds × 4 semillas × 2 encoders** (40 corridas), pareado por (fold, semilla);
el test entre folds usa **n = 5 folds**.

### Por qué

Los experimentos 17, 21, 23 y 26 miden **pérdida de reconstrucción** del encoder y
sacan conclusiones sobre el pipeline. Pero nunca se verificó que un encoder que
reconstruye mejor produzca una trayectoria mejor. Todo ese diagnóstico descansaba en
un supuesto sin medir.

### El diseño

El experimento 26 dejó, por fold, dos encoders del **mismo** pre-entrenamiento que
difieren en reconstrucción. Se re-midieron con **máscaras frescas** (semillas 100-103;
la selección del exp. 26 usó 0..3), porque la "mejor época" se eligió como mínimo de
91 y su ventaja medida está sesgada. La ventaja se encogió un **37 %** por regresión
a la media — y la del fold 0 se dio vuelta, confirmando **medido** que ese mínimo era
artefacto de selección.

**`use_gate=False`, y es lo central.** Con el gate aprendible el modelo lo cierra a
~0,004: la escena no llega al decoder y cambiar de encoder no movería nada. Con la
rama de escena siempre activa, la calidad del encoder puede expresarse. Es **la
condición más favorable posible** a que la reconstrucción importe.

Los dos encoders salen del mismo `work_dir`: la única diferencia entre brazos es la
época, no la corrida de pre-entrenamiento.

### Resultado

| fold | ventaja de reconstrucción | efecto en ADE | semillas a favor |
|---|---|---|---|
| 0 | +1,4 % | +1,4 % | 2/4 |
| 1 | −8,5 % | **−38,7 %** | 4/4 |
| 2 | **−0,4 %** | **+32,6 %** | 0/4 |
| 3 | −16,1 % | −14,5 % | 2/4 |
| 4 | **−24,4 %** | **+2,3 %** | 1/4 |

**Entre folds: −0,063 ± 1,027 · t=−0,14 · p=0,90 · 2/5 folds.**

**Correlación reconstrucción–ADE: r = +0,34** (t=0,62, df=3; en relativos r=+0,29).
Si la reconstrucción predijera el ADE, r debería estar cerca de **+1**.

### Lo que lo cierra

Las dos filas que matan la hipótesis son la 4 y la 2:

- El fold **4** tiene la **mayor** ventaja de reconstrucción de los cinco (−24,4 %) y
  produce un efecto en ADE de **+2,3 %** con 1/4 semillas: **cero**.
- El fold **2** tiene una diferencia de reconstrucción de **−0,4 %** —o sea ninguna— y
  produce **+32,6 %** de diferencia en ADE, con 4/4 semillas de acuerdo.

El orden de los efectos no sigue al de las ventajas. El efecto más grande está donde
la ventaja es mediana (fold 1) y el segundo más grande, invertido, donde la ventaja es
nula (fold 2).

### El piso de ruido, que es un resultado en sí

Los folds 0 y 2 funcionan como **control natural**: reconstrucción prácticamente
idéntica entre los dos encoders, y sin embargo dan **+1,4 %** y **+32,6 %** de
diferencia en ADE. O sea que **dos encoders que reconstruyen igual producen decoders
que difieren hasta un 33 % en ADE**.

Eso explica por qué el −38,7 % del fold 1 no significa nada: cae dentro de ese piso.
Y explica algo más: el pareo por semilla cancela el **87 %** del ruido (sd 1,996 →
0,265 en el fold 0), pero **no cancela nada del ruido de identidad del encoder**, que
es el que domina.

### Lo que hay que releer con esta luz

**Medir reconstrucción del MAE no informa sobre el desempeño río abajo.** Los
diagnósticos de encoder de los experimentos 17, 21, 23 y 26 son válidos como lo que
son —mediciones de reconstrucción— pero **no autorizan conclusiones sobre ADE**, y en
varios lugares se las usó como si lo hicieran.

En particular, el experimento 26 concluyó que la elección de época no compromete los
exp. 19-22 porque no hay pico en la curva de reconstrucción. Esa conclusión **se
mantiene, pero por otra razón y más fuerte**: la época no importa porque la
reconstrucción no importa.

### Los dos límites, dichos

1. **Régimen degradado.** Con `use_gate=False` el ADE absoluto ronda 7,4 contra 4,510
   del baseline cinemático en el mismo fold y las mismas semillas: forzar la escena
   activa cuesta un 66 %. La relación podría existir en un régimen donde la escena
   ayude — pero ese régimen no se ha encontrado en 20 experimentos, y el gate aprendido
   cierra justamente porque no existe.
2. **n = 5 folds, 4 semillas.** Con r=+0,34 y df=3 no se puede *descartar* una
   correlación moderada. Lo que sí se descarta es una relación fuerte y utilizable:
   el fold con 24 % de ventaja no mostró nada.

---

## Experimento 28: la escena no contenía al objeto — centrarla en él lo arregla

**Fecha:** 2026-09-04 · **Rama:** `decoder/multimodal-wta`
**Script:** `run_objcentrico.sh` · **CSV:** `work_dirs/objcentrico/objcentrico_results.csv`
**n = 5 folds × 4 semillas × 2 variantes** (40 corridas), pareado por (fold, semilla);
el test entre folds usa **n = 5 folds**.

### El hallazgo que lo origina

La caja de vóxeles de Fase 1 cubre **±10 m alrededor del EGO**
(`spatial_range=[-10,10,-10,10,-2,4]`, `voxel_res=2.0` → 10×10×3 = 300 tokens).
Medido sobre las 236 ventanas del fold 0:

| | |
|---|---|
| distancia mediana del objeto al ego | **32,7 m** |
| percentil 75 / 90 | 45,5 m / 56,7 m |
| **ventanas con el objeto dentro de la caja toda su historia** | **26/236 = 11,0 %** |
| ventanas con el futuro completo dentro | 7,2 % |

**En el 89 % de los casos el objeto a predecir no está en la escena que el encoder
ve.** El modelo mira el entorno inmediato del sensor y se le pide predecir un agente
que está a 33 m, fuera de la caja.

Probablemente se llegó ahí optimizando el número de tokens: el default de la clase
es ±40 m con `voxel_res=0.5` → 307.200 vóxeles, inviable. Bajarlo a ±10 m con res
2.0 da los 300 tokens que el ViT consume — pero **dejó a los objetos afuera**.

### Explica cuatro negativos de una vez

| experimento | resultado | por qué |
|---|---|---|
| 19-20 | la escena no aporta, el gate cierra a 0,0042 | no hay objeto que ver; el gate hace bien en descartarla |
| 19 | más capacidad no ayuda (p=0,102) | capacidad sobre una región irrelevante |
| 22 | la historia completa no ayuda | más frames de lo mismo irrelevante |
| 27 | la reconstrucción no predice el ADE (r=+0,34) | el encoder reconstruye el entorno del EGO |
| 21 | los encoders **sí** generalizan | compatible: generalizan reconstruyendo el entorno del ego |

No eran cinco resultados independientes apuntando a "faltan datos". Era **un defecto
geométrico** visto desde cinco ángulos.

### El cambio

`centrar_en_objeto=True` traslada la nube por `−centers[0]` antes de voxelizar.
Mismos 300 tokens, mismo costo. Verificado antes de correr nada:

| control | resultado |
|---|---|
| objeto dentro de la caja toda su historia | **11,0 % → 100,0 %** |
| la trayectoria cambia | **no** — max\|dif\| = 0 |
| la escena cambia | sí |
| ocupación / grillas vacías | 35,9 % → 27,4 % / ninguna |

Que la trayectoria no cambie es lo que hace limpio el experimento: **lo único que
difiere entre brazos es la escena**. Y con el default la ocupación da 35,9 %, el
número ya documentado: el camino anterior quedó intacto.

**Default `False` a propósito**: los experimentos 15-27 se midieron con la caja
ego-céntrica y tienen que seguir reproduciéndose.

**Arregla también la augmentación.** `_augment` rota `relative` alrededor del objeto
y la grilla con `np.rot90`, o sea alrededor del centro de la grilla. Con la caja
ego-céntrica son dos puntos distintos y el giro es incoherente, pese a que el
comentario dice "aplicada consistentemente". Centrando en el objeto comparten centro.

### Control de sanidad

Con el gate congelado en 0 la escena se anula, así que `gate0_obj` debe reproducir
exactamente el `gate0` ego-céntrico. Lo hace, semilla por semilla:

| semilla | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| `gate0` (ego) | 3,099 | 3,215 | 4,378 | 5,001 |
| `gate0_obj` | 3,099 | 3,215 | 4,378 | 5,001 |

Diferencia 0,000 en las cuatro. Confirma que el cambio no tocó nada fuera de la
escena y que las dos mediciones comparten referencia.

### Resultado

Efecto de la escena (`gated − gate0`), fold a fold, en las dos geometrías:

| fold | ego-céntrico | objeto-céntrico | mejora |
|---|---|---|---|
| 0 | +0,841 | +0,282 | −0,560 |
| 1 | +0,238 | −0,260 | −0,498 |
| 2 | +0,005 | −0,024 | −0,029 |
| 3 | +0,247 | +0,033 | −0,214 |
| 4 | +0,041 | −0,108 | −0,148 |
| **media** | **+0,274** | **−0,015** | **−0,290** |

**Centrar en el objeto mejora −0,290 ± 0,229 · t=−2,83 · p=0,0475 · 5/5 folds.**

Y el efecto absoluto de la escena pasa de **+0,274 (perjudica, 0/5 folds)** a
**−0,015 (neutro, p=0,87, 3/5 folds)**.

### Lo que se concluye, y lo que no

**Sí:** la escena LiDAR pasó de **perjudicar** a ser **neutra**, con los cinco folds
de acuerdo. Es el primer resultado significativo a favor de la escena en 28
experimentos, y tiene un mecanismo medido detrás, no una hipótesis post hoc.

**No:** que la escena aporte. El efecto absoluto sigue siendo indistinguible de cero.

Y el gate lo confirma: arrancando de 0,05, cierra a **−0,0001 / +0,0021 / +0,0030 /
+0,0043 / +0,0049** en los cinco folds. Con el objeto dentro de la caja el 100 % de
las veces y libertad para usar la escena, **el modelo la apaga igual**. Cuando el
gate cierra, `gated_obj` colapsa sobre `gate0_obj`, y por eso el efecto es cero.

O sea: el diagnóstico geométrico era **correcto pero incompleto**. Explicaba por qué
la escena hacía daño. No explica por qué, ya corregido, sigue sin haber señal.

**Lo que sí queda establecido es que el "la escena no aporta" de los exp. 19-20
estaba contaminado**: se midió con una escena que en el 89 % de los casos no
contenía al objeto. La pregunta central recién ahora está bien planteada.

### Salvedades

1. **p=0,0475 está justo bajo el umbral, con n=5 folds.** Este proyecto ya vio dos
   veces un efecto así darse vuelta al validarlo. Lo que da confianza no es la p
   sino los 5/5 folds y el mecanismo medido. **Replicar con 8 semillas antes de
   tratarlo como establecido.**
2. **El encoder MAE sigue siendo ego-céntrico.** `LidarSequenceDataset` no conoce
   los objetos, así que centrar el pre-entrenamiento es un cambio aparte. Hay
   desajuste de dominio, y juega EN CONTRA: el resultado se obtuvo a pesar de él.
3. **Los tipos de agente siguen mezclados** — 88,1 % vehículos, 5,9 % ciclistas,
   5,9 % peatones (clasificados por tamaño de caja; la extracción no guardó el tipo,
   `for track in proto.tracks` sin filtro). Un peatón se mueve 10× más lento y
   comparte cabeza y normalización con los autos. No se tocó a propósito: dos
   cambios a la vez habrían impedido atribuir la mejora.

### Lo que esto abre

Si el modelo apaga una escena que **sí** contiene al objeto, el cuello está antes o
después del encoder, no en él. Y hay un candidato medido: la escena que entra son
**300 vóxeles × 5 frames de ocupación BINARIA = 1.500 bits**, con vóxeles de 2 m
donde un auto ocupa 2,2×1 y **un peatón 0,4×0,4 — menos de uno**. Se comprimen 6.345
puntos LiDAR a 1.500 bits, 4 puntos por bit, sin intensidad ni densidad ni altura
fina.

Del otro lado, los 300 tokens de 1024 dims se comprimen con **una sola query** de
cross-attention a **64 dims** antes de concatenarse con la historia.

De los tres eslabones —representación de entrada, encoder, consumo en el decoder—
el encoder es el único medido y funciona (exp. 21). Los otros dos no se tocaron en
28 experimentos.

Esto además reinterpreta el exp. 27: **reconstruir ocupación binaria bien no exige
codificar nada útil para predecir movimiento**. El objetivo del MAE puede estar
desalineado con la tarea, que es distinto de que el encoder sea malo.
