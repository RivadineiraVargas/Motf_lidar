# Estudio: Wayformer y trabajos relacionados (Claudine Secs. 8-10, paso 12)

Objetivo: extraer de cada trabajo las decisiones de diseño que nuestro decoder
(fase 2) tiene que tomar. No es un survey — es la base para el diseño del
decoder mini a escala 10-lambidas y su versión final.

## 1. Wayformer (Nayakanti et al., Waymo, 2022) — la referencia principal

**Idea**: un par encoder-decoder Transformer simple y eficiente para motion
forecasting. Entrada multimodal (historia de agentes, mapa, semáforos) →
trayectorias futuras multimodales por agente.

**Scene encoder** — 3 decisiones estudiadas:
- **Fusión**: temprana (concatenar modalidades y un solo encoder — la más simple,
  y en sus ablaciones rinde igual o mejor), tardía (un encoder por modalidad),
  o jerárquica. *Nuestro caso*: una sola modalidad (LiDAR range-view) → fusión
  trivial; si sumamos historia del objeto como segunda modalidad, la evidencia
  de Wayformer sugiere fusión temprana.
- **Atención**: multi-eje (todo con todo) vs factorizada (espacio/tiempo por
  separado). Multi-eje gana en calidad; factorizada ahorra cómputo.
- **Latent queries**: comprimir la escena a un set chico de queries aprendidas
  (estilo Perceiver) para abaratar la atención. *Relevante para nosotros*: nuestros
  6784 patches del MAE son muchos tokens; un bloque de latent queries (p.ej. 64-256)
  entre encoder y decoder abarata todo lo que sigue.

**Decoder / salida** (lo central para las Secs. 8 y 10):
- k **queries aprendidas por modo** (una por hipótesis de trayectoria) atienden a la
  representación de la escena.
- Salida por modo: una **mezcla de Gaussianas** sobre la trayectoria (media +
  covarianza por waypoint) + un **logit de probabilidad del modo**.
- Loss: máxima verosimilitud sobre el modo más cercano al GT (winner-takes-all
  suavizado) + clasificación del modo.
- Predice **por agente conocido** (condiciona en la historia del agente), no
  descubre agentes — la detección viene dada.

## 2. Waymo Open Motion Dataset (Ettinger et al., 2021) — el benchmark

- Formato del problema: **1.1 s de historia** (11 frames a 10 Hz) → predecir
  **8 s de futuro** muestreados a 2 Hz (16 waypoints). *Nota*: nuestros datos
  WOMD-LiDAR dan exactamente esos 11 frames de historia con LiDAR — nuestra
  limitación de horizonte (~1 s) viene de que el LiDAR público cubre solo la
  ventana de historia, no el futuro.
- Métricas oficiales: **minADE, minFDE, Miss Rate, Overlap, mAP** (sobre k
  hipótesis). Nuestra ADE/FDE actual es el caso k=1 — alineado (Sec. 12 de
  Claudine: "aproximar la evaluación de los benchmarks a medida que madure").
- Define categorías de dificultad (vehículos/peatones/ciclistas) → "erro por
  classe" del pedido de Claudine mapea directo.

## 3. MotionTransformer / MTR (Shi et al., 2022)

- **Motion query pairs**: separa la query en (a) *intención global* — un set de
  ~64 puntos-objetivo aprendidos por k-means sobre los endpoints del dataset — y
  (b) *refinamiento local* de la trayectoria alrededor de esa intención.
- Ventaja: estabiliza el entrenamiento multimodal (cada query se especializa en
  una región del espacio de metas) — ataca el mode collapse del winner-takes-all.
- *Para nosotros*: a escala mini no hace falta; anotarlo como mejora si al pasar
  a multimodal las hipótesis colapsan.

## 4. DenseTNT (Gu et al., 2021)

- Goal-based **sin anclas**: primero estima una distribución densa de probabilidad
  de meta sobre el mapa, después completa la trayectoria hacia las metas top.
- Lección: separar "a dónde va" de "cómo llega" es una descomposición efectiva.
- *Para nosotros*: menos aplicable — depende fuerte de mapa HD que no tenemos
  (nuestra escena es LiDAR crudo). Se cita como alternativa descartada con motivo.

## 5. MotionLM (Seff et al., 2023)

- Trata las trayectorias como **tokens discretos de movimiento** (deltas
  cuantizados) y modela el futuro multi-agente como un language model
  autorregresivo conjunto.
- Es la versión más pura de la "representación tokenizada" que menciona Claudine
  (Sec. 8, última opción). Produce distribuciones *conjuntas* (interacción entre
  agentes) en vez de marginales por agente.
- *Para nosotros*: elegante pero costosa (vocabulario, autorregresión). Fase
  futura, no el primer decoder.

## 6. Decisiones para NUESTRO decoder (respuestas a las preguntas de la Sec. 10)

| Pregunta de Claudine | Decisión inicial (mini) | Basada en |
|---|---|---|
| ¿Qué tokens representan objetos? | K=100 queries aprendidas, una por slot de objeto, con flag de validez | Sec. 8 del pedido + DETR/Wayformer |
| ¿Qué representa cada punto futuro? | Secuencia de desplazamientos relativos al último punto observado (5 waypoints a 10 Hz) | Wayformer (relativo es invariante a posición global) |
| ¿Trayectorias inválidas? | Cabeza sigmoide de validez por slot; BCE | Sec. 8 del pedido |
| ¿Múltiples hipótesis? | k=1 al inicio; k>1 con winner-takes-all después | Sec. 10 ("una única hipótese" primero) |
| ¿Incertidumbre? | Reusar nuestra cabeza NLL (σ por waypoint) de la fase 1 | ya implementada y calibrada |
| ¿Loss? | Huber sobre waypoints válidos + BCE validez (+ NLL si σ) | Wayformer simplificado |
| ¿Asociación pred↔real? | Matching húngaro por posición del primer waypoint (estilo DETR); alternativa simple: condicionar cada slot en la posición actual del objeto (estilo Wayformer, sin matching) | trade-off descubrir vs condicionar |
| ¿Evaluación? | ADE/FDE (ya tenemos) + accuracy de validez; después minADE_k | WOMD |

**Recomendación concreta**: empezar estilo **Wayformer condicionado** (cada objeto
presente aporta su posición actual como query; sin matching húngaro, sin
detección) porque valida el eslabón encoder→decoder con el mínimo de piezas
nuevas; el flag de validez se entrena igual rellenando slots vacíos hasta K.

## Referencias

- Nayakanti et al., *Wayformer: Motion Forecasting via Simple & Efficient
  Attention Networks*, arXiv:2207.05844.
- Ettinger et al., *Large Scale Interactive Motion Forecasting for Autonomous
  Driving: The Waymo Open Motion Dataset*, ICCV 2021.
- Shi et al., *Motion Transformer with Global Intention Localization and Local
  Movement Refinement*, NeurIPS 2022.
- Gu et al., *DenseTNT: End-to-end Trajectory Prediction from Dense Goal Sets*,
  ICCV 2021.
- Seff et al., *MotionLM: Multi-Agent Motion Forecasting as Language Modeling*,
  ICCV 2023.
