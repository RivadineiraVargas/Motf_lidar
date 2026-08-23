# CONTEXTO — briefing para auditoría independiente del proyecto MOTF

**Fecha:** 2026-08-23 · **Rama:** `encoder/validacao-mae` · **Commit:** ver `git log -1`
**Repo:** https://github.com/RivadineiraVargas/Motf_lidar

---

## 0. INSTRUCCIONES PARA EL AGENTE AUDITOR

### Tu tarea

Auditar este trabajo **buscando errores**. No se busca validación ni un resumen: se
busca que encuentres lo que está mal, lo que no se sostiene, y lo que se afirmó con
más confianza de la que la evidencia permite.

Este documento lo escribió el asistente que hizo el trabajo. **Está sesgado por
construcción.** Desconfía especialmente de las secciones donde suena convencido. La §7
lista los puntos débiles que ese asistente identifica en sí mismo — úsala como punto de
partida, no como el alcance de tu revisión. Que los haya listado no significa que sean
los únicos ni los peores, y admitir una debilidad no la neutraliza.

### Orden de trabajo sugerido (de mayor a menor rendimiento)

1. **Recalcular la estadística desde los CSV crudos.** Es lo más barato y lo más
   probable que rinda. Los 405 registros están en `sapiens/pretrain/work_dirs/*/[a-z]*results*.csv`
   (columnas: `fold,seed,arch,horizon_s,n_wp,ade,fde,acc,best_ep`). Verificá cada media,
   cada desvío, cada t y cada intervalo de confianza de las §3 y §4. No necesitás GPU.
2. **Revisar la lógica de los tests.** ¿Están bien pareadas las comparaciones? ¿El n es
   el que se declara? ¿Se mezclan poblaciones (objetos parados vs móviles)? ¿Hay
   comparaciones múltiples sin corregir? ¿Se usan los grados de libertad correctos?
3. **Leer el código de los modelos** (`train_decoder_mini.py`) contra lo que la §3.2
   afirma sobre ellos. El hallazgo principal del proyecto es una afirmación sobre
   *diferencias arquitectónicas* entre `MiniBaseline` y `MiniWayformerGated`:
   verificá que sea cierta y completa.
4. **Buscar fuga de datos.** Cada encoder de dominio se pre-entrenó en las 20 escenas
   de train de SU fold. Verificá en los CSV que ninguna corrida use un encoder fuera de
   su fold.
5. **Contrastar las conclusiones con los datos.** ¿La §3.3 dice más de lo que las
   tablas permiten? ¿Hay saltos de "no significativo" a "no hay efecto"?
6. **Recién al final**, si hace falta y tenés los datos, re-correr algo (§8).

### Qué cuenta como hallazgo

- Un número mal calculado, un test mal aplicado, un n mal declarado.
- Una conclusión que la evidencia no sostiene, o que sobre-generaliza.
- Un confundido no controlado (el proyecto ya encontró uno grave; puede haber más).
- Fuga de datos, o un control que no controla lo que dice controlar.
- Código que no hace lo que la documentación afirma que hace.

### Qué NO cuenta

- **Diferencias en los decimales al reproducir.** El pipeline tiene no-determinismo de
  GPU documentado (la misma configuración dio 2.79 y 2.51). Una desviación solo es
  hallazgo si **cambia una conclusión**; reportala indicando el n que usaste.
- Preferencias de estilo, de nomenclatura o de organización del código.
- Que el resultado sea negativo. Un negativo bien medido es un resultado válido; lo
  auditable es si está bien medido.

### Cómo reportar

Por cada hallazgo: **qué afirmación es**, **dónde está** (archivo y sección o línea),
**por qué está mal**, **qué evidencia lo demuestra**, y **cuán grave es** — distinguí
entre "cambia la conclusión principal", "cambia un resultado secundario" y "es
impreciso pero no cambia nada".

Si no encontrás errores en algo que revisaste a fondo, decilo explícitamente: saber qué
fue verificado y resistió es tan útil como la lista de fallas.

### Permiso explícito

Podés cuestionar el encuadre mismo. Si te parece que la pregunta de investigación está
mal planteada, que el reencuadre de la §3.3 es una racionalización de un fracaso, o que
todo el enfoque tiene un problema anterior a los números — decilo. Eso es lo más valioso
que podrías aportar y es exactamente lo que el autor no puede ver.

Lenguaje del proyecto: español (la documentación técnica mezcla español e inglés).

**Aviso de nombres:** este archivo es `CONTEXTO_AUDITORIA.md`. Existe además un
`Contexto.md` en la raíz, de junio, en portugués, que documenta el pipeline VIEJO de
vóxeles (Fase 1) y **está desactualizado** — no lo uses como referencia del estado
actual, aunque sí es contexto histórico legítimo (ver §11 sobre Fase 1).

---

## 1. El proyecto en una página

**MOTF (Moving Object Trajectory Forecasting)** — disertación de maestría en el
LCAD/UFES. Predecir trayectorias futuras de objetos móviles a partir de barridos
LiDAR crudos, con un transformer en dos fases:

- **Fase encoder:** MAE (Masked Autoencoder) auto-supervisado sobre imágenes
  range-view de LiDAR. Sin etiquetas. Aprende representación de escena
  reconstruyendo parches enmascarados.
- **Fase decoder:** supervisado, estilo Wayformer. Toma las features de escena del
  encoder + histórico del objeto → predice trayectoria futura.

**La pregunta de la tesis:** ¿la escena LiDAR mejora la predicción de trayectorias
por sobre un baseline puramente cinemático (histórico + velocidad constante)?

**La respuesta a la que llegamos: NO — y además la pregunta estaba mal planteada.**
Ver §3.

### Datos (el límite que condiciona todo)

- **25 escenas**, 11 frames LiDAR cada una = **275 sweeps**. Eso es todo lo que hay
  en disco.
- **Límite duro del dataset:** WOMD-LiDAR solo trae LiDAR del primer ~1.1 s de cada
  ventana de 9 s (11 de 91 frames). Los otros 8 s tienen etiquetas de trayectoria
  pero **no nube de puntos**. Confirmado con el paper (Chen et al. 2023) y por
  conteo directo. Consecuencia: el histórico de LiDAR está limitado a ~1 s.
- 25 escenas es **poquísimo**. Casi todos los problemas de este proyecto salen de ahí.

---

## 2. Cronología de los experimentos (14 en total)

Detalle granular en `docs/EXPERIMENTOS_DECODER.md`. Resumen del arco:

| # | Qué se probó | Resultado |
|---|---|---|
| 1-2 | CV 5 folds × 3 semillas, cross-attn cruda | baseline gana (t=−3.07) |
| 3 | Puente con pooling (16 latentes, Perceiver) | empata con crudo, ambos pierden |
| 4-5 | Fine-tuning parcial del encoder | mejora mixta, la accuracy de validez colapsa |
| 6 | Barrido de horizonte 1/3/5/8 s (encoder genérico) | la escena no ayuda a ningún horizonte |
| 7-8 | **Encoder adaptado al dominio** (fold 0) | primera señal a favor: −20.4%, p=0.0006, 8/8 semillas |
| 9 | Barrido de horizonte con encoder de dominio | "punto dulce" a 3 s reaparece |
| 10 | Réplica en fold 4 | no replica (nulo limpio) |
| 11 | **CV completa 5 folds × 8 semillas** | **el efecto NO sobrevive** (ver §3.1) |
| 12 | Gate aprendible | no rescata; empeora el peor split |
| 13 | Descomposición dirección vs magnitud | acotado al fold 3; no generaliza |
| 14 | **Barrido de gate CONGELADO** | **el efecto era ARQUITECTURA, no escena** (ver §3.2) |

**Patrón que se repitió cuatro veces:** un resultado positivo medido con pocas
semillas o un solo split, que se evaporó al muestrear bien. Ver §6.

---

## 3. Los dos resultados que sostienen la conclusión

### 3.1 Validación cruzada completa (exp. 11)

Un encoder MAE adaptado al dominio **por fold** (re-pre-entrenado desde cero solo
en las 20 escenas de train de ese fold, ~12.5 h cada uno), después decoder a 3 s,
8 semillas, wayformer vs baseline, comparación pareada.

| fold | baseline | wayformer | diff | t | semillas | relativo |
|---|---|---|---|---|---|---|
| 0 | 0.912 | 0.726 | −0.186 ± 0.089 | −5.94 | 8/8 | −20.4% |
| 1 | 1.252 | 1.190 | −0.061 ± 0.074 | −2.35 | 6/8 | −4.9% |
| 2 | 1.082 | 1.168 | +0.086 ± 0.084 | +2.89 | 1/8 | +7.9% |
| 3 | 1.424 | 1.993 | +0.570 ± 0.130 | +12.40 | 0/8 | +40.0% |
| 4 | 1.816 | 1.792 | −0.024 ± 0.115 | −0.59 | 4/8 | −1.3% |

**Entre folds: +0.077 ± 0.292, t=0.589, gl=4, NO significativo.** IC95
[−0.286, +0.439] (incluye el 0), 3/5 folds a favor. La media ni siquiera mantiene
el signo. sd entre folds (0.292) = **3×** sd entre semillas (0.098).

Datos crudos: `sapiens/pretrain/work_dirs/horizon_domain/horizon_results.csv`
(fold 0) y `work_dirs/horizon_fold{1,2,3,4}/horizon_results.csv`.

### 3.2 El control de arquitectura (exp. 14) — el hallazgo principal

Se congeló el gate de escena en valores fijos (0.0 a 0.99) para obtener la curva
dosis-respuesta. **`gatefix0.0` (escena totalmente apagada) debía reproducir el
baseline. No lo hace.**

La causa NO es un bug: `MiniBaseline` corre un **MLP independiente por objeto**,
mientras el modelo gated conserva **self-attention ENTRE objetos**, 2 capas y FFN.
La comparación "wayformer vs baseline", sobre la que descansó el proyecto entero,
nunca aisló la escena: medía escena + capacidad del decoder + interacción entre
agentes, todo junto.

Descomposición (8 semillas):

| componente | fold 0 | fold 3 |
|---|---|---|
| **arquitectura** (gatefix0.0 vs baseline, SIN escena) | **−0.129** t=−9.19 8/8 | **+0.578** t=+9.15 0/8 |
| **escena** (gate aprendido vs gate 0, misma arq.) | −0.023 t=−1.80 (ns) | +0.121 t=+1.91 (ns) |
| total reportado históricamente | −0.186 | +0.570 |

La arquitectura explica el **69%** del efecto en el fold 0 y el **101%** en el
fold 3. Lo que queda para la escena no es significativo en ninguno.

**Y la curva es plana:** de 0% a 99% de escena el ADE no se mueve fuera del ruido.
Única celda significativa (fold 3, gate 0.5, t=+2.76) va *en contra* de la escena,
es 1 de 10 comparaciones —lo que el azar predice al 5%— y no es monótona
(0.5 significativo, 0.99 no). **No hay relación dosis-respuesta.**

Datos: `work_dirs/horizon_domain/` y `work_dirs/horizon_fold3/horizon_results.csv`,
filas con `arch` = `gatefix*`.

### 3.3 Conclusión

1. La escena LiDAR no aporta: con ningún encoder (genérico o de dominio), ningún
   puente (cross-attn cruda, pooling, fine-tuning), ningún horizonte, ninguna dosis.
2. **Lo que dependía del split nunca fue la escena: era la arquitectura.** Un
   decoder transformer de 2 capas entrenado con 20 escenas gana 14% en un split y
   pierde 39% en otro.
3. Reencuadre de la contribución: de un negativo a secas a una **crítica
   metodológica con evidencia** — con 25 escenas la capacidad del decoder domina
   cualquier efecto de las features de escena, y la comparación estándar
   "con LiDAR vs baseline simple" está confundida con capacidad del modelo.

**Chequeo de consistencia externa:** el paper de WOMD-LiDAR reporta mejora marginal
en ADE por agregar LiDAR incluso con ~100k escenas y features supervisadas. Nuestro
resultado negativo está en línea con el estado del arte, no es un artefacto.

---

## 4. Resultado secundario reproducible

El gate **aprendido** converge a **0.0968 ± 0.0135** en los 5 folds, desde 40
inicializaciones distintas en 0.5:

| fold | gate_final |
|---|---|
| 0 | 0.0917 ± 0.0051 |
| 1 | 0.1059 ± 0.0045 |
| 2 | 0.1026 ± 0.0075 |
| 3 | 0.0772 ± 0.0079 |
| 4 | 0.1016 ± 0.0087 |

Es la cantidad más reproducible del proyecto. **Pero ojo:** por §3.2 esa cantidad
de escena no compra nada. Que el modelo converja consistentemente a 0.10 no implica
que 0.10 sea útil — la curva plana dice que da igual.

**Trampa relacionada:** el `scene_gate` del checkpoint guardado es el del MEJOR
checkpoint, NO el convergido. Como el early-stop suele elegir la época 1, ahí el
gate todavía vale ~0.497. El valor interpretable es `gate_final`, que solo se
imprime en el log.

---

## 5. Latencia (puente a la etapa 2)

El objetivo final declarado es correr sobre el vehículo del LCAD. Medido en
RTX 4060 Laptop (`sapiens/pretrain/latency_benchmark.py`):

| etapa | fp32 | autocast fp16 |
|---|---|---|
| encoder MAE (forward, 1 sweep) | 139.0 ms | 26.4 ms (×5.3) |
| decoder (K slots) | 2.6 ms | — |
| **total cómputo** | **141.6 ms → 7.1 Hz** | **~29 ms → ~34 Hz** |

A 10 Hz el presupuesto es 100 ms: en fp32 no entra, con precisión mixta entra con
3× de margen (error numérico 0.046%). El encoder es el **98%** del cómputo.

---

## 6. Las cuatro retractaciones (lee esto antes de confiar en cualquier número)

Este proyecto produjo cuatro afirmaciones que hubo que retirar. Todas por el mismo
error: reportar desde **una semilla o un solo fold**.

1. **"La escena ayuda"** (7.19 vs 7.85 en una escena, 18/07) → revertido por la CV.
2. **"Punto dulce a 3 s, −20.4%, p=0.0006, 8/8 semillas"** (un fold, 06/08) → se
   evaporó al promediar los 5 folds. **Nótese que tenía 8 semillas y p<0.001**: un
   t enorme dentro de un fold da falsa confianza si el split no está muestreado.
3. **"En el fold 0 el wayformer tiene menos errores gruesos de dirección"**
   (1 semilla, 18/08) → con 8 semillas es empate (t=−1.23).
4. **"El wayformer calibra mejor la magnitud"** (18/08) → artefacto de promediar
   también los objetos parados (72-75% del total). Entre móviles va al revés.

Una quinta, del 18/08, alcanzó a durar horas: **"la atención entre objetos ayuda
un 14%"** (−0.129 en fold 0) → el fold 3 la revierte (+0.578). También depende del
split.

**Regla que salió de esto:** nunca reportar sin decir n (semillas Y folds) y sobre
qué población se promedió.

---

## 7. DÓNDE MIRAR PRIMERO — puntos débiles que yo mismo identifico

Orden aproximado de gravedad. Esta lista es mi mejor esfuerzo, no una garantía de
exhaustividad.

1. **El barrido de gate solo se corrió en folds 0 y 3** (los dos extremos), no en
   los 5. La conclusión "la curva es plana" y la descomposición arquitectura/escena
   descansan en 2 de 5 splits. Dado el historial de este proyecto con la varianza
   entre splits, **esto es exactamente el tipo de generalización que ya falló cuatro
   veces**. Es el agujero más grande.
2. **`gatefix0.0` vs `wayformer` no es una comparación limpia.** `wayformer` usa
   `nn.TransformerDecoder` y `gatefix*` usa `GatedDecoderLayer` escrito a mano. En
   el fold 0, `wayformer` (0.726) le gana a `gatefix0.99` (0.761) aunque ambos usan
   la escena casi a fuerza completa → hay ~0.035 de efecto por detalles de
   implementación, **del mismo orden que el efecto de la escena**. Verificar si
   `GatedDecoderLayer` es realmente equivalente a la capa de PyTorch (post-norm,
   orden de operaciones, dropout).
3. **Los t-tests entre folds usan n=5** (las medias por fold). Con gl=4 la potencia
   es bajísima. "No significativo" ahí no es lo mismo que "no hay efecto". Verificar
   si la conclusión está sobre-vendida.
4. **Comparaciones múltiples sin corrección.** En el barrido de gate son 5
   comparaciones por fold. Yo argumento que la única celda significativa es azar,
   pero no apliqué Bonferroni ni nada formal.
5. **`best_ep`=1 en muchísimas corridas** — el modelo suele quedarse en la primera
   época (piso de velocidad constante puro). Verificar si el protocolo de early
   stopping está sesgando sistemáticamente las comparaciones, y si comparar modelos
   cuyo mejor checkpoint es la época 1 tiene sentido.
6. **El encoder no es determinista** (permuta tokens en cada llamada). Es inocuo
   para la cross-attention, pero verificar que ningún análisis compare salidas del
   encoder elementwise sin fijar semilla.
7. **ADE está deflactado por objetos parados** (~72-75% se mueven <1 m). Algunos
   análisis filtran móviles y otros no. Verificar consistencia y que cada número
   diga sobre qué población está.
8. **Las escenas retenidas de cada fold difieren mucho en dificultad** (baseline ADE
   de 0.91 a 1.82; el fold 3 tiene objetos que se desplazan el doble). Los promedios
   entre folds mezclan poblaciones muy distintas. Verificar si corresponde
   normalizar.
9. **Fuga de datos:** cada encoder de dominio se pre-entrenó en las 20 escenas de
   train de SU fold. Usarlo fuera de su fold sería fuga. Verificar que ningún
   experimento lo haga (`--folds` es obligatorio en `horizon_sweep.py`, pero
   verificar que se respetó en todas las corridas registradas en los CSV).
10. **La selección de semilla "representativa"** para las simulaciones (la de ADE
    más cercano a la media) es una decisión mía, defendible pero arbitraria.

---

## 8. Cómo verificar (todo es reproducible)

```bash
# entorno
conda activate sapiens_gpu          # torch 2.5.1+cu118
cd sapiens/pretrain

# los datos crudos de TODOS los experimentos estan versionados:
ls work_dirs/*/[a-z]*results*.csv

# re-derivar la CV de 5 folds (§3.1) desde los CSV, sin GPU:
#   columnas: fold,seed,arch,horizon_s,n_wp,ade,fde,acc,best_ep

# re-correr un fold (requiere GPU + el encoder de ese fold):
python horizon_sweep.py --enc work_dirs/rv_rect_fold3/epoch_1000.pth \
    --folds 3 --seeds 0 1 2 3 4 5 6 7 --horizons 3s \
    --archs wayformer baseline --cache work_dirs/cache_fold3_domain \
    --out work_dirs/horizon_fold3 --epochs 100

bash run_gate_sweep.sh              # exp. 14 (~10 h)
python angular_error_analysis.py    # exp. 13
python latency_benchmark.py         # §5
```

### Qué está en el repo y qué no

**En el repo (suficiente para auditar números y razonamiento, sin GPU):** todo el
código, toda la documentación, y **los CSV crudos de todos los experimentos**
(`work_dirs/*/[a-z]*results*.csv`, 32 KB, 405 filas). Con eso se puede recalcular
cada media, cada t-test y cada intervalo de confianza de este documento de forma
independiente. **Esa es la vía recomendada para la auditoría.**

**Fuera del repo, en un paquete aparte (~1 GB, pedir al usuario el enlace de Drive):**

| archivo | contenido | tamaño |
|---|---|---|
| `motf_datos.tgz` | `waymo_clean/{objs_bbox, range_files, poses, beam_inclinations.npy}` | ver `SHA256SUMS.txt` |
| `motf_encoders.tgz` | los 5 encoders de dominio `rv_rect_fold{0..4}/epoch_1000.pth` | 712 MB |
| `SHA256SUMS.txt` | checksums — verificar antes de reproducir | — |

```
280a58c490a3dcfd4a59f0c4d63d508ca6219818c28bb0c60971c14f33ab14eb  motf_datos.tgz   (239 MB)
80de4e2452f12d2fdfecad8919dfa0ab8a5cdc62899054c3e9194b022a037eb3  motf_encoders.tgz (712 MB)
```

Se descomprimen desde la raíz del repo (`tar xzf motf_datos.tgz`) y reconstruyen la
estructura de directorios esperada. Solo hacen falta si querés **re-entrenar** para
confirmar que los CSV no están inventados.

**AVISO:** `objs_bbox` son 1.546.555 archivos de ~280 bytes. Al descomprimir ocupa
**6,1 GB en disco** aunque su contenido real sean 591 MB — es overhead de bloques del
sistema de archivos, no un error. Descomprimir tarda varios minutos.

**Sobre el conteo de escenas (no es un error):** `objs_bbox` y `poses` traen **492
escenas**, pero `range_files` solo **25**. Las etiquetas de trayectoria son livianas y
existen para muchas escenas; el LiDAR es lo pesado y solo se descargó para 25. **Los
experimentos usan las 25 que tienen LiDAR** — son las que enumera `make_folds()` en
`cross_validate_decoder.py`. El resto de las etiquetas es material sobrante de la
extracción.

**Sobre reproducir: los decimales NO van a coincidir.** El pipeline tiene
no-determinismo de GPU documentado en las operaciones de atención — la misma
configuración dio 2.79 y 2.51 en los exp. 4 y 5. Re-correr un fold dará números
*parecidos* pero no idénticos a los de los CSV. **"No me dio exactamente igual" no es
evidencia de error ni de fraude.** Lo que debe coincidir son las conclusiones (signos,
significancia, orden de magnitud), no las cifras. Si algo se aparta lo bastante como
para cambiar una conclusión, eso sí es un hallazgo — repórtalo con el n usado.

**Deliberadamente NO se distribuye:** las features cacheadas
(`work_dirs/cache_fold*_domain`, 14 GB) porque son exactamente la salida del encoder
sobre los `range_files` y se regeneran solas al correr cualquier experimento; los
`bin_files` y `range_png_rect` (no los usa el pipeline del decoder); y los GIFs de
simulación.

---

## 9. Mapa del repositorio

| ruta | qué es |
|---|---|
| `docs/PROJECT_STATE.md` | **documento maestro**, estado actual completo (en inglés) |
| `docs/EXPERIMENTOS_DECODER.md` | log granular de los 14 experimentos |
| `docs/CHECKLIST_CLAUDINE.md` | plan de 17 pasos de la tutora, con evidencia |
| `docs/ESTUDIO_WAYFORMER.md` | decisiones de diseño del decoder |
| `docs/RESULTADOS_ADE_FDE.md` | Fase 1 (pipeline viejo de vóxeles) |
| `sapiens/pretrain/train_decoder_mini.py` | decoder + `train_decoder()` (fuente única del loop) |
| `sapiens/pretrain/cross_validate_decoder.py` | driver de CV 5 folds, define `make_folds()` |
| `sapiens/pretrain/horizon_sweep.py` | driver de barridos (horizonte/arquitectura) |
| `sapiens/pretrain/angular_error_analysis.py` | descomposición dirección/magnitud |
| `sapiens/pretrain/latency_benchmark.py` | latencia fp32 vs fp16 |
| `sapiens/pretrain/run_gate_sweep.sh` | barrido de gate congelado |
| `sapiens/pretrain/export_decoder_mini_global.py` | simulador: GIFs + txt para el viewer C++ |
| `show_point_cloud.cpp` | viewer C++ **offline** (lee `predictions_global.txt`) |
| `lidar_sweep_viewer_main.cpp` | viewer **online** del stack astro del LCAD — NO se usa en este trabajo; es el punto de integración de la etapa 2 |
| `sapiens/pretrain/work_dirs/*/[a-z]*results*.csv` | **evidencia cruda de todos los experimentos** |

### Arquitecturas del decoder (`train_decoder_mini.py`)

- `MiniBaseline` — MLP **independiente por objeto**, sin escena. **NO es un control
  válido de arquitectura** (ver §3.2).
- `MiniWayformerDecoder` — cross-attn cruda sobre los ~6785 tokens del encoder.
- `MiniWayformerPooled` — 16 latentes estilo Perceiver resumen cada sweep.
- `MiniWayformerGated` — gate escalar aprendible `tanh(scene_gate)` sobre la
  cross-attn.
- `gatefix<v>` — el gated con el gate **congelado** en v. `gatefix0.0` es el
  control de arquitectura correcto.

---

## 10. Trampas conocidas (no las redescubras)

- **`MAEViT.eval()` devuelve `None`** en este fork — nunca encadenar `.to(dev).eval()`.
- **`mask=False` se ignora**; para features sin enmascarar usar `encoder.mask_ratio = 0.0`.
  Los tokens salen **permutados en cada llamada** (69.3% de error elementwise entre dos
  llamadas fp32; 0.000% con semilla fija). Inocuo para cross-attn, pero invalida
  cualquier comparación elementwise sin fijar semilla.
- **No-determinismo de GPU** en atención: la misma config dio 2.79 y 2.51.
- **`max_keep_ckpts`** borra checkpoints en silencio.
- **Viewer:** usar `waymo_clean_view`, NO `waymo_clean` (los bins dispersos rompen el
  `reshape(64,2650)`).
- **`decoder_pos_embed`** estaba comentado en el `mae_neck.py` del repo base — sin eso
  el MAE no overfittea. Ya corregido.

---

## 11. Lo que NO se hizo y podría objetarse

- **No se probó multimodalidad** (predecir K trayectorias con probabilidades, como
  Wayformer/MTR de verdad). El decoder predice una sola trayectoria por objeto.
- **No se probó un decoder más chico.** Dado el hallazgo de §3.2 —que la capacidad
  del decoder domina— la pregunta obvia es si un modelo intermedio entre el MLP y el
  transformer de 2 capas generalizaría mejor. **No se corrió.**
- **No se escaló a 1000 sweeps / 50k escenas** (bloqueado por datos; requiere bajar
  más WOMD-LiDAR).
- **No se comparó voxel vs range-view con el protocolo riguroso actual.** La Fase 1
  (vóxeles) reportó +25% pero con un solo split y sin control de arquitectura — o
  sea, con el mismo defecto que §3.2 identifica. Esa comparación quedó sin rehacer.
- **El informe final todavía no está escrito.** Es el entregable pendiente.
