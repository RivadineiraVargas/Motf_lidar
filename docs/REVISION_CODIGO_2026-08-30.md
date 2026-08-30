# Revisión de código del 30/08/2026 — los 13 hallazgos, verificados uno por uno

`/code-review` sobre `master...encoder/jointmotion-finetune` (169 archivos) devolvió
13 hallazgos. **Los 13 se verificaron a mano antes de aceptarlos**, con lectura de
código y medición sobre datos reales. Doce son reales; uno es real pero no llegó a
afectar ningún número publicado.

Este documento existe porque un informe de revisión sin verificar es una lista de
sospechas, no de hechos — y este proyecto ya se equivocó nueve veces por dar algo
por cierto sin medirlo.

## Resumen

| # | hallazgo | verificado | estado |
|---|---|---|---|
| 6 | **la escena estaba desalineada en el tiempo** | 43% de los objetos | **ARREGLADO** — 0 desalineados |
| 2 | el pos-embed del decoder MAE quedaba en ceros | `requires_grad=False` | **ARREGLADO** — aprendible |
| 7 | el sincos 2D estaba transpuesto | numéricamente | **ARREGLADO** en la función |
| 5 | `max_jump` solo cubría la primera ventana | `globals_[:sequence_len]` | **ARREGLADO** — por ventana |
| 8 | `_geo` nunca se limpiaba entre lotes | leído | **ARREGLADO** |
| 11 | reanudar duplicaba filas del CSV | reproducido | **ARREGLADO** en los 6 scripts |
| 10 | `strict=False` sin verificar la carga | leído | **ARREGLADO** en ambos |
| 13 | `sequence_len` aceptado y sin usar | leído | **ARREGLADO** — vuelve el guard |
| 1 | la etiqueta de arquitectura se ignoraba | CSV con valores distintos | **ARREGLADO** (no afectó nada) |
| 4 | el cache no distinguía encoders | `{scene}.pt` | **ARREGLADO** |
| 12 | efecto nulo como `t=inf, p=0.0000` | reproducido | **ARREGLADO** |
| 3 | el mejor checkpoint se elige sobre el test | `if un[0] < best_ade` | mitigado — se registra también la época final |
| 9 | tres evaluadores miden con el objetivo recortado | 0 menciones de `clip_norm` | marcados OBSOLETO en su docstring |

---

## 6 — La escena está desalineada en el tiempo (el importante)

`trajectory_dataset.py:255`. `centers` se construye desde `object_tracks[obj_id]`,
que solo contiene los frames donde ese objeto fue etiquetado, ordenados. O sea que
`centers` se indexa **por posición en el track**. Pero el mismo índice se usa como
**número de frame absoluto** para cargar la escena:

```python
t0 = item.get('t_start', 0)
for i in range(t0, t0 + self.history_len):
    points = self.load_bin(os.path.join(scene_bin, f"{i}.bin"))
```

Un objeto que aparece por primera vez en el frame 6 recibe la escena de los bins
0 a 4. Con **11 frames de LiDAR en total**, ese desfase es una parte distinta de
la secuencia.

**Medido sobre las dos escenas de validación del fold 0:**

| | |
|---|---|
| objetos evaluados | 51 |
| con la escena desalineada | **22 (43%)** |
| desfases observados | hasta 6 frames |

Sobre cuatro escenas, **el 56% de los tracks no arranca en el frame 0** y el 37%
tiene huecos.

**Ocurre con `eval_windows=1`**, o sea en todos los experimentos del proyecto —
no solo en los que usan ventanas múltiples.

**Qué NO invalida:** el resultado de capacidad (−10%, 8/8 semillas). `baseline` y
`gate0` no usan la escena.

**Qué pone en duda:** la conclusión central. La escena puede no haber ayudado
porque, para casi la mitad de los objetos, **era la escena de otro momento**. Es un
mecanismo plausible, no una causa probada: hay que medirlo alineando y repitiendo
el fold 0.

**Por qué no está arreglado todavía:** hay una CV de 20 h en curso sobre este
mismo código. Cambiar el alineamiento a mitad haría que los folds 1-4 no fueran
comparables con el fold 0.

## 2 — El pos-embed del decoder MAE se queda en ceros

`mae_neck.py:80` declara `decoder_pos_embed` con `requires_grad=False`. Para los
300 tokens de vóxeles el sincos 2D da 290 contra 301, así que cae al `else`, que
registra *"se deja aprendible"*. **Ese mensaje es falso**: el parámetro no es
aprendible y se queda exactamente en `torch.zeros`. Cada token enmascarado entra
al decoder como `mask_token + 0`, indistinguible de cualquier otro: el decoder no
puede saber qué vóxel está reconstruyendo.

Es de la misma familia que el bug `decoder_pos_embed` documentado del MAE del
colega, reintroducido acá al asumir que era inofensivo. El comentario lo escribí yo
en f82caee y está equivocado.

## 7 — El sincos 2D está transpuesto respecto de los tokens

`build_2d_sincos_position_embedding` hace `torch_meshgrid(grid_w, grid_h)` con
indexado `ij`, o sea una grilla `(w, h)`: la fila *k* del pos-embed corresponde a
`(w=k//h, h=k%h)`. `patchify` emite los tokens en orden fila-mayor, `(h*w + w)`.

**Verificado numéricamente** en la grilla real de range-view (4×32): el token 1
debería recibir el código de `(w=1, h=0)` y recibe el de `(w=0, h=1)`. Cada token
recibe el código de otro parche, lo que destruye la noción de vecindad 2D que el
cambio quería aportar. En grillas cuadradas es una transposición al menos
consistente; en 4×32 no.

## 3 — El mejor checkpoint se elige sobre el propio conjunto de test

`train_decoder_mini.py:520`: `if un[0] < best_ade`, donde `un[0]` es el ADE de las
escenas retenidas. Ese mismo mínimo es lo que se escribe en los CSV. Selección de
modelo y reporte usan los mismos datos, así que todo ADE de ese track es un
mínimo-sobre-épocas de la métrica de test: sesgado hacia abajo, y **de forma
desigual entre arquitecturas** (la más ruidosa gana más con el mínimo).

Es el hallazgo H1 de la auditoría del 23/08. Fase 1 lo resolvió evaluando en época
FIJA 100; el track `decoder_mini` sigue emitiendo el número sesgado.

## 5 — `max_jump` solo cubre la primera ventana

`trajectory_dataset.py:145`: `seq_g = globals_[:self.sequence_len]`. El filtro
anti-corrupción mira los primeros `sequence_len` frames, pero con `eval_windows>1`
se emiten ventanas que llegan más lejos. Un track que se rompe después del primer
tramo pasa el filtro y sus frames corruptos entran en las ventanas extra —
justamente el salto de decenas de metros que el filtro existe para atrapar.

Afecta a las corridas evaluadas con `--eval-windows 7`: experimentos 16, 17 y 18.

## Los arreglados

**1 — la etiqueta de arquitectura se ignoraba.** `cross_validate_decoder.py:144`
pasaba `arch=model_arch`, que solo se fija en la rama de fine-tuning y por defecto
vale `'wayformer'`. Toda arquitectura se entrenaba como wayformer y, con la misma
semilla, `baseline` salía idéntica bit a bit. **Verificado que no afectó nada
publicado:** las 4 arquitecturas de `cv_results.csv` dan valores distintos en los
15 pares, o sea que ese CSV es anterior a la regresión (2d4da08, 29/07).

**4 y 10 — cache y carga de checkpoints.** El cache era `{scene}.pt`, sin
referencia al encoder: cambiar `--enc` y reusar el `--cache` servía features del
modelo viejo. Ahora la clave incluye una huella del checkpoint. Y `load_state_dict(
strict=False)` aceptaba en silencio un checkpoint que no casara en nada, dejando un
encoder aleatorio; ahora falla con un mensaje explícito.

**12 — `t=inf, p=0.0000` para un efecto nulo.** En `agregar_resultados.py`, cinco
folds con efecto exactamente cero se imprimían como el resultado más significativo
posible.

## Lo que hay que hacer cuando termine la CV

1. **Alinear la escena con la trayectoria** y repetir el fold 0. Es el experimento
   que decide si el resultado negativo del proyecto era real.
2. Arreglar el pos-embed (2) y el sincos (7) — los dos tocan el pre-entrenamiento,
   así que hay que reentrenar encoders.
3. Recalcular `max_jump` por ventana (5).
4. Limpiar `_geo` al inicio de cada `forward` (8).
5. Mover la evaluación dentro del guard de reanudación en los `run_*.sh` (11).


---

## Lo que cambió al arreglar el hallazgo 6 (medido)

Alinear la escena cambia el tamaño de los conjuntos, y hay que decirlo antes de
comparar cualquier número nuevo con uno viejo:

| conjunto | antes | ahora |
|---|---|---|
| entrenamiento (ventana 1) | 200 | **252** |
| validación (ventana 1) | 51 | **32** |
| validación (7 ventanas) | 319 | **198** |

El de entrenamiento **crece**: antes, un objeto cuya primera ventana estaba
corrupta o desalineada se descartaba entero; ahora se busca una ventana válida más
adelante en su track.

El de validación **se achica**: los objetos que aparecen después del frame 6 no
tienen ninguna ventana con escena alineada, porque su historia se saldría de los
11 sweeps de LiDAR. Son objetos que nunca se pudieron evaluar bien.

**Ningún número anterior al 30/08 es comparable con los nuevos.** Cambió el
alineamiento, cambió el conjunto de test y cambió el pre-entrenamiento (el
pos-embed pasó de ceros a aprendible, así que hay que reentrenar los encoders).

Prueba de humo tras los arreglos: `noclip_dec_fold0`, 2 épocas, pérdida
0,2949 -> 0,2441, 0,48 s/paso, 1435 MiB. Entrena sano.


---

## Segunda vuelta: la revisión encontró que MIS arreglos rompieron cuatro cosas

Se corrió `/code-review` sobre los arreglos. Encontró 12 hallazgos nuevos, cuatro
de ellos introducidos por mí al arreglar los anteriores. **Tres eran bloqueantes.**

### El crítico: mi arreglo del pos-embed era un no-op

Puse `requires_grad_(True)` dentro de `init_weights()`. Pero mmengine arma el
optimizador (`Runner.train` -> `build_optim_wrapper`) **antes** de llamar a
`_init_model_weights()`, y `DefaultOptimWrapperConstructor.add_params` descarta con
`continue` todo parámetro con `requires_grad=False`. **Verificado en el entorno
real**: el tensor nunca entraba al optimizador.

O sea que el pos-embed pasaba de ceros a ruido fijo, y el log seguía diciendo
"APRENDIBLE" — exactamente la misma mentira del hallazgo original. Los cinco
encoders de la CV se habrían pre-entrenado mal, sin que nada fallara ni avisara.
**Habría costado las 26 horas en silencio.**

Arreglado decidiendo `requires_grad` en `__init__`. Verificado: en vóxeles
`requires_grad=True` y **entra al optimizador**; en grilla cuadrada queda fijo con
el sincos, como corresponde.

### Los otros tres bloqueantes

**El guard de bash se saltaba la evaluación.** Mi patrón
`[ $NUEVO -eq 1 ] && python train || { echo fallo; continue; }` hace que, con el
checkpoint ya presente, el `&&` falle, se dispare el `||`, se imprima un fallo
falso y el `continue` saltee la evaluación. Probado: nunca evaluaba. Rompía la
reanudación peor que el bug original.

**`run_fase1_seeds.sh` usaba `$V` y `$S`**, que no existen dentro de esa función
—sus locales son `$VAR` y `$SEED`—, así que el guard recibía cadenas vacías y no
bloqueaba nada.

**`mae._motf_ckpt = str(ckpt)`** con el parámetro llamado `enc_ckpt`: `NameError`
en cada llamada. Y el atributo iba al objeto equivocado (`mae` en vez de
`mae.backbone`, que es el que recibe `_enc_id`), así que la clave de caché seguía
sin distinguir encoders.

### También arreglados en esta vuelta

- La contigüidad se verificaba solo en el histórico: **15 de 198 ventanas (7,6%)
  tenían hueco en el tramo futuro**, o sea un "futuro a 3 s" que abarcaba más
  tiempo. Ahora se exige contigüidad en toda la ventana.
- `n_lidar_frames` era un 11 hardcodeado que nunca se definía; ahora se cuenta del
  disco, porque con la alineación arreglada `f0` ya no es siempre 0.
- El guard daba por hecha una combinación con una sola fila, cuando el evaluador
  escribe una por escena: una evaluación muerta a mitad dejaba medio resultado.
- `encode_sweeps` había perdido su `@torch.no_grad()` al insertar la función nueva.
- `best_metrics`/`final_metrics` sin guard contra `None`.
- Los contadores del log decían "tracks" cuando ya contaban ventanas.

### Tamaños finales

| conjunto | original | tras alinear | tras exigir contigüidad |
|---|---|---|---|
| entrenamiento | 200 | 252 | **236** |
| validación (1 ventana) | 51 | 32 | **29** |
| validación (7 ventanas) | 319 | 198 | **183** |

Pruebas de humo tras todo: decoder 2 épocas (pérdida 0,2921, 0,46 s/paso) y
**encoder MAE 2 épocas** (pérdida 0,2755, grad_norm 4,89) — este último es el que
ejercita el pos-embed arreglado.

### La lección

Los arreglos necesitan la misma verificación que los hallazgos. Cuatro de mis doce
arreglos estaban mal, y el peor no habría fallado: habría entrenado feliz durante
26 horas produciendo un encoder inútil.
