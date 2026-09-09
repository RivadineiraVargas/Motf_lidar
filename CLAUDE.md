# MOTF — predicción de trayectorias a partir de LiDAR

Tesis de maestría (LCAD/UFES). Adapta un ViT de 302,6 M (etiquetado `sapiens_0.3b`,
que es ViT-Large) con pre-entrenamiento MAE sobre LiDAR de Waymo, para responder:
**¿la escena LiDAR auto-supervisada aporta a la predicción de trayectorias sobre un
baseline puramente cinemático?**

**Stack:** PyTorch + mmengine/mmpretrain (fork de Sapiens de Meta), CUDA en una
RTX 4060 Laptop de 8 GB. Visor propio en C++/OpenGL. Entornos conda: `sapiens_gpu`
(entrenamiento), `waymo_env` (extracción de datos).

**Estructura:** el repositorio es un fork completo de Sapiens; el código propio son
167 archivos — el visor C++ en la raíz, `utilities/` (pipeline de datos),
`sapiens/pretrain/*.{py,sh}` (experimentos), y las capas MOTF dentro de
`sapiens/pretrain/mmpretrain/{datasets,models}/`. Todo lo demás es mmpretrain sin
modificar.

**Comunicación en español.**

## Reglas que este proyecto aprendió a la fuerza

1. **Nunca citar un número de memoria.** Los resultados crudos están en
   `sapiens/pretrain/work_dirs/*/*results*.csv`. Recalcular desde ahí, siempre.
2. **Ninguna medición es confiable con una sola semilla ni un solo fold.** Ya produjo
   ocho conclusiones falsas que hubo que retractar. Antes de presentar cualquier
   número, decir sobre cuántas semillas, cuántos folds y qué población se promedia.
3. **Distinguir los tres tracks** (`waymo_10` muerto, decoder_mini congelado, Fase 1
   vigente). Mezclar sus números es un error ya cometido.
4. **El visor usa `waymo_clean_view`, nunca `waymo_clean`.**
5. **La escalera 10 → 100 → 1000 es una PUERTA, no un itinerario.** La unidad es
   **SWEEPS** (frames de LiDAR), no escenas — `pedido_claudine.md` Sec. 13, resumido
   en `docs/CHECKLIST_CLAUDINE.md`; los documentos que dicen "escenas" están
   equivocados. No se sube a 100 sin un buen resultado en 10, ni a 1000 sin uno
   bueno en 100, y **CARMEN_LCAD no se toca hasta estar trabajando con 1000**.
   Antes de proponer cualquier aumento de datos: decir en qué peldaño estamos y por
   qué su resultado ya es bueno. Ya me desvié de esto cuatro veces.

**Estado al 09/09/2026.** Dos resultados cierran la semana, los dos negativos.

**Exp. 31 — la representación queda descartada.** Range-view a resolución nativa
(2.650 columnas azimutales, 660 tokens, contra 300 vóxeles de 2 m) da
**+0,351 · p=0,557 · 3/5 folds**: la escena tampoco aporta. El fold 0 daba −1,152
con 8/8 semillas y engañó tres días; los folds 3 y 4 lo dieron vuelta con 0/8
semillas cada uno. Lo único que sobrevive es que el gate se **abre** a 0,008-0,013
en los cinco folds (contra ~0,003 en vóxeles): la primera representación que el
modelo no apaga del todo — y aun así predice peor.

**Exp. 32 — el mejor resultado del proyecto era un artefacto.** `gate0` le ganaba al
baseline cinemático por −0,217 en 5/5 folds y se leía como "la arquitectura aporta
capacidad". No: los dos modelos tienen el **mismo** MLP, el mismo dataset y los
mismos hiperparámetros, y difieren solo en que `gate0` recibe 64 columnas que valen
**exactamente cero** — y `nn.Linear` inicializa con cota `1/sqrt(in_features)`, así
que las mismas entradas entran con pesos 2,3× más chicos. Pegándole al baseline 64
columnas de ceros se reproduce la ventaja: **−0,260 en 5/5 folds, r=+0,991 fold por
fold**, residuo +0,042. **No hay ninguna configuración en el proyecto que le gane al
baseline cinemático por una razón distinta de la escala de inicialización.**

**Exp. 33 — el artefacto no aguanta ni cambiar k, y aparece lo unico solido.** A
k=6 la ventaja de la inicializacion se evapora (−0,053, p=0,758, 3/5): el fold 4 la
dio vuelta, el mismo que dio vuelta el exp. 31. Pero el brazo secundario dio el
resultado mas firme de la semana: **k=6 empeora el ADE real en los 5 folds**
(+0,510, p=0,0055, 0/5) mientras el minADE_6 del oraculo "mejora" 26 %. Y ahora
esta medido POR QUE: sobre los 265 objetos de las 10 escenas, la dispersion entre
los 6 modos a 3 s es de **16,7 m contra 5,97 m de recorrido real** (2,8x), con
mediana de recorrido **0,00 m** porque la mitad de los objetos estan detenidos. Los
seis modos no son seis futuros plausibles: son seis tiros. Ver
`docs/figuras/k6_caja_y_modos.png`.

**Los tres eslabones, cerrados dos:** el **encoder** funciona y no es el cuello
(exps. 21, 27, 30: un encoder 2-5× mejor da la misma predicción); la
**representación** está descartada por dos vías independientes (exp. 29 densidad,
exp. 31 range-view nativo); el **consumo** —una sola query de cross-attention
comprimida a 64 dims— **no se tocó en 32 experimentos**. Es el único sospechoso que
queda.

**Lo defendible hoy no es un número de predicción sino lo metodológico:** minADE_6
mejora 29 % (p=0,003, 5/5) mientras el ADE real empeora en 0/5 folds —la métrica de
la literatura premia lo que empeora la predicción, y un survey de ~350 métodos no lo
advierte—; doce retractaciones por reportar con una semilla o un fold; y el
artefacto de inicialización, que se destapó preguntando **"¿por qué exactamente
sería mejor?"** en vez de "¿es significativo?".

El cuello de datos sigue en pie —236 ventanas desde 10 escenas, un MAE con 8
muestras— pero ya no es la única explicación disponible.

Pero que el cuello sea de datos **no autoriza a ir a buscarlos** (regla 5). El
peldaño vigente es **10-100 sweeps, con 275 en disco, y su resultado todavía no es
bueno**: a 100 sweeps el encoder pica cerca de la época 1000 y después memoriza, y
10× de datos dieron solo −6,8 %. **El trabajo es mejorar ahí**, no escalar — ver
"La ruta" en el mapa, que arranca con la puerta.

**Y agregar semillas no es el camino:** en el exp. 31 el ruido de semilla explica el
**8 %** de la varianza entre folds. De 8 a 16 semillas el error estándar baja un 2 %
y cuesta 39 h de GPU. Lo que da poder son **folds**, y harían falta 10.

Ver [docs/CODEBASE_MAP.md](docs/CODEBASE_MAP.md) para la arquitectura, el flujo de
datos, las 36 trampas, la guía de navegación y **la ruta**. Ver
`docs/EXPERIMENTOS_DECODER.md` para los 33 experimentos con sus números y comandos
de reproducción.
