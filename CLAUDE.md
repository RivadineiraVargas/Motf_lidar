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

**Estado al 02/09/2026.** Tres resultados independientes cierran la línea de
variantes de arquitectura: la escena LiDAR no aporta (0/5 folds), la capacidad va
en la misma dirección en los cinco folds pero no llega a significancia (p=0,102), y
la historia completa de 1,1 s no mejora y sobreajusta (1/5 folds). El encoder **sí**
generaliza, así que no es eso. El cuello está medido: 236 ventanas de entrenamiento
desde 8 escenas, y un MAE pre-entrenado con 8 muestras.

Pero que el cuello sea de datos **no autoriza a ir a buscarlos** (regla 5). El
peldaño vigente es **10-100 sweeps, con 275 en disco, y su resultado todavía no es
bueno**: a 100 sweeps el encoder pica cerca de la época 1000 y después memoriza, y
10× de datos dieron solo −6,8 %. **El trabajo es mejorar ahí**, no escalar — ver
"La ruta" en el mapa, que arranca con la puerta.

Ver [docs/CODEBASE_MAP.md](docs/CODEBASE_MAP.md) para la arquitectura, el flujo de
datos, las 24 trampas, la guía de navegación y **la ruta**. Ver
`docs/EXPERIMENTOS_DECODER.md` para los 22 experimentos con sus números y comandos
de reproducción.
