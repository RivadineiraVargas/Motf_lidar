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

Ver [docs/CODEBASE_MAP.md](docs/CODEBASE_MAP.md) para la arquitectura, el flujo de
datos, las trampas y la guía de navegación. Ver `docs/EXPERIMENTOS_DECODER.md` para
los 18 experimentos con sus números y comandos de reproducción.
