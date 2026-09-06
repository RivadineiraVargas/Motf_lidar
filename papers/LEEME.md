# Papers de referencia

Los PDF que este proyecto cita en sus decisiones de diseño. Casi todos están acá
porque **cambiaron algo concreto del código**, no como bibliografía general. La
excepción es el survey de Madjid, que sirve para ubicar el trabajo en el campo.

| paper | qué nos aportó | dónde se ve |
|---|---|---|
| **Wayformer** (Nayakanti 2022) — *no está el PDF, ver el resumen* | historia relativa (no global), incertidumbre por waypoint, k>1 con winner-takes-all | `docs/ESTUDIO_WAYFORMER.md`, `trajectory_model_attn.py` |
| **WOMD-LiDAR** (Chen 2023) | el dataset: 11 frames de LiDAR = 1,1 s de historia contra 91 de etiquetas. Es el origen del cuello de 275 sweeps | experimento 22, `docs/ESTUDIO_WAYFORMER.md:38` |
| **GeoMAE** (Tian 2023) | reconstruir *ocupación* puede ser el objetivo equivocado; propone predecir centroides | `mae_head_4d.py` (`target='centroide'`), experimento 17 |
| **JointMotion** (Wagner 2024) | no congelar el encoder: usar el pre-entrenamiento como *inicialización* | `finetune_blocks` en `trajectory_model_attn.py` |
| **ReZero** (Bachlechner 2020) | un escalar residual iniciado en cero SÍ recibe gradiente; de ahí `gate_init=0,05` | experimento 20, trampa 4 del mapa |
| **BEVTraj** (Kong 2025) | predicción sin mapa HD, sobre nuScenes (20 s de LiDAR continuo contra nuestros 1,1 s) | experimento 21; **sin estudiar a fondo** |
| **Survey** (Madjid 2025) | ubica el nicho: de ~350 métodos, **solo 2 usan LiDAR crudo** y NINGUNO combina LiDAR crudo + auto-supervisión + predicción. Tampoco advierte sobre minADE | `docs/EXPERIMENTOS_DECODER.md`, sección tras la tabla de papers |

**Cuidado con una cita del survey de Madjid.** La frase *"the shortage of
self-supervised solutions"* NO es de esos autores: describe el survey de Teeti
et al. [7] dentro de un párrafo que resume trabajos ajenos. Se detectó al
verificar el PDF con `pdftotext`; la versión HTML entrega la frase sin el contexto
de quién la dice. **Las citas que vayan a la tesis se verifican en el PDF.**

**Pendiente de estudiar:** BEVTraj. Resuelve nuestro mismo problema y su elección
de dataset podría importar más que cualquier arquitectura — ver "La ruta" en
`docs/CODEBASE_MAP.md`.

Los papers de Wayformer, MTR, DenseTNT y MotionLM no están como PDF; su lectura
está resumida en `docs/ESTUDIO_WAYFORMER.md` con las decisiones que salieron de
cada uno.
