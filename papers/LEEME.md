# Papers de referencia

Los cinco PDF que este proyecto cita en sus decisiones de diseño. Cada uno está
acá porque **cambió algo concreto del código**, no como bibliografía general.

| paper | qué nos aportó | dónde se ve |
|---|---|---|
| **Wayformer** (Nayakanti 2022) — *no está el PDF, ver el resumen* | historia relativa (no global), incertidumbre por waypoint, k>1 con winner-takes-all | `docs/ESTUDIO_WAYFORMER.md`, `trajectory_model_attn.py` |
| **WOMD-LiDAR** (Chen 2023) | el dataset: 11 frames de LiDAR = 1,1 s de historia contra 91 de etiquetas. Es el origen del cuello de 275 sweeps | experimento 22, `docs/ESTUDIO_WAYFORMER.md:38` |
| **GeoMAE** (Tian 2023) | reconstruir *ocupación* puede ser el objetivo equivocado; propone predecir centroides | `mae_head_4d.py` (`target='centroide'`), experimento 17 |
| **JointMotion** (Wagner 2024) | no congelar el encoder: usar el pre-entrenamiento como *inicialización* | `finetune_blocks` en `trajectory_model_attn.py` |
| **ReZero** (Bachlechner 2020) | un escalar residual iniciado en cero SÍ recibe gradiente; de ahí `gate_init=0,05` | experimento 20, trampa 4 del mapa |
| **BEVTraj** (Kong 2025) | predicción sin mapa HD, sobre nuScenes (20 s de LiDAR continuo contra nuestros 1,1 s) | experimento 21; **sin estudiar a fondo** |

**Pendiente de estudiar:** BEVTraj. Resuelve nuestro mismo problema y su elección
de dataset podría importar más que cualquier arquitectura — ver "La ruta" en
`docs/CODEBASE_MAP.md`.

Los papers de Wayformer, MTR, DenseTNT y MotionLM no están como PDF; su lectura
está resumida en `docs/ESTUDIO_WAYFORMER.md` con las decisiones que salieron de
cada uno.
