# MOTF — Guía de estructura y organización del repositorio

Documento de referencia del estado del proyecto tras la limpieza/orden de
2026-06-06. Para el contexto técnico (arquitectura, resultados, bug de datos),
ver [`AVANCES.md`](AVANCES.md).

---

## Ramas (branches)

| Rama | Propósito |
|---|---|
| **`master`** | Fuente de verdad. Modelo gated + limpieza de datos + visualizaciones. |
| `bugs` | Corrección de la extracción (`track.id` + horizonte) + soporte multi-escena (split train/val). |
| ~~`main`~~ | Rama vacía no relacionada (artefacto de GitHub) — a borrar. |

Flujo recomendado: ramas cortas → PR → merge a `master`. No dejar `master` atrás.

---

## Entornos conda

| Entorno | Uso | Notas |
|---|---|---|
| **`sapiens_gpu`** | Entrenamiento e inferencia | PyTorch 2.5.1+cu121, usa la GPU (RTX 4060). **Usar este.** |
| `sapiens_final` | Respaldo CPU | PyTorch CPU. Intacto como fallback. |
| `lidar_mae` | Visualización 3D | Tiene `open3d` 0.19. |

---

## Scripts canónicos (qué usar para qué)

Todo en `sapiens/pretrain/` salvo el de extracción.

### Entrenamiento
- `tools/train.py <config>` — entrena con mmengine
- Configs en `configs/sapiens_mae/lidar/`:
  - `trajectory_attn_multiescena.py` / `baseline_multiescena.py` — experimento multi-escena (rama `bugs`)
  - `baseline_overfit_500.py`, `trajectory_attn_overfit.py` — single-scene

### Exportar predicciones
- `export_predictions_npz.py <attn|baseline>` — para los visualizadores Python
- `export_predictions_global.py <attn|baseline>` — para el visor C++ (coords globales)

### Visualización (3 estilos)
- `viz_dashboard_cpp_style.py` — BEV estilo show_point_cloud (PNG, cv2)
- `viz_3d_open3d.py` — vista 3D Open3D (correr en `lidar_mae`)
- `visualize_bev_trajectories.py` — BEV matplotlib + **video MP4**
- `show_point_cloud` (C++, raíz) — visor interactivo con predicciones (tecla `t`)

### Re-extracción de datos (cuando se consigan los tfrecords)
- `utilities/save_point_cloud_data_fixed.py` — extracción corregida (`track.id` + horizonte completo). Ver [`AVANCES.md`](AVANCES.md).

---

## Limpieza realizada (2026-06-06)

Se eliminaron **10 scripts de visualización obsoletos** que referenciaban
checkpoints/work_dirs viejos (`epoch_200.pth`, `trajectory_attn_overfit`,
`baseline_overfit_norm`), superados por los 3 visualizadores actuales:
`visualizar_com_poses`, `visualizar_final`, `visualizar_resultado`,
`visualizar_zoom`, `visualize_one`, `visualize_trajectory`,
`visualize_baseline_results`, `visualize_scene_3d`, `animar_trajetoria`,
`presentacion_final`.

`.gitignore` excluye: datasets (`waymo_*`), checkpoints (`work_dirs/`, `*.pth`),
binarios C++, logs de entrenamiento y salidas generadas (`*.npz`, `*.mp4`,
`predictions_global.txt`).

---

## Datos

- Dataset: **Waymo WOMD-LiDAR** (64 beams). Carpeta `waymo_10/` (no versionada).
- Links de descarga (Drive) de waymo_10/100/1000 en el [`README.md`](../README.md).
- Estado: 10 escenas usables, 103 objetos válidos tras el filtro de consistencia.
- Para escalar/horizonte largo: re-extraer del WOMD-LiDAR oficial con
  `save_point_cloud_data_fixed.py`.
