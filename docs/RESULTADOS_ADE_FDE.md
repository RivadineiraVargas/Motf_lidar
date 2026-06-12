# Resultados ADE/FDE — Evaluación comparativa MOTF

Métrica estándar de predicción de trayectorias, medida en metros (solo plano XY,
coords sensor). Dataset: WOMD-LiDAR, 10 escenas, 103 objetos válidos (tras filtro
de consistencia `max_jump=5.0m`). Split: 8 escenas train / 2 escenas val.

Horizonte actual: **0.5s** (5 frames pasado → 5 frames futuro a ~0.1s/frame).

## Tabla comparativa — augmentación (2026-06-10)

| Métrica | Baseline | MOTF (gated) | **MOTF + Augmentación** |
|---|---|---|---|
| Train ADE | 0.113 m | **0.092 m** | 0.127 m |
| Train FDE | 0.124 m | **0.110 m** | 0.161 m |
| **Val ADE** | 0.173 m | 0.289 m | **0.117 m** ✅ |
| **Val FDE** | 0.295 m | 0.440 m | **0.235 m** ✅ |
| Total ADE | 0.131 m | 0.151 m | **0.124 m** |
| Total FDE | **0.175 m** | 0.209 m | 0.183 m |

ADE = Average Displacement Error · FDE = Final Displacement Error
(✅ = mejor en datos no vistos)

**Lectura:** la augmentación por rotación (0/90/180/270° + flip XY) eliminó el
overfitting. Val ADE bajó de 0.173 (baseline) a 0.117 (-32%). La mejora vino de
regularizar el **decoder** (rama del histórico), no de la escena.

## Ablación del gate — ¿la escena LiDAR aporta? (2026-06-11)

Test limpio (Opción C): se entrenó una variante `use_gate=False` donde la rama de
escena está **siempre activa** y recibe gradiente completo (rompe el "candado del
gate", verificado: `scene_proj.grad` pasó de ~0 a 0.25). Misma augmentación.

| Métrica | Baseline | Gated+Aug (escena ≈OFF) | SinGate+Aug (escena ON) |
|---|---|---|---|
| Val ADE | 0.173 m | **0.117 m** | 0.118 m |
| Val FDE | 0.295 m | **0.235 m** | 0.286 m |
| Total ADE | 0.131 m | **0.124 m** | 0.131 m |

**Resultado decisivo:** encender la escena (SinGate, 0.118) da el **mismo** Val ADE
que apagarla (Gated, 0.117); en FDE incluso empeora (0.235 → 0.286).

→ **La escena LiDAR NO aporta valor predictivo a 0.5s de horizonte.** Confirmado
por dos vías independientes:
1. El gate, libre de decidir, se queda en 0 (ignora la escena).
2. Forzar la escena activa (sin gate) no mejora — incluso empeora el FDE.

No es un bug ni el candado de gradiente (que rompimos): es real. A horizonte corto
el movimiento es casi lineal y el histórico basta.

## Conclusión y próximo paso

Para que la escena importe se necesita **horizonte largo** (3-8s), donde aparecen
maniobras (giros, frenadas, interacciones entre agentes) que el histórico no puede
extrapolar. Eso requiere **datos limpios con horizonte largo**: re-extracción
WOMD-LiDAR con `track.id` (`utilities/save_point_cloud_data_fixed.py`).

## Reproducir

```bash
conda activate sapiens_gpu
cd sapiens/pretrain
python evaluate_ade_fde.py      # tabla comparativa de 4 vías
python diagnose_gate.py         # diagnóstico del gate (gradiente, gate forzado, linealidad)
```

Checkpoints:
- `work_dirs/baseline_multiescena/epoch_300.pth`
- `work_dirs/trajectory_attn_multiescena/epoch_300.pth` (gated, sin aug)
- `work_dirs/trajectory_attn_augmented/epoch_500.pth` (gated, con aug)
- `work_dirs/trajectory_attn_nogate/epoch_500.pth` (sin gate, con aug)
