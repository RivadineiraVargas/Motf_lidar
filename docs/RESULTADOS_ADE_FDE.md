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

## Reproducir (experimentos con encoder viejo, waymo_10)

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

---

# FASE 1 — Datos limpios (waymo_clean) + encoder re-pretrenado (2026-06-15)

Protocolo de Claudine (10 → 100 → 1000). FASE 1 = 10 escenas limpias (8 train /
2 val), horizonte **3s**, encoder MAE **re-pretrenado en las 8 escenas de train**
(antes era 1 sola escena vieja). Ver `docs/BUGS_DATOS.md` por qué los datos viejos
estaban sucios, y `docs/NEXT_SESSION.md` por el pipeline.

## Tabla comparativa (horizonte 3s, encoder nuevo)

| Métrica | Baseline | Gated | **SinGate** |
|---|---|---|---|
| Train ADE | **0.667 m** | 0.875 m | 0.730 m |
| Val ADE | 2.013 m | 2.303 m | **1.492 m** ✅ |
| Val FDE | 2.417 m | 2.808 m | **1.877 m** ✅ |
| Total ADE | 0.857 m | 1.077 m | **0.838 m** ✅ |
| Total FDE | 1.060 m | 1.300 m | **1.008 m** ✅ |

## Hallazgo principal: la escena LiDAR YA APORTA, y con fuerza

```
Val ADE:  2.013 (baseline) → 1.492 (SinGate)  =  -26%
Val FDE:  2.417 (baseline) → 1.877 (SinGate)  =  -22%
```

Evolución del beneficio de la escena en val (escenas no vistas):

| Configuración | Beneficio de la escena |
|---|---|
| 0.5s, encoder viejo (1 escena) | inútil (igual/peor que baseline) |
| 3s, encoder viejo (1 escena) | +4% (señal débil) |
| **3s, encoder NUEVO (8 escenas limpias)** | **+26% (claro)** |

→ Confirma DOS hipótesis juntas: la escena necesita (1) **horizonte largo** para
que haya maniobras, y (2) un **encoder bien pre-entrenado** en datos limpios.

## El gate cerrado y su ARREGLO (Prioridad 1 — 2026-06-16)

Con `gate_init=0` el modelo gated mantenía el candado de gradiente
(`tanh(scene_gate) = -0.0047`, la rama de escena nunca aprendía). **Fix:** parámetro
`gate_init` en el modelo — arrancar el gate en `tanh=0.5` da gradiente real a la rama
desde la época 1, rompiendo el candado, pero manteniendo el gate APRENDIBLE.

Resultado (`clean10_gated_init.py`, gate_init=0.5):

| Métrica | Baseline | **Gated (init 0.5)** | SinGate |
|---|---|---|---|
| Val ADE | 2.013 m | **1.303 m** ✅ | 1.492 m |
| Val FDE | 2.417 m | **1.733 m** ✅ | 1.877 m |
| Total ADE | 0.857 m | **0.767 m** ✅ | 0.838 m |
| Total FDE | 1.060 m | **0.893 m** ✅ | 1.008 m |

- El gate **aprendió y se quedó abierto** en `tanh=0.20` (no colapsó a 0).
- Es el MEJOR de los tres: **Val ADE -35% vs baseline** (mejor que SinGate -26%).
- Un peso de escena moderado (0.20) regulariza mejor que forzarla al 100% (SinGate).

→ **Contribución limpia:** el modelo aprende solo cuánto pesar el contexto LiDAR.

## Próximo paso del protocolo

Gated init (1.303) < baseline (2.013) en val → **se cumple la condición de escalar**.
Antes de escalar (protocolo 10→100→1000), completar con 10: curva multi-horizonte,
incerteza, informe (ver memoria project_prioridades). Luego FASE 2 = waymo_100.

## Reproducir Fase 1

```bash
conda activate sapiens_gpu
cd sapiens/pretrain
bash run_next_session.sh            # MAE → encoder → baseline+gated+nogate → eval
python evaluate_clean10_newmae.py   # solo evaluar si ya están los checkpoints
```
