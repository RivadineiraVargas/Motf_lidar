# Próxima sesión — FASE 1 del protocolo de Claudine (waymo_10)

Protocolo de la tutora: **10 escenas → 100 → 1000**. Esta es la FASE 1 (10 escenas
limpias, 8 train / 2 val). Todo preparado y **validado por smoke test**. Para
arrancar (de noche, máquina prendida y sin suspender):

```bash
conda activate sapiens_gpu
cd /home/lcad/lidar_sweep_viewer/sapiens/pretrain
bash run_next_session.sh
```

Tiempo total ~1.5h. El log queda en `/tmp/next_session.log`.

## Qué hace el pipeline (4 pasos, encadenados con verificación)

1. **Re-pretrena el MAE** en las 8 escenas de train (`mae_clean10_pretrain.py`,
   1000 épocas, ~40min). Encoder con features de escena RICOS (el anterior fue
   en 1 sola escena vieja). NO ve las 2 de val (sin contaminación).
2. **Extrae el encoder** → `work_dirs/mae_encoder_clean10.pth`
   (`extract_mae_encoder.py`).
3. **Entrena** baseline + trajectory gated + nogate con el encoder nuevo
   (`clean10_baseline.py`, `clean10_gated_newmae.py`, `clean10_nogate_newmae.py`).
4. **Evalúa** ADE/FDE (`evaluate_clean10_newmae.py`) y compara contra baseline.

## Las 10 escenas (subconjunto de las 25 limpias)
- Train (8): 2a81f5233075e987, 2e41fe6faf5cd2ea, 367b072edc9822ea, 394e61f27c2a1700,
  4014ae5bcda2726f, 41692b0ec7ff4123, 4a2ef30000d19d90, 4b60f9400a30ceaf
- Val (2): 7e2f727866c69ea0, 82f90331a1dfe968

## Después de la Fase 1: escalar a 100, luego 1000
Bajar más LiDAR (ver flujo abajo) y repetir con configs análogos. El protocolo
busca demostrar que la escena aporta MÁS a medida que crece el dataset.

## La pregunta que responde

Con el encoder viejo (1 escena), a 3s la escena daba solo +4% sobre baseline
(ver `RESULTADOS_ADE_FDE.md`). **¿Un encoder pre-entrenado en 25 escenas limpias
hace que la escena aporte más?** Si SinGate_newmae mejora claramente sobre
baseline → la hipótesis de la escena se confirma con fuerza → contribución real.

## Validado antes de dejarlo listo
- Smoke test MAE: corre, loss baja 1.71→0.22, 5GB VRAM, ~4s/época.
- `extract_mae_encoder.py`: produce encoder de 294 keys, cargado OK por el
  trajectory model (history_len=5, compatible).

## Datos
- `waymo_clean/` — 25 escenas con LiDAR (track.id + horizonte 9s), 492 con bbox.
- Para escalar: bajar más LiDAR con el flujo de `utilities/` (ver abajo).

## Cómo se obtuvieron los datos limpios (para escalar después)
```bash
export PATH="/home/lcad/google-cloud-sdk/bin:$PATH"   # gcloud + gsutil
# 1. bajar un shard de scenario (430MB):
gsutil cp gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/scenario/training/training.tfrecord-NNNNN-of-01000 waymo_raw/scenario/
# 2. listar scene_ids:  conda run -n waymo_env python utilities/list_scene_ids.py <shard> --limit N
# 3. bajar LiDAR por escena (LOOP, no 'gsutil cp -I' que tiene un bug que para en 2):
#    for sid in ...; do gsutil cp gs://.../lidar/training/${sid}.tfrecord waymo_raw/lidar/; done
# 4. extraer:  conda run -n waymo_env python utilities/save_point_cloud_data_fixed.py \
#                 --tfrecord waymo_raw/lidar/ --scenario <shard> --root waymo_clean/ --max_traj_frames 91
```
