# Sesión: Validación del Encoder MAE (enfoque del colega) — estado al apagar

**Rama:** `encoder/validacao-mae` · **Fecha:** 2026-06-24

Documento de continuación para retomar sin perder contexto. Resume el plano de
Claudine, el análisis del repo del colega, lo construido, y los 2 bloqueos.

---

## 1. Contexto: el plano de Claudine (autoritativo)

Claudine elaboró un plano (`pedido_claudine.md`) que **reencuadra** el proyecto MOTF.
Puntos clave:
- **"10/100/1000" = SWEEPS (frames), NO escenas.** La validación pequeña es el overfit
  del ENCODER sobre N sweeps individuales.
- **Prioridad #1 URGENTE = el ENCODER (MAE)**, no el decoder. Demostrar overfit en
  10/100/1000 sweeps + **visualización de reconstrucción** (original/máscara/reconstruido,
  train/val/no-visto) + diferencia entrenado vs no-entrenado.
- **Reducir la arquitectura** (Sec.4): de 24 capas a 5-10 bloques.
- **Decoder estilo WAYFORMER** (Fase 2, multi-objeto, hasta 100 trayectorias).

## 2. Análisis del repo del colega (Gabriel Hendrix)

En `lidar_hendrix/lidar_sweep_viewer-master.zip`. Su metodología:
- **Range-view como PNG 2650×1024** (los 64 beams escalados a 1024 → "rayas"), gris invertido.
- Usa el **MAE de imágenes de Sapiens ESTÁNDAR** (off-the-shelf), NO ViT-4D custom.
  Base `mae_vit-base-p16.py`, `CustomDataset` (carpeta de imágenes), `MAEViT`+`MAEPretrainDecoder`+`MAEPretrainHead`.
- Patches **16×25** → 6784 tokens. Modelo **completo** (sapiens_0.3b, 24 capas). **Desde cero**
  (sin `load_from`). **18000 épocas**, lr 1e-5, mask **0.5**, `norm_pix=False`.
- Genera el PNG con `utilities/show_rangeview_and_birdview.py` (función `normalizar`:
  `255 - range*100/resolution`, gris→BGR). **También genera BEV** en el mismo script.
- Visualización: `demo/run_mae_reconstruction.py` (MAEInferencer) → 4 paneles
  (original | enmascarado | reconstruido | recon+visible).
- Su `mmpretrain/models/selfsup/mae.py` está modificado (~52 líneas) para parches
  rectangulares (`patch_height/patch_width`, `img_width/img_height`). Cambio contenido/portable.

## 3. Plan acordado (lo que estamos implementando)

Adoptar el **pipeline del colega** (MAE de imágenes off-the-shelf) + **datos LIMPIOS nuestros**
+ **arquitectura REDUCIDA** (Claudine). Nota honesta: para el ENCODER, "limpio" NO da mejor
reconstrucción (los bugs eran en trayectorias); el premio de limpio es la Fase 2.

## 4. Lo que YA se construyó (en disco, persiste el reinicio)

- `utilities/range_npy_to_png.py` — convierte nuestros `.npy` → PNG formato colega. **Validado:**
  genera 2650×1024 con las "rayas" como el colega.
- **Datos generados:** `waymo_clean/range_png/train/` (10 PNG, de los `range_files/*.npy`).
- `sapiens/pretrain/configs/sapiens_mae/lidar/config_rangeview_overfit10.py` — config del MAE
  de imágenes ESTÁNDAR + arquitectura reducida (6 capas, embed 384, img 512 cuadrada,
  patch 16, mask 0.5, `CustomDataset` sobre nuestras PNG). Fix de `num_patches` del decoder ya aplicado.
- Componentes estándar verificados presentes: `MAE`, `MAEViT`, `MAEPretrainDecoder`,
  `MAEPretrainHead`, `CustomDataset`.
- (De antes) `mae_sweep_overfit.py` + `viz_mae_reconstruction.py` — MI enfoque reducido
  (MAEViT4D), que YA overfittea 10 sweeps: MSE entrenado 0.026 vs no-entrenado 0.405 (16×).
  Sirve como respaldo funcional.

## 5. LOS 2 BLOQUEOS al apagar (retomar aquí)

### Bloqueo 1 — CUDA no inicializa (de la máquina) → POR ESO se reinicia
`CUDA available: False` / "CUDA unknown error" pese a `nvidia-smi` sano. Estado del driver
por muchas corridas + timeouts. **El reinicio debería arreglarlo.** Verificar con:
`conda run -n sapiens_gpu python -c "import torch; print(torch.cuda.is_available())"`

### Bloqueo 2 — código: `mae.py:266 train_step` (lo arreglo yo)
`ValueError: not enough values to unpack (expected 3, got 1)` en
`mmpretrain/models/selfsup/mae.py` línea ~266. El `MAE.train_step` (estándar de nuestro repo)
fue modificado para esperar una tupla de 3 (losses, preds, masks) para visualización, pero el
flujo de loss devuelve 1. **Mismo patrón que el `MAE4D` de antes.** Hay que ajustar el
`train_step` (o el retorno del loss) para que devuelva la tupla esperada, o desacoplar la viz.

## 6. Cómo retomar (orden)

1. Confirmar CUDA: `python -c "import torch; print(torch.cuda.is_available())"` → True
2. Arreglar Bloqueo 2 (`mae.py` train_step / tupla).
3. Smoke test: `python tools/train.py configs/sapiens_mae/lidar/config_rangeview_overfit10.py`
   (con max_epochs bajo) → confirmar que la loss baja (overfit).
4. Overfit completo 10 sweeps → adaptar `viz_mae_reconstruction.py` o usar `run_mae_reconstruction`
   del colega para la viz 4 paneles.
5. Escalar a 100 y 1000 sweeps (regenerar PNG con `--max 100/1000`; para 1000 bajar más lidar).
6. Consolidar visualización (Claudine Sec.5), arreglar logging del loss.

## 7. Ramas del repo (GitHub)
- `master`, `bugs`, `datos-limpios/fase1-waymo10` (track voxel: gate −35%, incerteza, informe),
  `rangeview/fase1` (track range-view), **`encoder/validacao-mae`** (ESTA — encoder + colega).

## 8. Estado de resultados previos (no se pierde, ya documentado)
- Voxel ganó a range-view a 10 escenas (Val ADE 1.303 vs 1.685) — ver `docs/RESULTADOS_ADE_FDE.md`.
- Gate arreglado (gate_init), multi-horizonte, incerteza calibrada, informe PDF (`docs/INFORME_FASE1.md`).
