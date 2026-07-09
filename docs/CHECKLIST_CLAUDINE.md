# Checklist del pedido de Claudine (pedido_claudine.md)

Estado al 2026-07-08. Escala de trabajo actual: "10 lambidas" = 10 sweeps limpios
de waymo_clean (escena 2a81f...), protocolo con val intra-escena y escena 82f9
excluida como no-visto.

| # (Sec.13) | Paso | Estado | Evidencia |
|---|---|---|---|
| 1 | Revisar código del encoder | ✅ | bug `decoder_pos_embed` (mae_neck.py) y `train_step` encontrados y corregidos (commit e600dea) |
| 2 | Reducir arquitectura | ✅ | 24→6 capas, embed 1024→384 (~15× menor), itera en horas en RTX 4060 |
| 3 | Carga correcta de sweeps | ✅ | pipeline range-PNG 2650×1024 validado (formato colega) |
| 4 | Mascaramiento funcionando | ✅ | panel "enmascarado" en viz; mask_ratio 0.5 |
| 5 | Overfit 10 sweeps | ✅ | rv_rect_overfit10: loss 2.72→0.055 (6000 ép) |
| 6 | Overfit 100 sweeps | ✅ | rv_rect_overfit100: loss 2.07→0.244 (3000 ép); hallazgo: generalización pica ~ép1000 y luego memoriza |
| 7 | 1.000 sweeps | ❌ datos | hay 275 sweeps en disco; requiere descargar más WOMD-LiDAR |
| 8 | Consolidar visualización | ✅ | 5 paneles (original/enmascarado/recon/recon+visible/diferencia) × {train, val, no-visto} × {entrenada, sin entrenar} en recon_out/ |
| 9 | Aumentar dataset gradualmente | 🟡 | 10→100 hecho; 275 posible sin descarga |
| 10 | ≥50 mil sweeps | ❌ datos | fase de escala, tras luz verde |
| 11 | Evaluar en no-visto | ✅ | eval_rect_loss.py: sin entrenar 3.52 / 10sw 3.39 / 100sw 3.16 |
| 12 | Estudiar Wayformer y relacionados | ✅ | docs/ESTUDIO_WAYFORMER.md (decisiones de diseño para nuestro decoder) |
| 13 | Representación de trayectorias | ✅ mini | K=100 slots + flag validez + 16 desplazamientos ego (8s a 2Hz, formato WOMD) — train_decoder_mini.py |
| 14 | Diseñar decoder | ✅ mini | Wayformer condicionado: 2 bloques TransformerDecoder, cross-attn a tokens del MAE congelado, k=1 |
| 15 | Entrenar decoder anotado | ✅ mini | overfit escena 2a81 (11 muestras): ADE 0.17m / FDE 0.29m a 8s, validez 100% |
| 16 | Visualizador trayectorias | ✅ | BEV nuevo (bev_train_t10.png: pred calca GT, giros incl.) + viewer C++/Open3D previos |
| 17 | Eval sistema completo | ✅ mini | ADE/FDE + acc validez en train Y escena no-vista (ADE ~42m -> sin transferencia con 1 escena, mismo patrón que el encoder) |

## Comparaciones pedidas (Sec. 6) ya cubiertas por trabajo previo
- Formas de representar sweeps: voxel vs range-view comparado a 10 escenas
  (voxel ganó, docs/RESULTADOS_ADE_FDE.md).
- Tasas de mascaramiento / tamaños de encoder: pendiente para la fase 50k.

## Cuantitativo clave (loss L2 enmascarada, seed 0)

| Caso | Sin entrenar | 10 sweeps (ep6000) | 100 sweeps (ep1000) | 100 sweeps (ep3000) |
|---|---|---|---|---|
| train | 2.973 | 0.053 | 0.461 | 0.235 |
| val (misma escena) | 3.059 | 0.986 | **0.787** | 0.944 |
| no-visto (82f9) | 3.520 | 3.389 | **3.157** | 3.237 |

Lecturas: (1) el encoder aprende (56× vs red aleatoria en train); (2) generaliza
dentro de la escena; (3) para escenas nuevas hacen falta más datos, no más épocas
(entre ép.1000 y 3000 del run de 100, train mejora 2× pero val/no-visto empeoran).

## Mini-fase 2 (decoder) — hallazgo de datos

`objs_bbox` de waymo_clean tiene los 91 frames de labels (9s completos del WOMD)
aunque el LiDAR cubra solo 0..10 → el horizonte de 0.5s de la fase vieja era
autoimpuesto. El decoder mini ya predice el formato WOMD real: 8s de futuro
(16 waypoints a 2Hz) desde el sweep actual. Trampas del fork documentadas en
train_decoder_mini.py: `MAEViT.eval()` retorna None; `mask=False` se ignora
(usar `mask_ratio=0` para extraer features sin máscara).
