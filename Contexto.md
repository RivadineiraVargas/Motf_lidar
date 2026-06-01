# MOTF — Moving Object Trajectory Forecasting

## Contexto do projeto

Dissertação de mestrado em informática (linha de pesquisa: veículos autônomos) no LCAD/UFES.
Desenvolvimento de um módulo de **predição de trajetórias de objetos móveis (MOTF)** para
veículos autônomos usando dados LiDAR 4D.

A abordagem adapta o modelo **Sapiens** da Meta (Masked Autoencoder + Vision Transformer,
originalmente para imagens) para processar grades de vóxels espaço-temporais geradas a
partir de nuvens de pontos LiDAR.

Pipeline em duas etapas:
1. Pré-treinamento auto-supervisionado do encoder via mascaramento temporal (MAE)
2. Fine-tuning supervisionado para predição de trajetórias futuras

## Ambiente

- **OS**: Ubuntu 20.04
- **Conda env**: `sapiens_final` (Python 3.10) — sempre ativar antes de rodar
- **Framework**: mmengine / mmpretrain (instalado from source, modo editable)
- **Stack**: PyTorch 2.5.1+cu118, mmcv 2.2.0, mmengine 0.10.7
- **Raiz do projeto**: `/home/lcad/lidar_sweep_viewer/sapiens/pretrain/`
- **Dataset**: Waymo Open — subsets em `/home/lcad/lidar_sweep_viewer/waymo_10/` (e futuramente waymo_100, waymo_1000)

## Estratégia experimental (orientação do tutor)

Escalar incrementalmente: **waymo_10 → waymo_100 → waymo_1000**.
Validar overfit e pipeline completo em cada nível antes de avançar.

- **Fase 1 (waymo_10)**: validar pipeline, comparar 3 modelos (baseline MLP, atenção sem
  pré-treino, atenção com pré-treino). EM ANDAMENTO.
- **Fase 2 (waymo_100)**: generalização, ablation freeze/unfreeze, curva de escala.
- **Fase 3 (waymo_1000)**: pré-treino robusto, métricas finais (minADE, minFDE), integração CARMEN.

## Arquivos principais

- `mmpretrain/datasets/lidar_sequence.py` — `LidarSequenceDataset` (pré-treino MAE)
- `mmpretrain/datasets/trajectory_dataset.py` — `TrajectoryDataset` (fine-tuning)
- `mmpretrain/models/backbones/mae_vit_4d.py` — `MAEViT4D` (encoder, herda de MAEViT)
- `mmpretrain/models/selfsup/mae_4d.py` — `MAE4D` (modelo MAE para pré-treino)
- `mmpretrain/models/heads/mae_head_4d.py` — `MAEPretrainHead4D` (reconstrução)
- `mmpretrain/models/trajectory_pred/baseline_model.py` — `BaselineTrajectoryModel` (MLP)
- `mmpretrain/models/trajectory_pred/trajectory_model_attn.py` — `TrajectoryModelWithAttention`
- `configs/sapiens_mae/lidar/mae_lidar_10_overfit.py` — config pré-treino
- `configs/sapiens_mae/lidar/trajectory_attn_overfit.py` — config fine-tuning atenção

## Parâmetros chave (waymo_10)

- `history_len = 5`, `pred_len = 5` (DEVEM ser iguais entre pré-treino e fine-tuning)
- `voxel_res = 2.0`, `spatial_range = [-10, 10, -10, 10, -2, 4]`
- `num_voxels = 10*10*3 = 300`
- `embed_dim = 1024`, `decoder_embed_dim = 512`, `mask_ratio = 0.75`
- 17 objetos com trajetória completa na cena `58d5f1b9e6a1a2f7`

## Correções já aplicadas (revisão diagnóstica)

Código original gerado com IA — auditoria completa encontrou e corrigiu:

1. **Imports circulares**: `backbones/__init__.py` importava `MAEViT` de arquivo
   inexistente; cadeia circular entre backbones/selfsup/models. Resolvido trocando
   imports via pacote pai por imports diretos em 5 arquivos selfsup (mae.py, maskfeat.py,
   mae_eva02.py, mae_sapiens2.py, simmim.py).
2. **Registro de módulos**: `trajectory_pred` não importado em `models/__init__.py`;
   `trajectory_pred/__init__.py` incompleto. Corrigido.
3. **Mascaramento no fine-tuning**: `TrajectoryModelWithAttention` aplicava 75% de
   mascaramento durante fine-tuning — modelo nunca via cena completa. Provável causa do
   ADE=4.16m. Corrigido com método `_encode_scene()` que bypassa mascaramento.
4. **Data leakage**: `TrajectoryDataset` normalizava usando média/std de toda a sequência
   (histórico + futuro). Corrigido para usar só o histórico.
5. **Voxelização lenta**: loop Python puro substituído por indexação numpy vetorizada.
6. **pos_embed perdendo gradiente**: recriado com `.to(device)` dentro do forward,
   virava tensor comum. Corrigido com `_ensure_pos_embed()`.
7. **Configs**: history_len inconsistente (10 vs 5), in_channels (768 vs 512),
   SelfSupDataPreprocessor de imagem para LiDAR, decoder sem in_chans=history_len.
8. **prints de debug**: removidos de forward(), __getitem__() e loops de treino.

## Detalhes técnicos importantes

- `MAEViT4D` usa `self.norm1` (NÃO `self.norm`), `self.layers`, `self.patch_embed`
- `.eval()` retorna `None` (bug do BaseModule do mmengine) — nunca atribuir o retorno:
  usar `model.eval()` em linha separada, não `m = Model().eval()`
- `MAEPretrainDecoder` projeta para `patch_size² * in_chans`; com patch_size=1 e
  in_chans=history_len → projeta para 5 frames (correto para LiDAR)
- `data_preprocessor = dict(type='BaseDataPreprocessor')` — neutro para LiDAR
- `mae.py` `train_step` espera `forward(mode='loss')` retornar tupla
  `(losses_dict, preds, masks)` — `MAE4D.forward` sobrescrito para isso

## Estado atual / próximo passo

ÚLTIMO PONTO: testando 2 épocas do pré-treino MAE para confirmar que o pipeline roda.
Erro mais recente era `not enough values to unpack (expected 3, got 1)` em `train_step` —
resolvido sobrescrevendo `MAE4D.forward` para retornar `(losses_dict, preds, masks)`.

PRÓXIMOS PASSOS:
1. Confirmar pré-treino de 2 épocas roda sem erro
2. Rodar pré-treino completo (4000 épocas) no waymo_10
3. Fine-tuning do modelo de atenção com encoder pré-treinado
4. Tabela comparativa: Baseline MLP vs Atenção s/ pretrain vs Atenção c/ pretrain
5. Visualização BEV (bird's eye view) com bounding boxes + trajetórias preditas
   sobre nuvem LiDAR real (pedido da tutora) — script visualize_bev_trajectories.py

## Deliverables pendentes para o tutor

- Material visual: vídeo mostrando predições sobre tomadas reais do Waymo (BEV com bboxes)
- Experimentos com mask_ratio=0.5 (atualmente 0.75)
- Bases com 10/100/1000 exemplos + overfit em cada
- Git: https://github.com/GabrielHendrix/lidar_sweep_viewer.git

## Convenções

- Comunicação em espanhol; comentários do código em espanhol/português misturados
- Trabalho incremental: validar cada componente com teste de sanidade antes de treino completo
- Sempre rodar testes rápidos (2 épocas) antes de experimentos longos