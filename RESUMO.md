# MOTF — predição de trajetórias a partir de LiDAR

Dissertação de mestrado (LCAD/UFES). Adapta um ViT de 302,6 M de parâmetros
(rotulado `sapiens_0.3b`, que na verdade é um ViT-Large) com pré-treinamento MAE
auto-supervisionado sobre LiDAR do Waymo, para responder a uma única pergunta:

> **A cena LiDAR auto-supervisionada acrescenta algo à predição de trajetórias,
> em comparação com uma baseline puramente cinemática?**

**Resposta, em 09/09/2026: não.** E, mais do que isso, o melhor resultado que
tínhamos também não era real.

Este documento resume 32 experimentos. Os números brutos estão em
`sapiens/pretrain/work_dirs/*/*results*.csv`; o detalhe de cada experimento, com
comandos de reprodução, está em [`docs/EXPERIMENTOS_DECODER.md`](docs/EXPERIMENTOS_DECODER.md).

---

## Como os números devem ser lidos

Todas as métricas são **ADE de objetos móveis**, em metros, com média ponderada
pelo número de objetos de cada cena. Toda comparação é **pareada por (fold,
semente)**, e o teste estatístico usa **n = 5 folds** — nunca o número de corridas.

Isso não é preciosismo: **doze conclusões deste projeto tiveram de ser retratadas**
por terem sido reportadas com uma única semente ou um único fold. A variância entre
folds é real e grande, porque os folds contêm cenas genuinamente diferentes.

---

## 1. O efeito da cena

Braço tratado (cena ligada) menos seu próprio controle (mesma arquitetura, cena
desligada). Negativo = melhor.

| experimento | efeito | p | folds a favor |
|---|---|---|---|
| cena, caixa centrada no ego | **+0,742** | 0,035 | 0/5 |
| cena com `gate_init=0,05` | +0,285 | 0,129 | 0/5 |
| 28 — caixa centrada no objeto | +0,047 | 0,618 | 2/5 |
| 29 — densidade em vez de ocupação binária | +0,031 | 0,793 | 3/5 |
| 30 — encoder retreinado com densidade | +0,038 | 0,737 | 3/5 |
| 31 — range-view em resolução nativa | +0,351 | 0,557 | 3/5 |

A cena passou de **prejudicar bastante** (+0,742, significativo *contra*) a ser
**neutra**. Nenhuma configuração conseguiu fazê-la contribuir.

O `gate` — um escalar aprendível que multiplica a contribuição da cena — **fecha
sozinho** em ~0,003 nos cinco folds, partindo de 0,05. O modelo desliga a cena por
conta própria.

---

## 2. Os três elos, e como ficaram

A cena percorre três elos até chegar à predição. Dois estão fechados.

| elo | situação |
|---|---|
| **encoder** | **descartado como culpado.** Reconstrói bem (exp. 21); a qualidade da reconstrução **não** prediz o ADE (r = +0,34, exp. 27); e um encoder **2 a 5× melhor** produz exatamente a mesma predição (+0,006, p=0,87, exp. 30) |
| **representação** | **descartada por duas vias independentes.** Densidade contínua em vez de ocupação binária — quatro ordens de grandeza a mais de informação: nada (exp. 29). Range-view em resolução nativa, 2.650 colunas azimutais contra 300 voxels de 2 m: nada (exp. 31) |
| **consumo** | **intocado em 32 experimentos.** A cena entra no decoder por **uma única query** de cross-attention, comprimida a 64 dimensões |

O consumo é o único suspeito que resta.

---

## 3. A retratação: o melhor resultado era um artefato

Durante semanas, o melhor resultado do projeto foi `gate0` — a arquitetura completa
com a **cena desligada** — vencendo a baseline cinemática por **−0,217 em 5/5
folds**. Lia-se como *"o decoder com cross-attention acrescenta capacidade sobre o
MLP"*.

Ao investigar **por quê**, descobriu-se que os dois modelos têm o **mesmo** MLP
(512-512-512), o **mesmo** dataset e os **mesmos** hiperparâmetros. Restava uma
única diferença: `gate0` concatena `scene_dim=64` à entrada e, com o gate congelado
em zero, essas 64 colunas valem **exatamente zero**.

Mas `nn.Linear` inicializa com limite `1/√in_features`:

```
baseline:  Linear(15, 512)   desvio dos pesos sobre a história = 0,14851
gate0:     Linear(79, 512)   desvio dos pesos sobre a história = 0,06454
```

As **mesmas** 15 entradas, com pesos iniciais **2,3× menores**.

**O controle (exp. 32).** Colando na baseline 64 colunas de zeros — sem informação
alguma; verificado que perturbá-las em +100 altera a saída em `0,00000000`:

| fold | zeros − baseline | `gate0` − baseline |
|---|---|---|
| 0 | −0,636 | −0,553 |
| 1 | −0,085 | −0,080 |
| 2 | −0,154 | −0,073 |
| 3 | −0,354 | −0,320 |
| 4 | −0,069 | −0,060 |

**−0,260 em 5/5 folds (p=0,072), com correlação r = +0,991** entre as duas colunas.
O resíduo — o que sobra de `gate0` depois de descontar a escala — é **+0,042**: a
arquitetura com atenção é, se tanto, ligeiramente **pior**.

O controle de sanidade passou **bit a bit** (`max|dif| = 0,0000` em 40 sementes),
então não há dúvida sobre o pipeline.

---

## 4. O ranking completo

Contra a baseline cinemática pura, k=1, 5 folds × 8 sementes:

| configuração | ADE | vs baseline | folds |
|---|---|---|---|
| **baseline + 64 zeros** | **2,776** | **−0,260** | 5/5 |
| `gate0` (arquitetura, sem cena) | 2,818 | −0,217 | 5/5 |
| `gated_dens` (densidade) | 2,849 | −0,187 | 5/5 |
| `gated_maedens` (encoder de densidade) | 2,856 | −0,179 | 5/5 |
| `gated_obj` (caixa no objeto) | 2,865 | −0,171 | 5/5 |
| baseline cinemática pura | 3,036 | — | — |
| `gated005` (cena, caixa no ego) | 3,103 | +0,067 | 2/5 |
| `gate0_rv` (range-view, sem cena) | 3,261 | +0,226 | 2/5 |
| `gated_rv` (range-view) | 3,612 | +0,576 | 1/5 |

As cinco configurações que vencem a baseline **compartilham `input_dim = 79`**. E a
baseline com essa mesma largura vence as quatro que carregam o ViT.

**Não existe no projeto nenhuma configuração que vença a baseline cinemática por
uma razão diferente da escala de inicialização.**

---

## 5. O que de fato temos: os achados metodológicos

São mais sólidos do que qualquer resultado de predição, e é aqui que está a
contribuição defensável.

**O `minADE_k` premia o que piora a predição.** Passando de k=1 para k=6 modos com
*winner-takes-all*, o `minADE_6` melhora **29 % (p=0,005, 5/5 folds)** enquanto o
ADE real **piora em 0/5 folds** (+0,303, p=0,036). O `minADE_k` escolhe a melhor
das k hipóteses *depois* de ver a resposta — é um oráculo. Um survey recente de
~350 métodos de predição não faz essa ressalva.

**Acrescentar sementes não dá poder estatístico; acrescentar folds dá.** No exp. 31,
o desvio entre sementes é 0,450 e o desvio real entre folds é 0,587 — o ruído de
semente explica apenas **8 %** da dispersão. Dobrar de 8 para 16 sementes custa 39 h
de GPU e reduz o erro padrão em 2 %. Seriam necessários **10 folds**; temos 5,
porque temos 10 cenas. (No exp. 28 a proporção era inversa, 85 % — **não é uma
constante, tem de ser medida a cada vez**.)

**Escolher hiperparâmetro em folds de validação, sempre.** Selecionar o melhor de
três pesos observando 2 folds fabricou um efeito de −8 % que parecia sólido e se
inverteu nos folds retidos. O desenho o pegou porque a partição foi fixada **antes**
de olhar os resultados.

**O braço de controle pega o que o braço experimental esconde.** Um `gate0_rv` com
ADE 11,277 contra 2,781 do controle equivalente denunciou um bug de normalização
(29,5 % do alvo sendo recortado) quatro horas antes de ele contaminar o resultado.
Um número *impossível* é mais informativo do que um número ruim.

**A assinatura do ruído.** No exp. 31, a correlação entre a dificuldade do fold e o
efeito da cena é **r = −0,72**: a cena "ajuda" onde o modelo prediz mal e
**prejudica** onde prediz bem. Não é o que faz a informação útil — é o que faz o
ruído.

**Dados parciais convencem e mentem.** No exp. 31, o fold 0 sozinho dava −1,152 com
8/8 sementes, o resultado mais convincente do projeto. Com 2 folds: −0,64. Com 3:
−0,47. Com 4: −0,04. Com 5: **+0,35**.

---

## 6. Onde estamos

O protocolo de escalonamento é uma **porta, não um itinerário**: 10 → 100 → 1000
*sweeps*, e não se sobe de degrau sem um bom resultado no anterior.

Degrau vigente: **10-100 sweeps**, 275 em disco, 10 cenas, 236 janelas de
treinamento. **O resultado deste degrau ainda não é bom**, portanto não se sobe — e
a base CARMEN_LCAD não é tocada até se estar trabalhando com 1000.

O trabalho é melhorar aqui, não escalar.

---

## Estrutura do repositório

O repositório é um fork completo do Sapiens (Meta). O código próprio são 167
arquivos:

| caminho | conteúdo |
|---|---|
| raiz (C++/OpenGL) | visor próprio de nuvens de pontos |
| `utilities/` | pipeline de extração e preparação de dados |
| `sapiens/pretrain/*.{py,sh}` | scripts de experimento |
| `sapiens/pretrain/mmpretrain/{datasets,models}/` | camadas MOTF (datasets, MAE 4D, decoder) |

Todo o restante é mmpretrain sem modificação.

**Stack:** PyTorch + mmengine/mmpretrain, CUDA numa RTX 4060 Laptop de 8 GB.
Ambientes conda: `sapiens_gpu` (treinamento), `waymo_env` (extração).

**Documentação:**
[`docs/EXPERIMENTOS_DECODER.md`](docs/EXPERIMENTOS_DECODER.md) — os 32 experimentos ·
[`docs/CODEBASE_MAP.md`](docs/CODEBASE_MAP.md) — arquitetura, fluxo de dados e as 36 armadilhas ·
[`CLAUDE.md`](CLAUDE.md) — estado atual e regras do projeto
