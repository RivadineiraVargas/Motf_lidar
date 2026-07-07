# Plano de Trabalho — Predição de Trajetórias Futuras de Objetos Movíveis a partir de Sweeps de LIDAR

Gabriel,

O objetivo do seu trabalho é desenvolver uma arquitetura neural capaz de, a partir de um conjunto de capturas passadas de um LIDAR de múltiplos raios usado em veículos autônomos, predizer as trajetórias futuras dos objetos movíveis presentes na cena. Por objetos movíveis, entendemos carros, caminhões, motocicletas, bicicletas, pedestres e outros agentes capazes de se mover, ainda que, no instante observado, possam estar parados. Esses objetos devem ser distinguidos de elementos não movíveis do ambiente, como postes, prédios, muros, calçadas e demais estruturas estáticas.

A arquitetura proposta deverá seguir a ideia geral de um Transformer completo, com encoder e decoder. O trabalho será dividido em duas grandes fases. Na primeira, será treinado um encoder auto-supervisionado a partir de sweeps de LIDAR, usando mascaramento da entrada e reconstrução dos dados completos, em uma abordagem inspirada no MAE — Masked Autoencoder — de He et al. Na segunda fase, esse encoder será acoplado a um decoder treinado de forma supervisionada para produzir trajetórias futuras dos objetos movíveis, tomando como inspiração arquiteturas modernas de previsão de movimento, em especial o Wayformer, da Waymo.

## 1. Definição da entrada do sistema

A entrada da rede será composta por múltiplos sweeps passados do LIDAR. Esses sweeps representam a evolução temporal da cena antes do instante atual. Ainda precisamos definir exatamente quantos sweeps serão usados, mas a ideia é trabalhar com janelas temporais que representem alguns segundos no passado.

Por exemplo, se o LIDAR operar a 20 Hz, uma janela de cinco segundos corresponderia a 100 sweeps. Entretanto, não é obrigatório usar todos os sweeps consecutivos. Podemos representar os mesmos cinco segundos usando uma amostragem temporal mais esparsa, como 10 sweeps selecionados ao longo da janela. Assim, a entrada poderá ser uma sequência temporal completa ou uma sequência subamostrada dos sweeps passados.

O primeiro passo será implementar e testar diferentes configurações de entrada:

* número pequeno de sweeps;
* número intermediário de sweeps;
* sequência temporal completa;
* sequência temporal subamostrada;
* diferentes tamanhos de janela temporal no passado.

O objetivo inicial não é ainda obter o melhor desempenho, mas garantir que a entrada esteja sendo corretamente representada, carregada, mascarada e visualizada.

## 2. Treinamento auto-supervisionado do encoder

A primeira fase do trabalho será o treinamento do encoder. Esse treinamento será auto-supervisionado, isto é, não dependerá de anotações de trajetórias. A ideia é usar grandes quantidades de sweeps de LIDAR sem rótulos, mascarar partes da entrada e treinar a rede para reconstruir os dados completos.

Essa estratégia é inspirada no trabalho de He et al. sobre Masked Autoencoders, no qual partes da entrada são ocultadas e a rede aprende a reconstruí-las. No nosso caso, em vez de imagens RGB, a entrada será composta por representações derivadas dos sweeps de LIDAR.

O encoder deve aprender representações úteis da cena. Espera-se que, ao reconstruir sweeps parcialmente mascarados, ele aprenda regularidades do mundo físico observado pelo LIDAR, incluindo:

* formas típicas de veículos;
* formas típicas de pedestres;
* distinção entre objetos movíveis e estruturas estáticas;
* continuidade espacial dos objetos;
* coerência temporal entre sweeps consecutivos;
* padrões de ocupação do ambiente;
* regiões livres, ocupadas e parcialmente observadas.

Durante esse treinamento, haverá uma cabeça de reconstrução responsável por produzir como saída uma reconstrução da entrada completa. Essa cabeça será necessária apenas na fase auto-supervisionada. Depois que o encoder estiver treinado, ela poderá ser descartada.

## 3. Validação inicial em pequena escala

Antes de treinar com grandes volumes de dados, precisamos demonstrar que o pipeline funciona em pequena escala. Esta é uma etapa crítica do trabalho.

Você deve preparar experimentos com conjuntos muito pequenos, por exemplo:

* 10 sweeps;
* 100 sweeps;
* 1.000 sweeps.

O objetivo desses experimentos é verificar se a rede consegue aprender nesses pequenos conjuntos e, idealmente, se consegue fazer overfit. Conseguir overfit em um conjunto pequeno é uma evidência importante de que:

* os dados estão sendo carregados corretamente;
* o mascaramento está funcionando;
* a arquitetura está conectada corretamente;
* a função de perda está adequada;
* o treinamento está atualizando os pesos;
* a saída reconstruída corresponde ao que se espera;
* a visualização permite interpretar os resultados.

Se a rede não conseguir aprender nem nesses conjuntos pequenos, não faz sentido avançar para treinamentos maiores. Portanto, esta etapa deve ser tratada como prioridade imediata.

## 4. Redução da arquitetura para testes rápidos

A arquitetura atual pode estar grande demais para os testes iniciais. Como o encoder é baseado em Transformer, ele deve ter vários blocos repetidos. Para a fase inicial, você deve reduzir o tamanho da rede no arquivo de configuração.

Por exemplo, se o encoder tiver 50 blocos Transformer, teste versões menores com:

* 25 blocos;
* 10 blocos;
* 5 blocos;
* eventualmente ainda menos, se necessário.

Também podem ser reduzidos outros hiperparâmetros, como dimensão dos embeddings, número de cabeças de atenção, tamanho do MLP interno e batch size. O objetivo não é obter a melhor arquitetura final agora, mas ter uma rede pequena o suficiente para treinar rapidamente em uma máquina menor.

Essa rede reduzida deve ser usada para depuração, entendimento do comportamento do treinamento e validação do pipeline.

## 5. Visualização da reconstrução do encoder

A avaliação visual será essencial. Embora seja importante calcular métricas de erro, a melhor forma inicial de saber se o encoder está aprendendo algo útil é olhar a reconstrução.

Você deve preparar uma visualização clara mostrando:

* sweep original;
* regiões mascaradas;
* reconstrução produzida pela rede;
* diferença entre original e reconstruído, se possível;
* exemplos do conjunto de treinamento;
* exemplos do conjunto de validação;
* exemplos nunca vistos pela rede.

A visualização deve permitir responder rapidamente perguntas como:

* a rede reconstrói formas plausíveis?
* objetos aparecem em posições coerentes?
* a estrutura geral da cena é preservada?
* há diferença clara entre uma rede treinada e uma rede não treinada?
* a rede generaliza para sweeps que não viu durante o treinamento?

Essa visualização, que você já começou a desenvolver, deve ser consolidada antes de avançarmos para treinamentos maiores.

## 6. Treinamento do encoder em escala maior

Depois que o pipeline estiver validado com 10, 100 e 1.000 sweeps, devemos avançar para um treinamento maior do encoder.

Uma primeira meta razoável é treinar com pelo menos 50 mil sweeps. Depois, se o treinamento estiver funcionando bem, podemos aumentar para centenas de milhares de sweeps ou mais.

Nesta fase, os objetivos serão:

* treinar o encoder com volume maior de dados;
* avaliar a reconstrução em dados nunca vistos;
* medir erro de reconstrução;
* analisar qualitativamente as reconstruções;
* verificar se a rede aprende representações úteis da cena;
* comparar diferentes taxas de mascaramento;
* comparar diferentes tamanhos de encoder;
* comparar diferentes formas de representar os sweeps de LIDAR.

O critério de sucesso desta fase será ter um encoder que, ao receber uma entrada parcialmente mascarada, consiga reconstruir de forma plausível e coerente os sweeps completos, inclusive em casos não vistos durante o treinamento.

## 7. Transição para o problema supervisionado de trajetórias

Quando estivermos satisfeitos com o encoder, passaremos para a segunda fase: a predição supervisionada de trajetórias futuras.

Nesta etapa, a cabeça de reconstrução será removida. O encoder treinado será conectado a um decoder. O decoder receberá a representação produzida pelo encoder e deverá gerar trajetórias futuras para os objetos movíveis presentes na cena.

Esse treinamento exigirá um conjunto de dados anotado, contendo, para cada cena:

* sweeps passados de LIDAR;
* objetos movíveis presentes;
* trajetórias futuras desses objetos;
* indicação de quais trajetórias são válidas;
* possivelmente classes dos objetos, como carro, pedestre, ciclista etc.

Nesta fase, o aprendizado deixa de ser auto-supervisionado e passa a ser supervisionado.

## 8. Representação das trajetórias de saída

Um dos problemas centrais do decoder será definir como representar as trajetórias na saída da rede.

Uma possibilidade prática é definir um número máximo fixo de trajetórias. Por exemplo, a rede poderia produzir sempre até 100 trajetórias. Cada trajetória teria uma marca de validade. Assim:

* se houver 20 objetos movíveis na cena, 20 trajetórias seriam válidas e 80 inválidas;
* se não houver objetos movíveis, todas as trajetórias seriam inválidas;
* se houver mais de 100 objetos movíveis, alguns não poderiam ser representados.

A expectativa é que casos com mais objetos do que o limite máximo sejam raros ou inexistentes no conjunto de dados. Ainda assim, esse limite precisa ser escolhido com cuidado.

Também será necessário definir a forma interna de cada trajetória. Algumas possibilidades são:

* sequência de posições futuras no plano;
* sequência de deslocamentos relativos;
* sequência de velocidades;
* sequência de waypoints;
* representação multimodal, com várias trajetórias candidatas por objeto;
* representação tokenizada, adequada para uso em Transformer.

Esse ponto exige estudo cuidadoso dos trabalhos existentes.

## 9. Estudo do Wayformer e de trabalhos relacionados

Você deve estudar o artigo Wayformer: Motion Forecasting via Simple & Efficient Attention Networks, da Waymo. Esse trabalho é particularmente relevante porque propõe uma arquitetura baseada em atenção para previsão de movimento em direção autônoma, usando um scene encoder e um decoder para produzir trajetórias futuras.

O Wayformer é importante para o nosso trabalho por vários motivos:

* usa uma arquitetura baseada em Transformer;
* trata o problema de previsão de movimento;
* considera entradas heterogêneas de cenas de direção autônoma;
* discute formas de fusão das modalidades de entrada;
* usa atenção para representar interações entre agentes e contexto;
* gera trajetórias futuras como saída.

Além do Wayformer, você deve estudar o artigo do Waymo Open Motion Dataset, que define um grande benchmark para previsão interativa de movimento em direção autônoma. Esse trabalho ajuda a entender o formato do problema, as métricas utilizadas e a importância de prever trajetórias em cenários com múltiplos agentes interagindo.

Também vale estudar trabalhos complementares de previsão de movimento com Transformers, como Motion Transformer, DenseTNT e MotionLM, para comparar formas alternativas de representar intenções, metas e trajetórias futuras.

## 10. Implementação do decoder

Depois de definida a representação das trajetórias, será implementado o decoder.

A arquitetura base deve seguir a ideia de um Transformer decoder. Ele deverá receber como entrada a representação latente produzida pelo encoder e gerar como saída uma sequência de tokens ou vetores representando trajetórias futuras.

Questões que precisarão ser definidas:

* quais tokens representarão objetos;
* quais tokens representarão pontos futuros da trajetória;
* como indicar trajetórias inválidas;
* como lidar com múltiplas hipóteses de trajetória;
* como representar incerteza;
* qual função de perda usar;
* como associar trajetórias preditas a trajetórias reais durante o treinamento;
* como avaliar a qualidade da predição.

Uma implementação inicial pode ser simplificada. Por exemplo, podemos começar com um número fixo de trajetórias e uma única hipótese por objeto. Depois, podemos avançar para múltiplas hipóteses por objeto, como é comum em previsão de movimento.

## 11. Visualização das trajetórias previstas

Assim como no treinamento do encoder, a visualização será essencial na fase do decoder.

Você deve desenvolver ou adaptar um visualizador capaz de mostrar:

* sweeps de LIDAR de entrada;
* objetos movíveis presentes na cena;
* trajetórias reais anotadas;
* trajetórias previstas pela rede;
* trajetórias válidas e inválidas;
* erros de predição;
* comparação entre diferentes modelos.

Esse visualizador deve permitir avaliar rapidamente se a rede está aprendendo comportamentos plausíveis. Por exemplo:

* carros seguem a direção da via?
* pedestres têm trajetórias compatíveis com sua posição?
* objetos parados permanecem parados quando apropriado?
* a rede evita prever trajetórias atravessando obstáculos?
* as previsões são coerentes com o histórico observado?

A inspeção visual será uma ferramenta fundamental de depuração e avaliação.

## 12. Métricas de avaliação

Além da avaliação visual, devemos usar métricas quantitativas para avaliar a predição de trajetórias.

Algumas métricas importantes são:

* erro médio de deslocamento ao longo da trajetória;
* erro no ponto final da trajetória;
* acurácia da indicação de validade da trajetória;
* erro por classe de objeto;
* comparação entre trajetórias previstas e trajetórias reais;
* métricas específicas de benchmarks de previsão de movimento, como as usadas no Waymo Open Motion Dataset.

Inicialmente, podemos usar métricas simples. Depois, à medida que o sistema amadurecer, podemos aproximar a avaliação das métricas usadas em benchmarks internacionais.

## 13. Sequência prática de execução

A sequência imediata de trabalho deve ser a seguinte:

1. Revisar o código atual do encoder.
2. Reduzir a arquitetura para uma versão pequena.
3. Confirmar o carregamento correto dos sweeps de LIDAR.
4. Confirmar o funcionamento do mascaramento.
5. Treinar com 10 sweeps e tentar obter overfit.
6. Treinar com 100 sweeps e tentar obter overfit.
7. Treinar com 1.000 sweeps e avaliar a reconstrução.
8. Consolidar a visualização da entrada, máscara e reconstrução.
9. Aumentar gradualmente o tamanho do conjunto de treinamento.
10. Treinar com pelo menos 50 mil sweeps.
11. Avaliar o encoder em dados nunca vistos.
12. Estudar Wayformer, Waymo Open Motion Dataset e trabalhos relacionados.
13. Definir a representação tokenizada das trajetórias.
14. Projetar o decoder.
15. Treinar o decoder com dados anotados de trajetórias.
16. Criar o visualizador de trajetórias previstas versus trajetórias reais.
17. Avaliar quantitativa e qualitativamente o sistema completo.

## 14. Critério de avanço

Só devemos avançar para o treinamento grande depois que os experimentos pequenos mostrarem que o encoder aprende corretamente.

Só devemos avançar para o decoder depois que o encoder estiver suficientemente validado, tanto por métricas quanto por inspeção visual.

O ponto mais urgente agora é sair da fase indefinida do encoder e demonstrar, de forma objetiva, que o treinamento funciona em pequena escala. Precisamos ver a rede aprendendo com 10, 100 e 1.000 sweeps, entender seu comportamento, ajustar a arquitetura e consolidar a visualização. Depois disso, avançaremos para o treinamento em larga escala e, finalmente, para a predição supervisionada de trajetórias com decoder.

