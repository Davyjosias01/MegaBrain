# Capítulo 6 — Fine-tuning for Classification

Glossário dos termos introduzidos na seção 5.5 e no Capítulo 6 de RASCHKA, S. *Build a Large Language
Model (From Scratch)*: carregamento de pesos pré-treinados e adaptação do GPT para classificação de
textos. Entradas em ordem alfabética.

Até o Capítulo 5 o modelo foi construído e treinado do zero. Aqui o ponto de partida muda: os pesos
já vêm treinados, e o trabalho passa a ser **adaptar** o modelo a uma tarefa com poucos dados
rotulados. O caminho é:

```
pesos publicados do GPT-2          arquivo com 160 tensores
  → GPTModel da Sprint 4           mesma arquitetura, agora com conhecimento da língua
  → troca da cabeça de saída       Linear(768, 50257)  →  Linear(768, 2)
  → congelamento                   só o fim do modelo continua treinável
  → lotes (mensagem, rótulo)       (b, L) e (b,)
  → logits do último token         (b, 2)
  → entropia cruzada + AdamW       ajuste supervisionado
  → acurácia em dados não vistos   avaliação
```

Cada entrada diz a **função no fine-tuning** e a **relação com os demais conceitos**. Os exemplos
numéricos vêm das execuções do projeto (`results-by-sprints/sprint06/`): GPT-2 124M ajustado em CPU
sobre a *SMS Spam Collection*. Os termos do fine-tuning de instruções estão no
[glossário do Capítulo 7](capitulo-07.md); *fine-tuning* e *pré-treinamento* foram apresentados, em
nível conceitual, no [glossário do Capítulo 1](capitulo-01.md).

Notação usada nas fórmulas:

| Símbolo | Significado |
|---|---|
| `b` | tamanho do lote (`batch_size`) |
| `L` | número de tokens por sequência (`max_length`) |
| `d` | dimensão dos embeddings (`emb_dim`) — 768 no GPT-2 124M |
| `V` | tamanho do vocabulário — 50.257 no GPT-2 |
| `C` | número de classes — 2 (não spam, spam) |
| `VP`, `FP`, `VN`, `FN` | verdadeiros positivos, falsos positivos, verdadeiros negativos, falsos negativos |

---

## Class balancing (balanceamento de classes)

**Definição.** Ajuste da base para que todas as classes tenham o mesmo número de exemplos. A forma
usada no capítulo é a subamostragem (*undersampling*): sortear, da classe majoritária, tantos exemplos
quantos tem a minoritária, e descartar o resto.

**Função no fine-tuning.** Impede que o modelo aprenda a responder sempre a classe mais frequente e
dá à [acurácia](#classification-accuracy-acurácia-de-classificação) um ponto de referência claro: com
duas classes iguais, um palpite constante acerta 50%, e tudo acima disso é aprendizado.

**Relação com outros conceitos.** Vem antes da divisão em treino, validação e
[teste](#evaluation-dataset-conjunto-de-avaliação). É uma alternativa a manter a base desbalanceada e
avaliar com [precisão, revocação e F1](#precision-recall-e-f1-score-precisão-revocação-e-f1), que não
se deixam enganar pela classe majoritária. O custo é jogar dados fora.

**Exemplo.** A base original tem 4.827 mensagens legítimas e 747 de spam (87% / 13%): responder
sempre "não spam" daria 87% de acurácia sem aprender nada. `balancear_classes`
(`src/classificacao.py`) reduz as duas classes a 653 mensagens cada — o número de spams únicos, depois
de `remover_duplicatas` tirar as 403 mensagens repetidas da base.

---

## Classification accuracy (acurácia de classificação)

**Definição.** Fração dos exemplos classificados corretamente: `(VP + VN) / total`. A classe prevista
é a de maior logit (`argmax`).

**Função no fine-tuning.** É a métrica que diz se o modelo executa a tarefa. Não serve de função de
perda porque não é diferenciável: uma mudança pequena nos pesos quase nunca altera o `argmax`, então o
gradiente seria zero em quase todo lugar. Por isso treina-se com a entropia cruzada e **mede-se** com a
acurácia.

**Relação com outros conceitos.** Só é informativa sozinha quando as classes estão
[balanceadas](#class-balancing-balanceamento-de-classes). Deve ser medida no
[conjunto de avaliação](#evaluation-dataset-conjunto-de-avaliação), não no de treino. Resume em um
número a [matriz de confusão](#confusion-matrix-matriz-de-confusão), escondendo que tipo de erro o
modelo comete — o que [precisão e revocação](#precision-recall-e-f1-score-precisão-revocação-e-f1)
mostram.

**Exemplo.** `acuracia_do_loader` (`src/classificacao.py`). Antes do ajuste, o GPT-2 com a cabeça
nova acerta 45,4% do teste (responde "spam" para todas as 262 mensagens, e 119 delas são spam). Depois
de 5 épocas: 96,60% no treino, 96,92% na validação e 98,09% no teste — 257 acertos em 262.

---

## Classification fine-tuning (fine-tuning para classificação)

**Definição.** [Fine-tuning](#fine-tuning-ajuste-fino) em que o modelo é adaptado para atribuir a cada
texto uma classe de um conjunto fixo e pequeno, em vez de gerar texto.

**Função no modelo de linguagem.** Transforma um gerador em um modelo especializado. O corpo do GPT
continua produzindo, para cada token, um vetor de dimensão `d`; a
[cabeça de classificação](#classification-head-cabeça-de-classificação) converte um desses vetores em
`C` pontuações. O modelo resultante só sabe responder com uma das classes que viu no treino.

**Relação com outros conceitos.** É a contraparte do
[fine-tuning de instruções](capitulo-07.md#instruction-fine-tuning-fine-tuning-de-instruções) do
Capítulo 7: a classificação precisa de menos dados e menos computação e é mais fácil de avaliar, mas o
modelo fica restrito a uma tarefa; o de instruções mantém o modelo capaz de várias tarefas, a um custo
maior. É um caso de [fine-tuning supervisionado](#supervised-fine-tuning-fine-tuning-supervisionado) e
de [aprendizado por transferência](#transfer-learning-aprendizado-por-transferência).

**Exemplo.** `experiments/sprint06-fine-tuning/treinar_classificador.py`: GPT-2 124M ajustado para
dizer se um SMS é spam, com 914 mensagens de treino, 5 épocas, AdamW (`lr = 5e-5`,
`weight_decay = 0.1`), lotes de 8 — 40 minutos em CPU.

---

## Classification head (cabeça de classificação)

**Definição.** A camada de saída do classificador: uma `Linear(d, C)` que substitui a cabeça original
`Linear(d, V)` do modelo de linguagem.

**Função no fine-tuning.** É a única parte nova do modelo. A cabeça original devolvia uma pontuação
para cada um dos 50.257 tokens; a nova devolve uma por classe. Ela começa com pesos aleatórios — por
isso o modelo recém-adaptado não sabe classificar, embora todo o resto já esteja treinado — e é o
primeiro alvo do ajuste.

**Relação com outros conceitos.** Recebe o vetor do
[último token](#last-token-representation-representação-do-último-token). Seus parâmetros são sempre
[treináveis](#trainable-parameters-parâmetros-treináveis), mesmo quando todas as
[camadas congeladas](#frozen-layers-camadas-congeladas) ficam de fora. Os logits que produz vão para a
entropia cruzada, a mesma função de perda do [Capítulo 5](capitulo-05.md).

**Exemplo.** `adaptar_para_classificacao` (`src/classificacao.py`) faz
`modelo.out_head = nn.Linear(768, 2)`: 768 × 2 pesos + 2 bias = 1.538 parâmetros, contra os 38,6
milhões da cabeça original. A saída do modelo passa de `(b, L, 50257)` para `(b, L, 2)`.

---

## Confusion matrix (matriz de confusão)

**Definição.** Tabela que cruza a classe verdadeira com a classe prevista. Com duas classes tem
quatro células: `VP` (spam classificado como spam), `FP` (legítima classificada como spam), `VN`
(legítima classificada como legítima) e `FN` (spam que passou como legítimo).

**Função no fine-tuning.** Mostra **que tipo** de erro o classificador comete, o que a
[acurácia](#classification-accuracy-acurácia-de-classificação) esconde. Em um filtro de spam os dois
erros não custam o mesmo: bloquear uma mensagem legítima costuma ser pior do que deixar um spam passar.

**Relação com outros conceitos.** É a origem de todas as métricas de classificação: acurácia,
[precisão, revocação e F1](#precision-recall-e-f1-score-precisão-revocação-e-f1) são razões entre suas
células.

**Exemplo.** `metricas_de_classificacao` (`src/classificacao.py`), no conjunto de teste
(`results-by-sprints/sprint06/exp02-antes-e-depois.csv`):

| | previsto: não spam | previsto: spam |
|---|---|---|
| **verdadeiro: não spam** | VN = 142 | FP = 1 |
| **verdadeiro: spam** | FN = 4 | VP = 115 |

Antes do ajuste a matriz era `VN = 0, FP = 143, FN = 0, VP = 119`: o modelo marcava tudo como spam.

---

## Evaluation dataset (conjunto de avaliação)

**Definição.** Parte dos dados separada do treino e usada apenas para medir o desempenho do modelo.
Divide-se em dois papéis: o conjunto de **validação**, consultado durante o treino para acompanhar a
perda e escolher configurações, e o de **teste**, usado uma única vez, no fim.

**Função no fine-tuning.** Mede generalização: se o modelo acerta em textos que não ajustaram seus
pesos. A acurácia no treino só diz que ele se ajustou àqueles exemplos, o que pode ser memorização.

**Relação com outros conceitos.** A separação entre validação e teste existe porque escolher
configurações olhando a validação já a "contamina": o teste é o único número que não influenciou
nenhuma decisão. Para que a medida valha, nenhum exemplo pode aparecer em dois conjuntos — o que
depende de remover repetições **antes** de dividir. O comprimento das sequências
([padding](#padding-preenchimento)) também deve vir do treino, não dos dados de avaliação.

**Exemplo.** `dividir_treino_validacao_teste` (`src/classificacao.py`): 914 mensagens de treino, 130
de validação e 262 de teste (70% / 10% / 20%). Seguindo o capítulo ao pé da letra, sem remover as 403
mensagens repetidas da base, 26 mensagens do teste também estariam no treino.

---

## Fine-tuning (ajuste fino)

**Definição.** Continuação do treinamento de um modelo já treinado, em um conjunto de dados menor e
específico, para adaptá-lo a uma tarefa.

**Função no modelo de linguagem.** É a segunda etapa do ciclo de vida de um LLM. O pré-treinamento
(Capítulo 5), caro e feito uma vez sobre texto sem rótulo, dá ao modelo conhecimento geral da língua;
o fine-tuning, barato e repetível, direciona esse conhecimento. Mecanicamente é o mesmo laço de
treinamento — perda, retropropagação, otimizador —, com outro ponto de partida, outros dados e, em
geral, taxa de aprendizado menor, para não apagar o que o pré-treino aprendeu.

**Relação com outros conceitos.** Parte de [pesos pré-treinados](#pretrained-weights-pesos-pré-treinados)
e é a forma mais comum de [aprendizado por transferência](#transfer-learning-aprendizado-por-transferência).
Tem duas variantes no livro: [para classificação](#classification-fine-tuning-fine-tuning-para-classificação)
e [para instruções](capitulo-07.md#instruction-fine-tuning-fine-tuning-de-instruções). Ambas são
[supervisionadas](#supervised-fine-tuning-fine-tuning-supervisionado).

**Exemplo.** No projeto, o treinamento do zero (Sprint 5) usou `lr = 4e-4`; o fine-tuning usa
`lr = 5e-5`, oito vezes menor. Em 5 épocas sobre 914 mensagens, a perda de validação caiu de 2,323
para 0,093 e a acurácia de teste subiu de 45,4% para 98,1%.

---

## Frozen layers (camadas congeladas)

**Definição.** Camadas cujos parâmetros não são atualizados durante o treinamento. No PyTorch,
congela-se um parâmetro com `requires_grad = False`: o gradiente dele deixa de ser calculado e o
otimizador não o altera.

**Função no fine-tuning.** Preserva o conhecimento do pré-treino e reduz o custo. As primeiras camadas
de um modelo pré-treinado guardam padrões gerais da língua, úteis em qualquer tarefa; as últimas são
as mais ligadas à tarefa original. Ajustar só o fim costuma bastar, exige menos memória (nada de
gradiente nem estado do otimizador para o que está congelado) e diminui o risco de sobreajuste quando
há poucos exemplos.

**Relação com outros conceitos.** É o complemento dos
[parâmetros treináveis](#trainable-parameters-parâmetros-treináveis): todo parâmetro está em um dos
dois grupos. As camadas congeladas continuam participando do passo direto — a mensagem atravessa os 12
blocos —; o que deixa de acontecer é a atualização. A retropropagação só precisa voltar até a primeira
camada treinável.

**Exemplo.** `adaptar_para_classificacao(modelo, blocos_treinaveis=1)` congela os 11 primeiros blocos
e os embeddings do GPT-2 124M e deixa treináveis o último bloco, a `final_norm` e a cabeça. Nos testes
da task B3, depois de um passo do AdamW nenhum tensor congelado tinha mudado e todos os treináveis
tinham. O experimento `exp03_camadas_treinaveis.py` mede o efeito de congelar mais ou menos, com o
mesmo treino:

| Treinável | Parâmetros treináveis | Acurácia no teste | Tempo (CPU) |
|---|---|---|---|
| só a cabeça | 1.538 | 83,59% | 32,0 min |
| último bloco + norma final + cabeça | 7.090.946 | 98,09% | 40,0 min |
| 6 últimos blocos + norma final + cabeça | 42.530.306 | 99,24% | 53,7 min |

Mesmo com tudo congelado menos a cabeça, o tempo cai pouco: o passo direto pelos 12 blocos continua
sendo feito em todo lote.

---

## Last-token representation (representação do último token)

**Definição.** O vetor de saída do modelo na última posição da sequência, usado como resumo do texto
inteiro para a classificação.

**Função no fine-tuning.** O modelo produz um vetor por token, mas a mensagem tem um único rótulo: é
preciso escolher de qual posição ler a classe. No GPT a escolha é a última, por causa da máscara
causal — cada posição só enxerga a si mesma e as anteriores, de modo que a última é a única cujo vetor
foi calculado a partir da mensagem inteira.

**Relação com outros conceitos.** Depende da *causal attention* do [Capítulo 3](capitulo-03.md). É o
que alimenta a [cabeça de classificação](#classification-head-cabeça-de-classificação). Convive com o
[padding](#padding-preenchimento): nas mensagens curtas a última posição é um token de padding, mas
como o padding fica à direita ela continua enxergando todos os tokens reais.

**Exemplo.** `logits_de_classe` (`src/classificacao.py`) faz `modelo(entrada)[:, -1, :]`: de
`(b, L, 2)` para `(b, 2)`. O parâmetro `posicao` permite ler de outro token; o experimento
`exp04_token_de_classificacao.py` compara o último com o primeiro, que só enxerga a primeira palavra:
com o mesmo treino, a acurácia no teste cai de 98,09% para 64,50%. Uma simples tabela de contagem —
a classe mais frequente, no treino, para cada primeiro token — acerta 74,43%: lendo a primeira
posição, o modelo de 124 milhões de parâmetros não faz melhor do que decorar primeiras palavras.

---

## Padding (preenchimento)

**Definição.** Acréscimo de um token sem conteúdo ao fim das sequências curtas, para que todas as de
um lote tenham o mesmo comprimento. O token usado é `<|endoftext|>` (ID 50256).

**Função no fine-tuning.** Um [lote](#training-batch-lote-de-treinamento) é um tensor retangular, e as
mensagens têm tamanhos diferentes. O padding as iguala; as mensagens maiores que o comprimento
escolhido são truncadas.

**Relação com outros conceitos.** O comprimento é definido pelo conjunto de treino e aplicado aos de
[avaliação](#evaluation-dataset-conjunto-de-avaliação). Não atrapalha a leitura do
[último token](#last-token-representation-representação-do-último-token) porque fica à direita. No
[fine-tuning de instruções](capitulo-07.md#collate-function-função-de-agrupamento) o padding é feito
por lote, e não pela base inteira, e é excluído do cálculo da perda.

**Exemplo.** `DatasetDeClassificacao` (`src/classificacao.py`): a maior mensagem do treino tem 188
tokens, e as três partições usam esse comprimento. A mensagem `"Hi"` vira `[17250, 50256, 50256, …]`.
Os lotes têm forma `(8, 188)`.

---

## Precision, recall e F1-score (precisão, revocação e F1)

**Definição.** Três métricas calculadas a partir da
[matriz de confusão](#confusion-matrix-matriz-de-confusão), tomando uma classe como positiva (aqui,
spam):

- **precisão** `= VP / (VP + FP)` — das mensagens marcadas como spam, quantas eram spam;
- **revocação** `= VP / (VP + FN)` — dos spams existentes, quantos foram pegos;
- **F1** `= 2 · P · R / (P + R)` — média harmônica das duas: só é alta se ambas forem.

**Função no fine-tuning.** Separam os dois tipos de erro que a
[acurácia](#classification-accuracy-acurácia-de-classificação) mistura. A precisão cai quando o filtro
bloqueia mensagens legítimas; a revocação cai quando deixa spam passar.

**Relação com outros conceitos.** São indispensáveis quando as classes **não** estão
[balanceadas](#class-balancing-balanceamento-de-classes), caso em que a acurácia engana. Uma revocação
de 100% sozinha não significa nada: marcar tudo como positivo a atinge.

**Exemplo.** `metricas_da_matriz` (`src/classificacao.py`, implementação própria). No teste, depois do
ajuste: precisão `115 / 116 = 99,1%`, revocação `115 / 119 = 96,6%`, F1 `= 0,979`. Antes do ajuste a
revocação era 100% e a precisão 45,4% — o modelo "pegava" todo spam porque marcava tudo como spam.

---

## Pretrained weights (pesos pré-treinados)

**Definição.** Valores dos parâmetros de um modelo obtidos em um treinamento anterior, em geral longo
e sobre muito texto, e distribuídos em arquivo para serem reutilizados.

**Função no modelo de linguagem.** São o conhecimento do modelo. A arquitetura define **quais**
operações são feitas; os pesos definem **o que** elas calculam. Carregar pesos pré-treinados dá ao
`GPTModel` da Sprint 4, sem nenhum treino nosso, a capacidade de prever texto em inglês.

**Relação com outros conceitos.** Carregá-los exige que a arquitetura seja idêntica à do modelo
original e que cada tensor do arquivo seja associado ao parâmetro certo — o que envolve nomes
diferentes e, no caso do GPT-2, dois ajustes de formato: os pesos das camadas lineares estão
transpostos (`Conv1D` guarda `(entrada, saída)`; `nn.Linear`, `(saída, entrada)`) e as projeções de
*query*, *key* e *value* vêm concatenadas em um único tensor. São o ponto de partida do
[fine-tuning](#fine-tuning-ajuste-fino).

**Exemplo.** `src/pesos_pretreinados.py`: o arquivo do GPT-2 124M tem 160 tensores (548 MB), dos quais
148 são pesos e 12 são máscaras causais. `mapa_de_parametros` os associa aos 197 parâmetros do nosso
modelo. Com a mesma arquitetura, a perplexidade em `the-verdict.txt` é 58.916 com pesos aleatórios e
41,9 com os pré-treinados; o próximo token após "Every effort moves you" é ` forward`, com 17,0% de
probabilidade (`exp01_aleatorio_vs_pretreinado.py`).

---

## Supervised fine-tuning (fine-tuning supervisionado)

**Definição.** [Fine-tuning](#fine-tuning-ajuste-fino) feito com exemplos **rotulados**: cada entrada
vem acompanhada da saída correta, fornecida por uma pessoa.

**Função no modelo de linguagem.** Diz ao modelo explicitamente o que se espera dele. No pré-treino o
"rótulo" é o próprio texto — o alvo de cada posição é o token seguinte, e nenhuma anotação humana é
necessária (aprendizado autossupervisionado). No fine-tuning supervisionado alguém precisou marcar
cada mensagem como spam ou não, ou escrever a resposta de cada instrução.

**Relação com outros conceitos.** Abrange as duas variantes do livro:
[classificação](#classification-fine-tuning-fine-tuning-para-classificação) (o rótulo é uma classe) e
[instruções](capitulo-07.md#instruction-fine-tuning-fine-tuning-de-instruções) (o rótulo é um texto).
Como rotular custa caro, as bases são pequenas — e é o
[aprendizado por transferência](#transfer-learning-aprendizado-por-transferência) que faz poucos
exemplos bastarem.

**Exemplo.** O pré-treino do GPT-2 usou dezenas de gigabytes de texto sem rótulo; o classificador do
projeto foi ajustado com 914 mensagens rotuladas e chegou a 98,1% no teste.

---

## Trainable parameters (parâmetros treináveis)

**Definição.** Os parâmetros que o otimizador atualiza durante o treinamento: aqueles com
`requires_grad = True`.

**Função no fine-tuning.** Determinam a capacidade de adaptação e o custo. Mais parâmetros treináveis
permitem ao modelo se afastar mais do ponto de partida, mas cada um deles exige memória para o
gradiente e para o estado do otimizador (no AdamW, duas médias móveis), além de retropropagação até a
camada onde está.

**Relação com outros conceitos.** São o complemento das
[camadas congeladas](#frozen-layers-camadas-congeladas). A
[cabeça de classificação](#classification-head-cabeça-de-classificação) está sempre entre eles. Só os
treináveis devem ser entregues ao otimizador.

**Exemplo.** Na configuração base, 7.090.946 dos 124.441.346 parâmetros são treináveis (5,7%):
7.087.872 do último bloco Transformer, 1.536 da `final_norm` e 1.538 da cabeça
(`contar_treinaveis`, `src/classificacao.py`). O ajuste do modelo inteiro não foi executado: modelo,
gradientes e estado do AdamW somariam cerca de 2 GB, mais do que a máquina usada tinha livre.

---

## Training batch (lote de treinamento)

**Definição.** Grupo de exemplos processados juntos em um passo de treinamento. Na classificação, um
lote é o par `(entradas, rótulos)`: as entradas com forma `(b, L)` e os rótulos com forma `(b,)`.

**Função no fine-tuning.** É a unidade sobre a qual se calcula a perda e se dá um passo do otimizador.
A média da perda sobre o lote dá um gradiente menos ruidoso do que o de um exemplo isolado, e
processar vários exemplos de uma vez aproveita o paralelismo do hardware.

**Relação com outros conceitos.** Exige sequências do mesmo comprimento, o que na classificação é
resolvido por [padding](#padding-preenchimento) até um comprimento único da base. Difere do lote do
pré-treino ([Capítulo 2](capitulo-02.md)), em que o alvo era outra sequência `(b, L)` — o texto
deslocado em uma posição —, e do lote de instruções
([Capítulo 7](capitulo-07.md#collate-function-função-de-agrupamento)), montado por uma função de
agrupamento. O tamanho do lote e o embaralhamento são hiperparâmetros do treino.

**Exemplo.** `criar_loaders_classificacao` (`src/classificacao.py`): lotes de 8 mensagens, forma
`(8, 188)`, rótulos como `[1, 1, 0, 0, 1, 1, 1, 0]`. As 914 mensagens de treino formam 114 lotes por
época; o loader de treino embaralha e descarta o último lote incompleto.

---

## Transfer learning (aprendizado por transferência)

**Definição.** Aproveitamento, em uma tarefa nova, do conhecimento que um modelo adquiriu em outra
tarefa, em vez de aprender tudo do zero.

**Função no modelo de linguagem.** É o que torna os LLMs práticos. Aprender a língua exige muito texto
e muita computação; aprender a tarefa, depois disso, exige pouco. O modelo transfere para a
classificação o que aprendeu prevendo o próximo token: o que são palavras, como se combinam, que
vocabulário acompanha um pedido de dinheiro ou uma conversa entre amigos.

**Relação com outros conceitos.** O [fine-tuning](#fine-tuning-ajuste-fino) é a forma de fazer a
transferência; os [pesos pré-treinados](#pretrained-weights-pesos-pré-treinados) são o que se
transfere; as [camadas congeladas](#frozen-layers-camadas-congeladas) são o conhecimento mantido
intacto.

**Exemplo.** No teste da task B5, um GPT pequeno treinado **do zero** em 160 mensagens chegou a 100%
de acurácia no treino e só 79,7% na validação — decorou os exemplos. O GPT-2 pré-treinado, ajustando
5,7% dos parâmetros em 914 mensagens, chegou a 96,9% na validação e 98,1% no teste.
