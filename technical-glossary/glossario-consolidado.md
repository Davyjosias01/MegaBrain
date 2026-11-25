# Glossário consolidado — Capítulos 1 a 7

Índice de todos os termos técnicos do projeto, em ordem alfabética. Cada linha traz o termo, o
capítulo em que ele foi definido, a primeira frase da definição e o link para a entrada completa — que
tem também a **função no modelo**, a **relação com outros conceitos** e um **exemplo**.

Arquivo gerado por [`consolidar.py`](consolidar.py) a partir dos glossários de cada capítulo; não edite
à mão.

| Capítulo | Entradas | Arquivo |
|---|---|---|
| 1 | 30 | [`capitulo-01.md`](capitulo-01.md) |
| 2 | 28 | [`capitulo-02.md`](capitulo-02.md) |
| 3 | 25 | [`capitulo-03.md`](capitulo-03.md) |
| 4 | 30 | [`capitulo-04.md`](capitulo-04.md) |
| 5 | 23 | [`capitulo-05.md`](capitulo-05.md) |
| 6 | 16 | [`capitulo-06.md`](capitulo-06.md) |
| 7 | 8 | [`capitulo-07.md`](capitulo-07.md) |
| **Total** | **160** | |

Um termo listado em dois capítulos foi apresentado em nível conceitual no primeiro e retomado, com a
implementação do projeto, no segundo.


## A

| Termo | Cap. | Definição |
|---|---|---|
| [AdamW (otimizador Adam com weight decay desacoplado)](capitulo-05.md#adamw-otimizador-adam-com-weight-decay-desacoplado) | 5 | Otimizador que, para cada peso, mantém duas médias móveis: a do gradiente (`m`, "momento") e a do gradiente ao quadrado (`v`). |
| [Artificial intelligence (inteligência artificial)](capitulo-01.md#artificial-intelligence-inteligência-artificial) | 1 | Campo amplo da computação que busca construir sistemas capazes de executar tarefas que normalmente exigiriam inteligência humana — entender linguagem, reconhecer padrões, tomar decisões. |
| [Attention score (score de atenção)](capitulo-03.md#attention-score-score-de-atenção) | 3 | Valor não normalizado que mede o quanto um token deve prestar atenção em outro. |
| [Attention weight (peso de atenção)](capitulo-03.md#attention-weight-peso-de-atenção) | 3 | Score de atenção normalizado por softmax, de modo que os pesos de uma mesma linha somem 1 e sejam todos positivos: `αᵢⱼ = softmax(ωᵢ)ⱼ`. |
| [Autoregressive generation (geração autoregressiva)](capitulo-04.md#autoregressive-generation-geração-autoregressiva) | 4 | Procedimento de geração em que o modelo produz um token por vez e cada token produzido é anexado à entrada antes da previsão seguinte. |
| [Autoregressive model (modelo autorregressivo)](capitulo-01.md#autoregressive-model-modelo-autorregressivo) | 1 | Modelo que gera uma sequência um elemento por vez, usando as próprias saídas anteriores como entrada para prever o próximo elemento. |

## B

| Termo | Cap. | Definição |
|---|---|---|
| [Backpropagation (retropropagação)](capitulo-05.md#backpropagation-retropropagação) | 5 | Algoritmo que calcula, pela regra da cadeia, a derivada da perda em relação a cada peso do modelo, percorrendo o grafo de computação de trás para frente — da perda até os embeddings. |
| [Bahdanau attention](capitulo-03.md#bahdanau-attention) | 3 | Mecanismo de atenção proposto em 2014 para redes recorrentes, que permite ao decoder acessar seletivamente todos os estados ocultos do encoder, em vez de depender apenas do último. |
| [Batch (lote)](capitulo-02.md#batch-lote) | 2 | Conjunto de `b` sequências processadas juntas em uma única passagem pelo modelo. |
| [Batch normalization (normalização em lote)](capitulo-04.md#batch-normalization-normalização-em-lote) | 4 | Técnica de normalização que calcula média e variância de cada característica *ao longo do lote*, isto é, comparando os exemplos do lote entre si, e usa essas estatísticas para normalizar a ativação. |
| [BERT (Bidirectional Encoder Representations from Transformers)](capitulo-01.md#bert-bidirectional-encoder-representations-from-transformers) | 1 | Família de modelos construída sobre o [encoder](capitulo-01.md#encoder-codificador) do Transformer original e pré-treinada com **previsão de palavras mascaradas**: palavras escondidas no meio da frase devem ser recuperadas a partir do contexto dos dois lados. |
| [Byte Pair Encoding (BPE, codificação por pares de bytes)](capitulo-02.md#byte-pair-encoding-bpe-codificação-por-pares-de-bytes) | 2 | Algoritmo de tokenização por subpalavras. |

## C

| Termo | Cap. | Definição |
|---|---|---|
| [Causal attention (atenção causal) / Masked attention](capitulo-03.md#causal-attention-atenção-causal--masked-attention) | 3 | Variante da self-attention em que cada token só pode atender a si mesmo e às posições anteriores. |
| [Causal mask (máscara causal)](capitulo-03.md#causal-mask-máscara-causal) | 3 | Matriz triangular que marca as posições a esconder — o triângulo estritamente acima da diagonal. |
| [Checkpoint (ponto de salvamento)](capitulo-05.md#checkpoint-ponto-de-salvamento) | 5 | Arquivo que guarda o estado de um treinamento num dado momento: os pesos do modelo (`state_dict`), o estado do otimizador, a configuração da arquitetura e do treino e, opcionalmente, o histórico das perdas. |
| [Class balancing (balanceamento de classes)](capitulo-06.md#class-balancing-balanceamento-de-classes) | 6 | Ajuste da base para que todas as classes tenham o mesmo número de exemplos. |
| [Classification accuracy (acurácia de classificação)](capitulo-06.md#classification-accuracy-acurácia-de-classificação) | 6 | Fração dos exemplos classificados corretamente: `(VP + VN) / total`. |
| [Classification fine-tuning (fine-tuning para classificação)](capitulo-01.md#classification-fine-tuning-fine-tuning-para-classificação) | 1 | Modalidade de [fine-tuning](capitulo-01.md#fine-tuning-ajuste-fino) em que o conjunto rotulado é formado por textos e seus rótulos de classe. |
| [Classification fine-tuning (fine-tuning para classificação)](capitulo-06.md#classification-fine-tuning-fine-tuning-para-classificação) | 6 | [Fine-tuning](capitulo-06.md#fine-tuning-ajuste-fino) em que o modelo é adaptado para atribuir a cada texto uma classe de um conjunto fixo e pequeno, em vez de gerar texto. |
| [Classification head (cabeça de classificação)](capitulo-06.md#classification-head-cabeça-de-classificação) | 6 | A camada de saída do classificador: uma `Linear(d, C)` que substitui a cabeça original `Linear(d, V)` do modelo de linguagem. |
| [Collate function (função de agrupamento)](capitulo-07.md#collate-function-função-de-agrupamento) | 7 | Função que recebe uma lista de exemplos do dataset e devolve os tensores de um lote. |
| [Confusion matrix (matriz de confusão)](capitulo-06.md#confusion-matrix-matriz-de-confusão) | 6 | Tabela que cruza a classe verdadeira com a classe prevista. |
| [Context cropping (corte do contexto na geração)](capitulo-04.md#context-cropping-corte-do-contexto-na-geração) | 4 | Passo da geração que mantém apenas os últimos `C` tokens da sequência antes de cada chamada ao modelo — no código do livro, `idx[:, -context_size:]`. |
| [Context length (comprimento do contexto, `max_length`)](capitulo-02.md#context-length-comprimento-do-contexto-max_length) | 2 | Número máximo de tokens que o modelo recebe de uma vez — o comprimento `L` de cada sequência de entrada. |
| [Context vector (vetor de contexto)](capitulo-03.md#context-vector-vetor-de-contexto) | 3 | Saída da atenção para um token: a soma dos vetores de value de toda a sequência, ponderada pelos pesos de atenção — `z⁽ⁱ⁾ = Σⱼ αᵢⱼ · vⱼ`. |
| [Cross-entropy loss (perda de entropia cruzada)](capitulo-05.md#cross-entropy-loss-perda-de-entropia-cruzada) | 5 | Para cada posição, transforma os [logits](capitulo-04.md#logits) em probabilidades com softmax e toma o negativo do logaritmo da probabilidade atribuída ao token correto. |

## D

| Termo | Cap. | Definição |
|---|---|---|
| [DataLoader (carregador de dados)](capitulo-02.md#dataloader-carregador-de-dados) | 2 | Classe `torch.utils.data.DataLoader` do PyTorch, que percorre um [Dataset](capitulo-02.md#dataset-conjunto-de-dados) e agrupa suas amostras em lotes, podendo embaralhar a ordem (`shuffle`) e usar processos paralelos (`num_workers`). |
| [Dataset (conjunto de dados)](capitulo-02.md#dataset-conjunto-de-dados) | 2 | Subclasse de `torch.utils.data.Dataset` que define como obter uma amostra a partir de um índice. |
| [Decoder (decodificador)](capitulo-01.md#decoder-decodificador) | 1 | Submódulo do Transformer responsável por **gerar** a sequência de saída, um token por vez. |
| [Decoder-only (arquitetura somente decodificador)](capitulo-04.md#decoder-only-arquitetura-somente-decodificador) | 4 | Arquitetura Transformer formada apenas por blocos do tipo decodificador — sem o codificador e sem a atenção cruzada entre os dois — e com atenção causal, que faz cada posição enxergar somente a si mesma e às anteriores. |
| [Decoding strategy (estratégia de decodificação)](capitulo-05.md#decoding-strategy-estratégia-de-decodificação) | 5 | A regra que escolhe o próximo token a partir das probabilidades produzidas pelo modelo: [greedy](capitulo-04.md#greedy-decoding-decodificação-gulosa) (o mais provável), [amostragem](capitulo-04.md#multinomial-sampling-amostragem-multinomial), [temperatura](capitulo-04.md#temperature-temperatura), [top-k](capitulo-04.md#top-k-sampling-amostragem-top-k) ou combinações. |
| [Deep learning (aprendizado profundo)](capitulo-01.md#deep-learning-aprendizado-profundo) | 1 | Ramo do [machine learning](capitulo-01.md#machine-learning-aprendizado-de-máquina) que usa [redes neurais](capitulo-01.md#neural-network-rede-neural) com três ou mais camadas (redes profundas) para modelar padrões e abstrações complexas nos dados. |
| [Dot product (produto escalar)](capitulo-03.md#dot-product-produto-escalar) | 3 | Soma dos produtos elemento a elemento de dois vetores: `a · b = Σᵢ aᵢbᵢ`. |
| [`drop_last` (descarte do lote incompleto)](capitulo-02.md#drop_last-descarte-do-lote-incompleto) | 2 | Parâmetro do `DataLoader` que, quando `True`, descarta o último lote caso ele tenha menos de `batch_size` amostras. |
| [Dropout](capitulo-03.md#dropout) | 3 | Técnica de regularização que zera aleatoriamente uma fração `p` dos elementos de um tensor durante o treinamento e multiplica os sobreviventes por `1/(1-p)`. |
| [Dropout (no modelo GPT)](capitulo-04.md#dropout-no-modelo-gpt) | 4 | Regularização que, durante o treinamento, zera aleatoriamente uma fração `p` dos elementos de um tensor e multiplica os restantes por `1/(1 − p)`, preservando a escala esperada da saída. |

## E

| Termo | Cap. | Definição |
|---|---|---|
| [Embedding dimension (dimensão do embedding)](capitulo-02.md#embedding-dimension-dimensão-do-embedding) | 2 | Número de componentes `d` do vetor que representa cada token — o `output_dim` (ou `embedding_dim`) da camada de embedding. |
| [Embedding layer (camada de embedding)](capitulo-02.md#embedding-layer-camada-de-embedding) | 2 | A camada `torch.nn.Embedding(num_embeddings, embedding_dim)`: uma matriz de pesos `V × d` em que a linha `i` é o vetor do token de ID `i`. |
| [Embedding (representação vetorial)](capitulo-02.md#embedding-representação-vetorial) | 2 | Mapeamento de objetos discretos — palavras, tokens, imagens, documentos — para pontos de um espaço vetorial contínuo de dimensão fixa. |
| [Embedding (vetor de representação)](capitulo-01.md#embedding-vetor-de-representação) | 1 | Representação de um token (ou de um texto) como um vetor de números reais em um espaço de várias dimensões, construída de forma que a geometria do espaço carregue informação — tokens com papéis parecidos ficam próximos. |
| [Emergent behavior (comportamento emergente)](capitulo-01.md#emergent-behavior-comportamento-emergente) | 1 | Capacidade que um modelo apresenta sem ter sido treinado explicitamente para ela. |
| [Encode e decode (codificação e decodificação)](capitulo-02.md#encode-e-decode-codificação-e-decodificação) | 2 | Os dois métodos de todo tokenizador. |
| [Encoder (codificador)](capitulo-01.md#encoder-codificador) | 1 | Submódulo do Transformer original que lê o texto de entrada inteiro e o converte em uma sequência de vetores que capturam o contexto de cada posição. |
| [End-of-text token (token de fim de texto)](capitulo-07.md#end-of-text-token-token-de-fim-de-texto) | 7 | O token especial `<\|endoftext\|>`, de ID 50256 no vocabulário do GPT-2. |
| [Epoch (época)](capitulo-05.md#epoch-época) | 5 | Uma passagem completa por todos os lotes do conjunto de treinamento. |
| [Epsilon (`eps`, constante de estabilidade numérica)](capitulo-04.md#epsilon-eps-constante-de-estabilidade-numérica) | 4 | Constante pequena somada à variância no denominador da [LayerNorm](capitulo-04.md#layernorm-normalização-de-camada): `(x − μ) / √(σ² + ε)`. |
| [`eval()` e `train()` (modos de execução do modelo)](capitulo-04.md#eval-e-train-modos-de-execução-do-modelo) | 4 | Métodos de todo `nn.Module` do PyTorch que alternam o modelo entre o modo de treinamento (`model.train()`, o padrão) e o modo de avaliação (`model.eval()`). |
| [Evaluation dataset (conjunto de avaliação)](capitulo-06.md#evaluation-dataset-conjunto-de-avaliação) | 6 | Parte dos dados separada do treino e usada apenas para medir o desempenho do modelo. |
| [Evaluation frequency (frequência de avaliação)](capitulo-05.md#evaluation-frequency-frequência-de-avaliação) | 5 | De quantos em quantos passos as perdas de treino e de validação são medidas durante o treinamento (`eval_freq` no livro), e sobre quantos lotes (`eval_iter`). |

## F

| Termo | Cap. | Definição |
|---|---|---|
| [FeedForward network (rede feed forward)](capitulo-04.md#feedforward-network-rede-feed-forward) | 4 | Pequena rede neural presente em cada bloco Transformer, formada por uma camada linear que expande a dimensão de `d` para `d_ff = 4 · d`, a ativação [GELU](capitulo-04.md#gelu-gaussian-error-linear-unit) e uma segunda camada linear que comprime de volta para `d`. |
| [Few-shot learning (aprendizado com poucos exemplos)](capitulo-01.md#few-shot-learning-aprendizado-com-poucos-exemplos) | 1 | Capacidade de executar uma tarefa nova a partir de um pequeno número de exemplos fornecidos **na própria entrada** do modelo, sem alterar seus pesos. |
| [Final LayerNorm (normalização final)](capitulo-04.md#final-layernorm-normalização-final) | 4 | [LayerNorm](capitulo-04.md#layernorm-normalização-de-camada) aplicada uma única vez, à saída do último bloco Transformer, antes da cabeça de saída (`final_norm`). |
| [Fine-tuning (ajuste fino)](capitulo-01.md#fine-tuning-ajuste-fino) | 1 | Segunda etapa de treinamento de um LLM: o modelo já pré-treinado continua sendo treinado, agora com um conjunto de dados **rotulado**, menor e específico de uma tarefa ou domínio. |
| [Fine-tuning (ajuste fino)](capitulo-06.md#fine-tuning-ajuste-fino) | 6 | Continuação do treinamento de um modelo já treinado, em um conjunto de dados menor e específico, para adaptá-lo a uma tarefa. |
| [Foundation model (modelo de fundação, modelo base)](capitulo-01.md#foundation-model-modelo-de-fundação-modelo-base) | 1 | Modelo resultante do [pré-treinamento](capitulo-01.md#pretraining-pré-treinamento): treinado em um corpus grande e diverso, sabe completar texto, mas ainda não foi especializado em nenhuma tarefa. |
| [Frozen layers (camadas congeladas)](capitulo-06.md#frozen-layers-camadas-congeladas) | 6 | Camadas cujos parâmetros não são atualizados durante o treinamento. |

## G

| Termo | Cap. | Definição |
|---|---|---|
| [GELU (Gaussian Error Linear Unit)](capitulo-04.md#gelu-gaussian-error-linear-unit) | 4 | Função de ativação definida como `GELU(x) = x · Φ(x)`, em que `Φ` é a função de distribuição acumulada da normal padrão. |
| [Generalization (generalização)](capitulo-05.md#generalization-generalização) | 5 | Capacidade do modelo de prever bem dados que não viu no treinamento. |
| [GPT (Generative Pretrained Transformer)](capitulo-01.md#gpt-generative-pretrained-transformer) | 1 | Família de LLMs da OpenAI, introduzida em 2018, formada **apenas pela parte de decoder** do Transformer e pré-treinada com previsão da próxima palavra. |
| [GPT model configuration (configuração do modelo)](capitulo-04.md#gpt-model-configuration-configuração-do-modelo) | 4 | Conjunto de hiperparâmetros que define a forma de um modelo GPT, passado como dicionário ao construtor do modelo: `vocab_size`, `context_length`, `emb_dim`, `n_heads`, `n_layers`, `drop_rate` e `qkv_bias`. |
| [`GPTModel` (modelo GPT)](capitulo-04.md#gptmodel-modelo-gpt) | 4 | O modelo completo do capítulo: embeddings de token e de posição, dropout, uma pilha de `n_layers` [blocos Transformer](capitulo-04.md#transformer-block-bloco-transformer), a [LayerNorm final](capitulo-04.md#final-layernorm-normalização-final) e a [cabeça de saída](capitulo-04.md#language-model-head-cabeça-de-saída-out_head). |
| [Gradient (gradiente)](capitulo-05.md#gradient-gradiente) | 5 | Vetor das derivadas parciais da perda em relação a cada peso: indica a direção em que a perda mais cresce. |
| [Greedy decoding (decodificação gulosa)](capitulo-04.md#greedy-decoding-decodificação-gulosa) | 4 | Estratégia de seleção que escolhe, a cada passo, o token de maior probabilidade — `argmax` sobre a distribuição do próximo token. |

## H

| Termo | Cap. | Definição |
|---|---|---|
| [Head dimension (`head_dim`)](capitulo-03.md#head-dimension-head_dim) | 3 | Dimensão do subespaço em que cada cabeça compara queries e keys: `head_dim = d_out // num_heads`. |

## I

| Termo | Cap. | Definição |
|---|---|---|
| [Ignore index (índice ignorado)](capitulo-07.md#ignore-index-índice-ignorado) | 7 | Valor especial do alvo que a entropia cruzada desconsidera: as posições marcadas com ele não entram na média da perda nem geram gradiente. |
| [Inference (inferência)](capitulo-04.md#inference-inferência) | 4 | Uso do modelo já construído (e, depois da Sprint 5, treinado) para produzir saídas a partir de novas entradas, sem ajustar os pesos. |
| [Information leakage (vazamento de informação)](capitulo-03.md#information-leakage-vazamento-de-informação) | 3 | Situação em que a representação de um token incorpora informação de posições que deveriam estar escondidas. |
| [Input embeddings (embeddings de entrada)](capitulo-02.md#input-embeddings-embeddings-de-entrada) | 2 | Soma dos embeddings de token com os embeddings posicionais: o tensor de forma `(b, L, d)` que efetivamente entra nas camadas principais do LLM. |
| [Input-target pairs (pares entrada-alvo)](capitulo-02.md#input-target-pairs-pares-entrada-alvo) | 2 | Pares `(x, y)` de sequências de token IDs em que o alvo `y` é a entrada `x` deslocada uma posição para a frente: `y[i] = x[i + 1]`. |
| [Instruction dataset (conjunto de dados de instruções)](capitulo-07.md#instruction-dataset-conjunto-de-dados-de-instruções) | 7 | Base supervisionada em que cada exemplo tem uma **instrução** (a tarefa, em linguagem natural), uma **entrada** opcional (o texto sobre o qual a tarefa atua) e a **resposta** esperada. |
| [Instruction fine-tuning (fine-tuning de instruções)](capitulo-07.md#instruction-fine-tuning-fine-tuning-de-instruções) | 7 | Continuação do treinamento de um modelo pré-treinado sobre pares (instrução, resposta), para que ele passe a **atender** a pedidos em vez de apenas continuar o texto. |
| [Instruction fine-tuning (fine-tuning por instruções)](capitulo-01.md#instruction-fine-tuning-fine-tuning-por-instruções) | 1 | Modalidade de [fine-tuning](capitulo-01.md#fine-tuning-ajuste-fino) em que o conjunto rotulado é formado por pares **instrução → resposta**. |

## K

| Termo | Cap. | Definição |
|---|---|---|
| [Key (chave)](capitulo-03.md#key-chave) | 3 | Projeção do embedding de entrada por `Wk`: `kᵢ = x⁽ⁱ⁾ · Wk`. |

## L

| Termo | Cap. | Definição |
|---|---|---|
| [Labeled and unlabeled data (dados rotulados e não rotulados)](capitulo-01.md#labeled-and-unlabeled-data-dados-rotulados-e-não-rotulados) | 1 | Dados **rotulados** acompanham cada exemplo da resposta esperada (uma classe, uma tradução, uma resposta). |
| [Language Model Head (cabeça de saída, `out_head`)](capitulo-04.md#language-model-head-cabeça-de-saída-out_head) | 4 | Camada linear final do GPT, sem viés, que projeta cada vetor de dimensão `d` no espaço do vocabulário, de dimensão `V`: `nn.Linear(d, V, bias=False)`. |
| [Language tasks (tarefas de linguagem)](capitulo-01.md#language-tasks-tarefas-de-linguagem) | 1 | Ações específicas que um sistema executa para processar, entender ou gerar linguagem humana: tradução, resumo, análise de sentimento, resposta a perguntas, redação, extração de informação. |
| [Large Language Model (LLM, modelo de linguagem de grande escala)](capitulo-01.md#large-language-model-llm-modelo-de-linguagem-de-grande-escala) | 1 | [Rede neural](capitulo-01.md#neural-network-rede-neural) profunda, baseada na arquitetura [Transformer](capitulo-01.md#transformer-arquitetura-transformer), treinada em grandes volumes de texto para entender, gerar e responder a textos em linguagem humana. |
| [Last-token representation (representação do último token)](capitulo-06.md#last-token-representation-representação-do-último-token) | 6 | O vetor de saída do modelo na última posição da sequência, usado como resumo do texto inteiro para a classificação. |
| [LayerNorm (normalização de camada)](capitulo-04.md#layernorm-normalização-de-camada) | 4 | Normalização que ajusta o vetor de cada token para média 0 e variância 1 ao longo das suas `d` características e, em seguida, aplica uma reescala e um deslocamento treináveis: `y = scale · (x − μ) / √(σ² + ε) + shift`. |
| [Learning rate (taxa de aprendizado)](capitulo-05.md#learning-rate-taxa-de-aprendizado) | 5 | O fator `η` que multiplica a atualização dos pesos em cada passo do otimizador. |
| [Logits](capitulo-04.md#logits) | 4 | Pontuações não normalizadas que o modelo atribui a cada token do vocabulário, para cada posição da sequência. |

## M

| Termo | Cap. | Definição |
|---|---|---|
| [Machine learning (aprendizado de máquina)](capitulo-01.md#machine-learning-aprendizado-de-máquina) | 1 | Ramo da [IA](capitulo-01.md#artificial-intelligence-inteligência-artificial) que desenvolve algoritmos capazes de aprender a partir de dados e fazer previsões ou tomar decisões **sem serem programados explicitamente** com as regras da tarefa. |
| [Memorization (memorização)](capitulo-05.md#memorization-memorização) | 5 | Quando o modelo reproduz trechos literais dos dados de treinamento em vez de produzir texto novo plausível. |
| [Multi-head attention (atenção multi-cabeça)](capitulo-03.md#multi-head-attention-atenção-multi-cabeça) | 3 | Execução de vários mecanismos de atenção em paralelo sobre a mesma sequência, cada um com suas próprias matrizes `Wq`, `Wk` e `Wv`, com as saídas concatenadas. |
| [`MultiHeadAttention` (divisão de pesos)](capitulo-03.md#multiheadattention-divisão-de-pesos) | 3 | Implementação eficiente (listagem 3.5): uma única projeção linear de tamanho `d_out` por papel, cujo resultado é fatiado em `num_heads` blocos de `head_dim` colunas por `view` e `transpose`. |
| [`MultiHeadAttentionWrapper` (empilhamento de cabeças)](capitulo-03.md#multiheadattentionwrapper-empilhamento-de-cabeças) | 3 | Implementação didática (listagem 3.4): uma `nn.ModuleList` de `num_heads` módulos `CausalAttention` independentes, com as saídas concatenadas na última dimensão. |
| [Multinomial sampling (amostragem multinomial)](capitulo-04.md#multinomial-sampling-amostragem-multinomial) | 4 | Seleção aleatória do próximo token em que cada token é sorteado com probabilidade igual à que o modelo lhe atribui — `torch.multinomial(probs, num_samples=1)`. |

## N

| Termo | Cap. | Definição |
|---|---|---|
| [Natural Language Processing (NLP, processamento de linguagem natural)](capitulo-01.md#natural-language-processing-nlp-processamento-de-linguagem-natural) | 1 | Área da computação dedicada a fazer computadores lerem, interpretarem e produzirem linguagem humana. |
| [Neural network (rede neural)](capitulo-01.md#neural-network-rede-neural) | 1 | Modelo computacional formado por camadas de unidades que aplicam transformações lineares (pesos e vieses) seguidas de funções não lineares. |
| [Next-token prediction (previsão do próximo token)](capitulo-02.md#next-token-prediction-previsão-do-próximo-token) | 2 | Tarefa de treinamento em que o modelo recebe uma sequência de tokens e deve prever o token seguinte. |
| [Next-token probabilities (probabilidades do próximo token)](capitulo-04.md#next-token-probabilities-probabilidades-do-próximo-token) | 4 | Distribuição de probabilidade sobre o vocabulário obtida aplicando o [softmax](capitulo-03.md#softmax) aos [logits](capitulo-04.md#logits) de uma posição: `V` valores positivos que somam 1. |
| [Next-word prediction (previsão da próxima palavra)](capitulo-01.md#next-word-prediction-previsão-da-próxima-palavra) | 1 | Tarefa de prever a palavra (ou, mais precisamente, o token) seguinte de uma sequência a partir das anteriores. |

## O

| Termo | Cap. | Definição |
|---|---|---|
| [Optimizer (otimizador)](capitulo-05.md#optimizer-otimizador) | 5 | Algoritmo que atualiza os pesos do modelo a partir dos gradientes, segundo uma regra própria (SGD, Adam, AdamW…). |
| [Output projection (`out_proj`)](capitulo-03.md#output-projection-out_proj) | 3 | Camada `nn.Linear(d_out, d_out)` aplicada depois de reunir as saídas das cabeças. |
| [Overfitting (sobreajuste)](capitulo-05.md#overfitting-sobreajuste) | 5 | Situação em que o modelo continua melhorando nos dados de treino enquanto piora — ou para de melhorar — nos de validação: ajusta-se às particularidades do conjunto de treino em vez dos padrões gerais da língua. |

## P

| Termo | Cap. | Definição |
|---|---|---|
| [Padding (preenchimento)](capitulo-06.md#padding-preenchimento) | 6 | Acréscimo de um token sem conteúdo ao fim das sequências curtas, para que todas as de um lote tenham o mesmo comprimento. |
| [Padding token (token de preenchimento)](capitulo-07.md#padding-token-token-de-preenchimento) | 7 | Token acrescentado ao final das sequências curtas para que todas as do lote tenham o mesmo comprimento. |
| [Parameter count (contagem de parâmetros e memória)](capitulo-04.md#parameter-count-contagem-de-parâmetros-e-memória) | 4 | Número total de pesos treináveis de um modelo, obtido somando o número de elementos de todos os seus tensores de parâmetros (`sum(p.numel() for p in model.parameters())`); multiplicado pelo tamanho de cada número, dá a memória ocupada pelo modelo. |
| [Parameters (parâmetros, pesos)](capitulo-01.md#parameters-parâmetros-pesos) | 1 | Valores numéricos ajustáveis de uma [rede neural](capitulo-01.md#neural-network-rede-neural) — pesos e vieses — que são otimizados durante o treinamento. |
| [Perplexity (perplexidade)](capitulo-05.md#perplexity-perplexidade) | 5 | A exponencial da perda de entropia cruzada: `perplexidade = exp(ℒ)`. |
| [Positional embedding (embedding posicional)](capitulo-02.md#positional-embedding-embedding-posicional) | 2 | Vetor que representa a posição de um token na sequência e é somado ao embedding do token. |
| [Precision, recall e F1-score (precisão, revocação e F1)](capitulo-06.md#precision-recall-e-f1-score-precisão-revocação-e-f1) | 6 | Três métricas calculadas a partir da [matriz de confusão](capitulo-06.md#confusion-matrix-matriz-de-confusão), tomando uma classe como positiva (aqui, spam): |
| [Pre-LayerNorm e Post-LayerNorm (posição da normalização)](capitulo-04.md#pre-layernorm-e-post-layernorm-posição-da-normalização) | 4 | As duas posições possíveis da LayerNorm em um bloco Transformer. |
| [Pretrained weights (pesos pré-treinados)](capitulo-06.md#pretrained-weights-pesos-pré-treinados) | 6 | Valores dos parâmetros de um modelo obtidos em um treinamento anterior, em geral longo e sobre muito texto, e distribuídos em arquivo para serem reutilizados. |
| [Pretraining (pré-treinamento)](capitulo-01.md#pretraining-pré-treinamento) | 1 | Primeira etapa de treinamento de um LLM: o modelo é treinado em um corpus grande e diverso de texto **não rotulado**, com a tarefa de [previsão da próxima palavra](capitulo-01.md#next-word-prediction-previsão-da-próxima-palavra). |
| [Prompt format (formato do prompt)](capitulo-07.md#prompt-format-formato-do-prompt) | 7 | O molde de texto que transforma um exemplo `{instruction, input, output}` numa única sequência. |

## Q

| Termo | Cap. | Definição |
|---|---|---|
| [Query (consulta)](capitulo-03.md#query-consulta) | 3 | Projeção do embedding de entrada por `Wq`: `qᵢ = x⁽ⁱ⁾ · Wq`. |

## R

| Termo | Cap. | Definição |
|---|---|---|
| [Raw text (texto bruto)](capitulo-02.md#raw-text-texto-bruto) | 2 | O texto como sequência de caracteres, antes de qualquer processamento — por exemplo, os 20.479 caracteres do conto *The Verdict*, lidos de `sprints/the-verdict.txt`. |
| [`register_buffer`](capitulo-03.md#register_buffer) | 3 | Método do `nn.Module` que registra um tensor como estado do módulo sem torná-lo parâmetro treinável. |
| [ReLU (Rectified Linear Unit)](capitulo-04.md#relu-rectified-linear-unit) | 4 | Função de ativação `ReLU(x) = max(0, x)`: devolve a entrada quando ela é positiva e zero caso contrário. |
| [Residual connection (conexão residual, shortcut ou skip connection)](capitulo-04.md#residual-connection-conexão-residual-shortcut-ou-skip-connection) | 4 | Ligação que soma a entrada de uma camada (ou bloco de camadas) à sua saída: `y = x + F(x)`. |
| [Response evaluation (avaliação das respostas)](capitulo-07.md#response-evaluation-avaliação-das-respostas) | 7 | Medição da qualidade das respostas geradas pelo modelo ajustado, comparando-as com as respostas esperadas de um conjunto que ele não viu no treino. |

## S

| Termo | Cap. | Definição |
|---|---|---|
| [Scale e shift (parâmetros treináveis da LayerNorm)](capitulo-04.md#scale-e-shift-parâmetros-treináveis-da-layernorm) | 4 | Os dois vetores treináveis da LayerNorm, ambos de dimensão `d`: `scale` (γ), que multiplica o vetor normalizado, e `shift` (β), que é somado a ele. |
| [Scaled dot-product attention (atenção por produto escalar escalado)](capitulo-03.md#scaled-dot-product-attention-atenção-por-produto-escalar-escalado) | 3 | A self-attention com pesos treináveis usada no Transformer e nos modelos GPT: |
| [Scaling factor (fator de escala, `1/√d_k`)](capitulo-03.md#scaling-factor-fator-de-escala-1d_k) | 3 | Divisão dos scores de atenção pela raiz quadrada da dimensão das keys, antes do softmax. |
| [Self-attention (autoatenção)](capitulo-01.md#self-attention-autoatenção) | 1 | Mecanismo que permite a cada token de uma sequência ponderar a importância de todos os outros tokens da mesma sequência ao construir sua própria representação. |
| [Self-attention (autoatenção)](capitulo-03.md#self-attention-autoatenção) | 3 | Mecanismo em que cada posição de uma sequência atende a todas as posições *da mesma sequência* para construir sua representação. |
| [`SelfAttentionV1` e `SelfAttentionV2`](capitulo-03.md#selfattentionv1-e-selfattentionv2) | 3 | Duas implementações da mesma self-attention: a primeira declara `Wq`, `Wk` e `Wv` como `nn.Parameter(torch.rand(...))`; a segunda usa `nn.Linear(..., bias=False)`. |
| [Self-supervised learning (aprendizado autossupervisionado)](capitulo-01.md#self-supervised-learning-aprendizado-autossupervisionado) | 1 | Forma de aprendizado em que os rótulos não são coletados à parte: são gerados automaticamente a partir da estrutura dos próprios dados. |
| [Sliding window (janela deslizante)](capitulo-02.md#sliding-window-janela-deslizante) | 2 | Estratégia de amostragem que percorre a sequência de token IDs com uma janela de tamanho `L`, avançando `S` posições por vez e recortando um par entrada-alvo em cada posição. |
| [Softmax](capitulo-03.md#softmax) | 3 | Função que transforma um vetor de números reais em uma distribuição de probabilidade: `softmax(x)ᵢ = e^{xᵢ} / Σⱼ e^{xⱼ}`. |
| [Special tokens (tokens especiais)](capitulo-02.md#special-tokens-tokens-especiais) | 2 | Tokens que não vêm do texto, adicionados ao vocabulário para sinalizar algo ao modelo. |
| [`state_dict` (dicionário de estado)](capitulo-05.md#state_dict-dicionário-de-estado) | 5 | Dicionário do PyTorch que associa o nome de cada parâmetro e buffer de um módulo (ou de um otimizador) ao seu tensor. |
| [Straightforward pattern recognition (reconhecimento de padrões simples)](capitulo-01.md#straightforward-pattern-recognition-reconhecimento-de-padrões-simples) | 1 | Identificação de regularidades nos dados por meio de regras fixas ou comparações diretas, que podem ser escritas à mão ou capturadas por modelos simples. |
| [Stride (passo)](capitulo-02.md#stride-passo) | 2 | Número de posições que a janela deslizante avança entre uma amostra e a seguinte. |
| [Subword (subpalavra)](capitulo-02.md#subword-subpalavra) | 2 | Unidade de tokenização intermediária entre caractere e palavra: um pedaço de palavra, como `"token"` + `"ization"`. |
| [Supervised fine-tuning (fine-tuning supervisionado)](capitulo-06.md#supervised-fine-tuning-fine-tuning-supervisionado) | 6 | [Fine-tuning](capitulo-06.md#fine-tuning-ajuste-fino) feito com exemplos **rotulados**: cada entrada vem acompanhada da saída correta, fornecida por uma pessoa. |

## T

| Termo | Cap. | Definição |
|---|---|---|
| [Temperature (temperatura)](capitulo-04.md#temperature-temperatura) | 4 | Número positivo `T` pelo qual os logits são divididos antes do softmax: `probs = softmax(logits / T)`. |
| [tiktoken](capitulo-02.md#tiktoken) | 2 | Biblioteca de código aberto da OpenAI que implementa tokenizadores BPE, com núcleo em Rust. |
| [Token](capitulo-02.md#token) | 2 | Unidade mínima em que o texto é dividido para ser processado pelo modelo. |
| [Token embedding (embedding de token)](capitulo-02.md#token-embedding-embedding-de-token) | 2 | Vetor de dimensão `d` associado a um token ID, obtido pela consulta à camada de embedding de token. |
| [Token ID (identificador de token)](capitulo-02.md#token-id-identificador-de-token) | 2 | Inteiro único que o vocabulário associa a cada token — o índice da linha correspondente na matriz de embedding. |
| [Token (unidade de texto)](capitulo-01.md#token-unidade-de-texto) | 1 | Unidade de texto que o modelo lê e gera. |
| [Tokenization (tokenização)](capitulo-02.md#tokenization-tokenização) | 2 | Processo de dividir o texto bruto em tokens. |
| [Tokenizer (tokenizador)](capitulo-02.md#tokenizer-tokenizador) | 2 | Componente que implementa a tokenização e a conversão entre tokens e IDs, expondo os métodos `encode` e `decode`. |
| [Tokens seen (tokens vistos)](capitulo-05.md#tokens-seen-tokens-vistos) | 5 | Total de tokens de entrada processados pelo treinamento até um dado passo: `passos · b · L`. |
| [Top-k sampling (amostragem top-k)](capitulo-04.md#top-k-sampling-amostragem-top-k) | 4 | Estratégia que restringe a escolha do próximo token aos `k` tokens de maiores logits: os demais logits são substituídos por `−∞` antes do softmax, recebendo probabilidade exatamente zero, e a probabilidade se redistribui entre os `k` restantes. |
| [Trainable parameters (parâmetros treináveis)](capitulo-06.md#trainable-parameters-parâmetros-treináveis) | 6 | Os parâmetros que o otimizador atualiza durante o treinamento: aqueles com `requires_grad = True`. |
| [Training batch (lote de treinamento)](capitulo-06.md#training-batch-lote-de-treinamento) | 6 | Grupo de exemplos processados juntos em um passo de treinamento. |
| [Training corpus (corpus de treinamento)](capitulo-01.md#training-corpus-corpus-de-treinamento) | 1 | Coleção de textos usada para treinar o modelo. |
| [Training loop (laço de treinamento)](capitulo-05.md#training-loop-laço-de-treinamento) | 5 | O procedimento que repete, para cada [época](capitulo-05.md#epoch-época) e cada lote: zerar os gradientes, calcular a perda, retropropagar e atualizar os pesos — avaliando periodicamente. |
| [Training loss (perda de treinamento)](capitulo-05.md#training-loss-perda-de-treinamento) | 5 | A entropia cruzada medida nos dados de treino. |
| [Training step (passo de treinamento)](capitulo-05.md#training-step-passo-de-treinamento) | 5 | Uma única atualização dos pesos, feita a partir de um lote. |
| [Train/validation split (divisão treino/validação)](capitulo-05.md#trainvalidation-split-divisão-treinovalidação) | 5 | Separação do corpus em um conjunto usado para ajustar os pesos (treinamento) e outro, nunca usado para atualizá-los, que serve só para medir o desempenho (validação). |
| [Transfer learning (aprendizado por transferência)](capitulo-06.md#transfer-learning-aprendizado-por-transferência) | 6 | Aproveitamento, em uma tarefa nova, do conhecimento que um modelo adquiriu em outra tarefa, em vez de aprender tudo do zero. |
| [Transformer](capitulo-04.md#transformer) | 4 | Arquitetura de rede neural proposta em 2017, no artigo *Attention Is All You Need*, baseada inteiramente em mecanismos de atenção, sem recorrência. |
| [Transformer (arquitetura Transformer)](capitulo-01.md#transformer-arquitetura-transformer) | 1 | Arquitetura de rede neural profunda introduzida em 2017 no artigo *Attention Is All You Need*, originalmente para tradução automática. |
| [Transformer block (bloco Transformer)](capitulo-03.md#transformer-block-bloco-transformer) | 3 | Unidade que combina multi-head attention com normalização, rede *feed-forward* e conexões residuais. |
| [Transformer block (bloco Transformer)](capitulo-04.md#transformer-block-bloco-transformer) | 4 | Unidade que se repete ao longo do GPT, formada por duas subcamadas, cada uma precedida por uma LayerNorm, seguida de dropout e envolvida por uma conexão residual: |

## U

| Termo | Cap. | Definição |
|---|---|---|
| [Unknown token (token desconhecido)](capitulo-02.md#unknown-token-token-desconhecido) | 2 | O token especial `<\|unk\|>`, que substitui qualquer palavra ausente do vocabulário (*out-of-vocabulary*). |

## V

| Termo | Cap. | Definição |
|---|---|---|
| [Validation loss (perda de validação)](capitulo-05.md#validation-loss-perda-de-validação) | 5 | A entropia cruzada medida nos dados de validação, que o modelo nunca usa para ajustar os pesos. |
| [Value (valor)](capitulo-03.md#value-valor) | 3 | Projeção do embedding de entrada por `Wv`: `vᵢ = x⁽ⁱ⁾ · Wv`. |
| [Vanishing gradient (desaparecimento do gradiente)](capitulo-04.md#vanishing-gradient-desaparecimento-do-gradiente) | 4 | Fenômeno em que os gradientes ficam progressivamente menores à medida que são retropropagados da saída para as primeiras camadas de uma rede profunda, até se tornarem pequenos demais para atualizar os pesos de forma útil. |
| [Vocabulary (vocabulário)](capitulo-02.md#vocabulary-vocabulário) | 2 | Mapeamento entre o conjunto de tokens possíveis e os inteiros que os identificam. |

## W

| Termo | Cap. | Definição |
|---|---|---|
| [Weight decay (decaimento de pesos)](capitulo-05.md#weight-decay-decaimento-de-pesos) | 5 | Termo que reduz cada peso em uma pequena fração a cada passo (`θ ← θ − η·λ·θ`), puxando-o para zero. |
| [Weight parameter (parâmetro de peso)](capitulo-03.md#weight-parameter-parâmetro-de-peso) | 3 | Os coeficientes treináveis das matrizes `Wq`, `Wk` e `Wv`, ajustados por retropropagação durante o treinamento. |
| [Weight tying (compartilhamento de pesos)](capitulo-04.md#weight-tying-compartilhamento-de-pesos) | 4 | Reutilização da mesma matriz de pesos na tabela de [embedding de token](capitulo-02.md#token-embedding-embedding-de-token) e na [cabeça de saída](capitulo-04.md#language-model-head-cabeça-de-saída-out_head). |
| [Word2Vec](capitulo-02.md#word2vec) | 2 | Método de 2013 para gerar embeddings de palavras treinando uma rede neural rasa a prever as palavras vizinhas de uma palavra (ou a palavra a partir das vizinhas). |

## Z

| Termo | Cap. | Definição |
|---|---|---|
| [`zero_grad` (zerar gradientes)](capitulo-05.md#zero_grad-zerar-gradientes) | 5 | Método do otimizador que apaga os gradientes acumulados nos pesos. |
| [Zero-shot learning (aprendizado sem exemplos)](capitulo-01.md#zero-shot-learning-aprendizado-sem-exemplos) | 1 | Capacidade de executar uma tarefa nunca vista, a partir apenas da descrição da tarefa, sem nenhum exemplo na entrada e sem treinamento adicional. |
