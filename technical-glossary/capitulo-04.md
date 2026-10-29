# Capítulo 4 — Implementing a GPT Model from Scratch to Generate Text

Glossário dos termos introduzidos no Capítulo 4 de RASCHKA, S. *Build a Large Language Model
(From Scratch)*. Entradas em ordem alfabética.

O capítulo junta as peças das sprints anteriores — embeddings (Capítulo 2) e atenção multi-cabeça
(Capítulo 3) — em um modelo GPT completo, capaz de receber token IDs e devolver uma previsão para o
próximo token:

```
token IDs (b, L)
  → embedding de token + embedding posicional        (b, L, d)
  → dropout
  → n_layers × bloco Transformer                     (b, L, d)
  → LayerNorm final                                  (b, L, d)
  → cabeça de saída (out_head)                       (b, L, V)   ← logits
  → softmax e seleção do token, na última posição    (b, 1)      → anexado à entrada
```

Salvo indicação em contrário, os números dos exemplos são do GPT-2 small, a configuração
`GPT_CONFIG_124M` do livro (no projeto, `CONFIG_GPT2_124M` em `src/gpt.py`). Os termos *temperatura*,
*top-k* e *amostragem multinomial* vêm da seção 5.3 do livro; entram aqui porque a instrução da
Sprint 4 pede que as estratégias de seleção de tokens sejam exploradas junto com a geração.

Notação usada nas fórmulas:

| Símbolo | Significado |
|---|---|
| `b` | tamanho do lote (`batch_size`) |
| `L` | número de tokens da sequência (`num_tokens`), no máximo `C` |
| `C` | comprimento máximo do contexto (`context_length`) — 1.024 no GPT-2 |
| `d` | dimensão do embedding (`emb_dim`) — 768 no GPT-2 small |
| `V` | tamanho do vocabulário (`vocab_size`) — 50.257 no GPT-2 |
| `n_heads` | número de cabeças de atenção — 12 no GPT-2 small |
| `n_layers` | número de blocos Transformer empilhados — 12 no GPT-2 small |
| `d_ff` | dimensão interna da rede feed forward — `4 · d` = 3.072 no GPT-2 small |

---

## Autoregressive generation (geração autoregressiva)

**Definição.** Procedimento de geração em que o modelo produz um token por vez e cada token produzido
é anexado à entrada antes da previsão seguinte. A sequência cresce de um token a cada iteração, até
atingir o número de tokens pedido (`max_new_tokens`) ou um critério de parada.

**Função no modelo.** É como o GPT escreve texto. O modelo, sozinho, só sabe responder "qual é o
próximo token?"; a geração autoregressiva transforma essa resposta pontual em um texto inteiro,
repetindo a pergunta com o contexto ampliado. Cada passo do laço é: [cortar o
contexto](#context-cropping-corte-do-contexto-na-geração), executar o modelo, pegar os
[logits](#logits) da última posição, convertê-los em
[probabilidades](#next-token-probabilities-probabilidades-do-próximo-token), escolher o token e
concatená-lo à sequência.

**Relação com outros conceitos.** É a contrapartida, na [inferência](#inference-inferência), da
[previsão do próximo token](capitulo-02.md#next-token-prediction-previsão-do-próximo-token) usada no
treinamento, e só é possível porque o modelo é
[somente decodificador](#decoder-only-arquitetura-somente-decodificador), com máscara causal. A escolha
do token em cada passo pode ser [gulosa](#greedy-decoding-decodificação-gulosa) ou por
[amostragem](#multinomial-sampling-amostragem-multinomial), com
[temperatura](#temperature-temperatura) e [top-k](#top-k-sampling-amostragem-top-k). Ver também
[modelo autorregressivo](capitulo-01.md#autoregressive-model-modelo-autorregressivo) no Capítulo 1.

**Exemplo.** `"Hello, I am"` vira os IDs `[15496, 11, 314, 716]`, um tensor `(1, 4)`. Com
`max_new_tokens=6`, o laço roda seis vezes e a saída tem forma `(1, 10)`; a cada iteração a entrada
ganha uma coluna — `(1, 4) → (1, 5) → … → (1, 10)`. No projeto, `gerar_texto_simples` e `gerar`, em
`src/geracao.py`, implementam esse laço. Com pesos aleatórios, o texto produzido não tem sentido: o
procedimento funciona, mas o modelo ainda não aprendeu nada.

---

## Batch normalization (normalização em lote)

**Definição.** Técnica de normalização que calcula média e variância de cada característica *ao longo
do lote*, isto é, comparando os exemplos do lote entre si, e usa essas estatísticas para normalizar a
ativação.

**Função no modelo.** Não é usada no GPT. Aparece no capítulo como contraste com a
[LayerNorm](#layernorm-normalização-de-camada): como depende das outras amostras do lote, seu
comportamento muda com o tamanho do lote e fica instável com lotes pequenos — justamente o caso da
geração de texto, que costuma processar uma única sequência.

**Relação com outros conceitos.** A LayerNorm normaliza na direção oposta: cada token é normalizado
sobre as suas próprias `d` características, sem olhar o resto do lote. Por isso ela funciona igual
com `b = 1` ou `b = 512`, o que também facilita o treinamento distribuído.

**Exemplo.** Em um tensor `(b, L, d)`, a batch normalization calcularia, para cada uma das `d`
características, uma média sobre as `b · L` posições do lote; a LayerNorm calcula, para cada uma das
`b · L` posições, uma média sobre as `d` características.

---

## Context cropping (corte do contexto na geração)

**Definição.** Passo da geração que mantém apenas os últimos `C` tokens da sequência antes de cada
chamada ao modelo — no código do livro, `idx[:, -context_size:]`.

**Função no modelo.** O modelo só aceita sequências de até `context_length` tokens, porque a tabela de
[embedding posicional](capitulo-02.md#positional-embedding-embedding-posicional) tem exatamente `C`
linhas (e a máscara causal da atenção, `C × C` posições). Como a
[geração autoregressiva](#autoregressive-generation-geração-autoregressiva) faz a sequência crescer
indefinidamente, o corte garante que a entrada nunca ultrapasse esse limite: os tokens mais antigos
simplesmente deixam de ser vistos.

**Relação com outros conceitos.** O limite é definido pela
[configuração do modelo](#gpt-model-configuration-configuração-do-modelo) e corresponde ao
[comprimento do contexto](capitulo-02.md#context-length-comprimento-do-contexto-max_length) da
Sprint 2. É a razão concreta pela qual um LLM "esquece" o início de textos muito longos.

**Exemplo.** Com `C = 1024`, quando a sequência gerada chega a 1.030 tokens, o modelo recebe só as
posições 6 a 1.029. Com um modelo configurado para `C = 4`, uma sequência `[a, b, c, d, e]` seria
cortada para `[b, c, d, e]` antes da próxima previsão.

---

## Decoder-only (arquitetura somente decodificador)

**Definição.** Arquitetura Transformer formada apenas por blocos do tipo decodificador — sem o
codificador e sem a atenção cruzada entre os dois — e com atenção causal, que faz cada posição
enxergar somente a si mesma e às anteriores.

**Função no modelo.** É a arquitetura do GPT. Sem codificador, o modelo não recebe uma "frase de
origem" separada: tudo o que ele sabe da tarefa está no próprio texto de entrada, lido da esquerda
para a direita. É essa leitura unidirecional que permite gerar texto um token por vez.

**Relação com outros conceitos.** É uma simplificação do [Transformer](#transformer) original, que
tinha [codificador e decodificador](capitulo-01.md#decoder-decodificador). A unidirecionalidade vem da
[máscara causal](capitulo-03.md#causal-mask-máscara-causal) do Capítulo 3, aplicada dentro de cada
[bloco Transformer](#transformer-block-bloco-transformer). O BERT faz a escolha oposta: só o
codificador, com atenção nos dois sentidos.

**Exemplo.** Em `"Every effort moves you"`, o vetor de contexto calculado para `"moves"` combina
apenas `"Every"`, `" effort"` e `" moves"`; o token `" you"` só influencia a sua própria posição.

---

## Dropout (no modelo GPT)

**Definição.** Regularização que, durante o treinamento, zera aleatoriamente uma fração `p` dos
elementos de um tensor e multiplica os restantes por `1/(1 − p)`, preservando a escala esperada da
saída.

**Função no modelo.** Evita que o modelo dependa de um conjunto fixo de ativações e, com isso, reduz o
sobreajuste. No GPT o dropout aparece em três lugares, todos com a taxa `drop_rate`: logo depois da
soma dos embeddings de token e de posição (`drop_emb`); sobre os pesos de atenção, dentro da atenção
multi-cabeça; e na saída de cada subcamada do bloco, antes da soma residual (no PDF do projeto,
`drop_resid`).

**Relação com outros conceitos.** O uso dentro da atenção foi apresentado no
[Capítulo 3](capitulo-03.md#dropout). Só age em modo de treino: [`eval()`](#eval-e-train-modos-de-execução-do-modelo)
o desliga, o que é obrigatório na [inferência](#inference-inferência) para que a geração seja
reprodutível. O exercício 4.3 propõe uma taxa separada para cada um dos três lugares.

**Exemplo.** Com `drop_rate = 0.1`, cerca de 10% dos elementos do tensor `(b, L, d)` que sai dos
embeddings são zerados a cada passagem de treino, e os demais são multiplicados por `1/0,9 ≈ 1,11`.
Não há parâmetros: o dropout não altera a contagem de parâmetros do modelo.

---

## Epsilon (`eps`, constante de estabilidade numérica)

**Definição.** Constante pequena somada à variância no denominador da
[LayerNorm](#layernorm-normalização-de-camada): `(x − μ) / √(σ² + ε)`. No livro, `ε = 1e-5`.

**Função no modelo.** Impede a divisão por zero — e o aparecimento de `inf` ou `NaN` — quando todas as
características de um token têm o mesmo valor e a variância é nula. Como é muito menor que as
variâncias típicas, não altera o resultado nos casos normais.

**Relação com outros conceitos.** Faz parte da LayerNorm, ao lado dos parâmetros
[scale e shift](#scale-e-shift-parâmetros-treináveis-da-layernorm). Não é treinável: é um
hiperparâmetro fixo, igual em todas as LayerNorms do modelo.

**Exemplo.** Para o vetor `[2, 2, 2]`, a média é 2 e a variância é 0. Sem `ε`, a normalização calcula
`0 / 0`; com `ε`, o resultado é `[0, 0, 0]`, que a LayerNorm então reescala e desloca.

---

## `eval()` e `train()` (modos de execução do modelo)

**Definição.** Métodos de todo `nn.Module` do PyTorch que alternam o modelo entre o modo de
treinamento (`model.train()`, o padrão) e o modo de avaliação (`model.eval()`). O modo altera o
comportamento de camadas como o [dropout](#dropout-no-modelo-gpt).

**Função no modelo.** Em modo de avaliação o dropout é desligado e a saída do modelo passa a ser
determinística: a mesma entrada produz sempre os mesmos [logits](#logits). O livro chama
`model.eval()` antes de gerar texto por esse motivo.

**Relação com outros conceitos.** É um dos dois cuidados da [inferência](#inference-inferência). O
outro, `torch.no_grad()`, é independente: não muda o resultado, apenas dispensa o registro do grafo
de gradientes, o que economiza memória e tempo. Esquecer o `eval()` não causa erro — causa uma geração
que muda a cada execução.

**Exemplo.** Com `drop_rate = 0.1`, duas chamadas `model(batch)` em modo de treino devolvem logits
diferentes; depois de `model.eval()`, as duas chamadas devolvem tensores idênticos.

---

## FeedForward network (rede feed forward)

**Definição.** Pequena rede neural presente em cada bloco Transformer, formada por uma camada linear
que expande a dimensão de `d` para `d_ff = 4 · d`, a ativação [GELU](#gelu-gaussian-error-linear-unit)
e uma segunda camada linear que comprime de volta para `d`.

**Função no modelo.** Processa cada token **individualmente** — a mesma rede é aplicada a cada posição,
sem trocar informação entre posições. É o complemento da atenção: a atenção mistura informação
*entre* tokens; a feed forward transforma a representação *de cada* token, em um espaço
intermediário maior, onde a não linearidade da GELU permite representar relações mais ricas. Como
devolve a mesma dimensão que recebe, os blocos podem ser empilhados sem ajuste.

**Relação com outros conceitos.** É a segunda subcamada do
[bloco Transformer](#transformer-block-bloco-transformer), precedida por uma
[LayerNorm](#layernorm-normalização-de-camada) e envolvida por uma
[conexão residual](#residual-connection-conexão-residual-shortcut-ou-skip-connection). Concentra a
maior parte dos parâmetros do bloco — cerca do dobro da
[atenção multi-cabeça](capitulo-03.md#multi-head-attention-atenção-multi-cabeça) (exercício 4.1).

**Exemplo.** No GPT-2 small: `(b, L, 768) → Linear → (b, L, 3072) → GELU → Linear → (b, L, 768)`.
Parâmetros: `768·3072 + 3072 + 3072·768 + 768 = 4.722.432`, contra 2.360.064 da atenção do mesmo
bloco. Nos 12 blocos, a feed forward soma 56,7 milhões de parâmetros e a atenção, 28,3 milhões.

---

## Final LayerNorm (normalização final)

**Definição.** [LayerNorm](#layernorm-normalização-de-camada) aplicada uma única vez, à saída do
último bloco Transformer, antes da cabeça de saída (`final_norm`).

**Função no modelo.** Padroniza as representações que chegam à
[cabeça de saída](#language-model-head-cabeça-de-saída-out_head). Como cada bloco
[pre-LayerNorm](#pre-layernorm-e-post-layernorm-posição-da-normalização) normaliza só a entrada das
suas subcamadas, o fluxo residual que sai do último bloco é a soma de muitas contribuições sem
normalização; a LayerNorm final coloca essa soma em escala controlada antes do cálculo dos logits.

**Relação com outros conceitos.** Completa o padrão pre-LayerNorm: sem ela, a saída do último bloco
chegaria crua à camada linear de saída. Tem `2 · d` parâmetros
([scale e shift](#scale-e-shift-parâmetros-treináveis-da-layernorm)).

**Exemplo.** No GPT-2 small, recebe e devolve `(b, L, 768)` e tem 1.536 parâmetros — 0,001% do total
do modelo.

---

## GELU (Gaussian Error Linear Unit)

**Definição.** Função de ativação definida como `GELU(x) = x · Φ(x)`, em que `Φ` é a função de
distribuição acumulada da normal padrão. Na prática, o GPT-2 usa uma aproximação mais barata:
`0,5 · x · (1 + tanh[√(2/π) · (x + 0,044715 · x³)])`.

**Função no modelo.** É a não linearidade da [rede feed forward](#feedforward-network-rede-feed-forward).
Sem ela, as duas camadas lineares da feed forward se reduziriam a uma única transformação linear. Por
ser suave e por deixar passar valores negativos pequenos, a GELU dá gradiente diferente de zero também
para entradas negativas, o que tende a facilitar a otimização.

**Relação com outros conceitos.** Substitui a [ReLU](#relu-rectified-linear-unit), que zera tudo o que
é negativo e tem um "canto" em zero. Para entradas positivas grandes, as duas praticamente coincidem.
O livro menciona também a SwiGLU, outra ativação suave usada em LLMs mais recentes.

**Exemplo.** Valores da aproximação: `GELU(−3) ≈ −0,004`, `GELU(−1) ≈ −0,159`, `GELU(0) = 0`,
`GELU(1) ≈ 0,841`, `GELU(3) ≈ 2,996`. A ReLU daria `0, 0, 0, 1, 3`. O menor valor da GELU é cerca de
`−0,17`, perto de `x = −0,75`.

---

## GPT model configuration (configuração do modelo)

**Definição.** Conjunto de hiperparâmetros que define a forma de um modelo GPT, passado como
dicionário ao construtor do modelo: `vocab_size`, `context_length`, `emb_dim`, `n_heads`, `n_layers`,
`drop_rate` e `qkv_bias`.

**Função no modelo.** Uma única classe de modelo produz GPTs de tamanhos muito diferentes; o que
distingue o GPT-2 small do XL é apenas a configuração. Cada chave controla uma parte da arquitetura:
`vocab_size` define o tamanho da tabela de embedding e da cabeça de saída; `context_length`, o número
de posições; `emb_dim`, a largura de todos os tensores internos; `n_heads`, a divisão da atenção;
`n_layers`, o número de blocos empilhados; `drop_rate`, a taxa de dropout; e `qkv_bias`, se as
projeções de query, key e value têm viés.

**Relação com outros conceitos.** Determina diretamente a
[contagem de parâmetros](#parameter-count-contagem-de-parâmetros-e-memória) e o limite do
[corte de contexto](#context-cropping-corte-do-contexto-na-geração). `emb_dim` precisa ser divisível
por `n_heads`, exigência da
[`MultiHeadAttention`](capitulo-03.md#multiheadattention-divisão-de-pesos) da Sprint 3.

**Exemplo.** GPT-2 small: `vocab_size=50257`, `context_length=1024`, `emb_dim=768`, `n_heads=12`,
`n_layers=12`, `drop_rate=0.1`, `qkv_bias=False`. O exercício 4.2 pede as variantes maiores:
medium (`emb_dim=1024`, 24 blocos, 16 cabeças), large (1.280, 36, 20) e XL (1.600, 48, 25).

---

## `GPTModel` (modelo GPT)

**Definição.** O modelo completo do capítulo: embeddings de token e de posição, dropout, uma pilha de
`n_layers` [blocos Transformer](#transformer-block-bloco-transformer), a
[LayerNorm final](#final-layernorm-normalização-final) e a
[cabeça de saída](#language-model-head-cabeça-de-saída-out_head).

**Função no modelo.** Recebe um lote de token IDs `(b, L)` e devolve um lote de
[logits](#logits) `(b, L, V)`: para cada posição de cada sequência, uma pontuação para cada token do
vocabulário. É a peça que a [geração autoregressiva](#autoregressive-generation-geração-autoregressiva)
chama a cada passo e que será treinada na Sprint 5.

**Relação com outros conceitos.** Reúne os componentes de todas as sprints anteriores: os
[embeddings de entrada](capitulo-02.md#input-embeddings-embeddings-de-entrada) do Capítulo 2 e a
atenção multi-cabeça do Capítulo 3. No projeto, a classe `GPTModel` de `src/gpt.py` reaproveita
`EmbeddingDeEntrada` (`src/embeddings.py`) e `MultiHeadAttention` (`src/attention.py`). O livro começa
por uma versão com componentes vazios (`DummyGPTModel`) para mostrar o fluxo antes de implementar cada
peça.

**Exemplo.** O lote `[[6109, 3626, 6100, 345], [6109, 1110, 6622, 257]]` — `"Every effort moves you"` e
`"Every day holds a"` — tem forma `(2, 4)`; o modelo GPT-2 small devolve um tensor `(2, 4, 50257)`.
Pelo caminho: `(2, 4) → (2, 4, 768)` nos embeddings, `(2, 4, 768)` em todos os 12 blocos e
`(2, 4, 50257)` na cabeça de saída.

---

## Greedy decoding (decodificação gulosa)

**Definição.** Estratégia de seleção que escolhe, a cada passo, o token de maior probabilidade —
`argmax` sobre a distribuição do próximo token.

**Função no modelo.** É a estratégia da primeira função de geração do livro (`generate_text_simple`;
no projeto, `gerar_texto_simples`). É determinística: com o modelo em
[modo de avaliação](#eval-e-train-modos-de-execução-do-modelo), a mesma entrada gera sempre o mesmo
texto. O custo dessa previsibilidade é a falta de variedade: o modelo sempre segue o caminho mais
provável, o que tende a produzir textos repetitivos.

**Relação com outros conceitos.** Como o [softmax](capitulo-03.md#softmax) é monotônico, o `argmax` das
[probabilidades](#next-token-probabilities-probabilidades-do-próximo-token) é o mesmo dos
[logits](#logits) — o softmax é dispensável aqui e aparece no livro só por clareza. É o caso-limite da
[temperatura](#temperature-temperatura) tendendo a zero; na função `gerar`, temperatura 0 seleciona
exatamente esse ramo. A alternativa é a [amostragem multinomial](#multinomial-sampling-amostragem-multinomial).

**Exemplo.** No vocabulário de brinquedo da seção 5.3, com o contexto `"every effort moves you"` e os
logits `[4.51, 0.89, −1.90, 6.75, 1.63, −1.62, −1.89, 6.28, 1.79]`, o maior valor está no índice 3
(`"forward"`), e é esse o token escolhido — sempre.

---

## Inference (inferência)

**Definição.** Uso do modelo já construído (e, depois da Sprint 5, treinado) para produzir saídas a
partir de novas entradas, sem ajustar os pesos. No GPT, inferir é calcular a distribuição do próximo
token para uma sequência.

**Função no modelo.** É a etapa em que o modelo é usado de fato. Exige dois cuidados: colocar o modelo
em [modo de avaliação](#eval-e-train-modos-de-execução-do-modelo), para desligar o dropout, e executar
sob `torch.no_grad()`, para não registrar o grafo de gradientes. Só há a passagem direta
(*forward*); não há perda, retropropagação nem otimizador.

**Relação com outros conceitos.** A [geração autoregressiva](#autoregressive-generation-geração-autoregressiva)
é uma sequência de inferências, uma por token gerado. É o oposto do treinamento, em que a mesma
passagem direta é seguida pelo cálculo da perda e pela atualização dos pesos.

**Exemplo.** `model.eval()`, depois `with torch.no_grad(): logits = model(idx)`, com `idx` de forma
`(1, L)`, devolve logits `(1, L, 50257)`; só a última linha, `logits[:, -1, :]` de forma `(1, 50257)`,
é usada para escolher o próximo token.

---

## Language Model Head (cabeça de saída, `out_head`)

**Definição.** Camada linear final do GPT, sem viés, que projeta cada vetor de dimensão `d` no espaço
do vocabulário, de dimensão `V`: `nn.Linear(d, V, bias=False)`.

**Função no modelo.** Transforma a representação contextual de cada posição em uma pontuação para cada
token possível — os [logits](#logits). É a ponte entre o espaço interno do modelo e o vocabulário do
tokenizador: sem ela, o modelo produziria vetores de 768 números, e não uma escolha entre 50.257
tokens.

**Relação com outros conceitos.** Recebe a saída da [LayerNorm final](#final-layernorm-normalização-final).
Sua matriz de pesos tem a mesma forma `(V, d)` da tabela de
[embedding de token](capitulo-02.md#token-embedding-embedding-de-token), o que permite o
[weight tying](#weight-tying-compartilhamento-de-pesos). Não confundir com a
[projeção de saída](capitulo-03.md#output-projection-out_proj) da atenção (`out_proj`, de `d` para `d`).

**Exemplo.** No GPT-2 small, `(b, L, 768) → (b, L, 50257)`, com `50.257 × 768 = 38.597.376`
parâmetros — sozinha, quase um quarto dos 163 milhões do modelo e mais que todas as camadas de
atenção somadas.

---

## LayerNorm (normalização de camada)

**Definição.** Normalização que ajusta o vetor de cada token para média 0 e variância 1 ao longo das
suas `d` características e, em seguida, aplica uma reescala e um deslocamento treináveis:
`y = scale · (x − μ) / √(σ² + ε) + shift`. A variância é calculada dividindo por `d` (estimativa
enviesada, sem a correção de Bessel), como na implementação original do GPT-2.

**Função no modelo.** Estabiliza as ativações ao longo da rede profunda. Sem normalização, a escala dos
vetores pode crescer ou encolher de bloco para bloco, o que torna o treinamento instável e lento. No
GPT ela aparece antes da atenção e antes da feed forward, em cada bloco, e uma vez antes da cabeça de
saída.

**Relação com outros conceitos.** Usa [epsilon](#epsilon-eps-constante-de-estabilidade-numérica) e os
parâmetros [scale e shift](#scale-e-shift-parâmetros-treináveis-da-layernorm). Difere da
[batch normalization](#batch-normalization-normalização-em-lote) por não depender do lote. Sua posição
no bloco é o que define o padrão
[pre-LayerNorm](#pre-layernorm-e-post-layernorm-posição-da-normalização). Junto com as
[conexões residuais](#residual-connection-conexão-residual-shortcut-ou-skip-connection), combate o
[desaparecimento do gradiente](#vanishing-gradient-desaparecimento-do-gradiente).

**Exemplo.** No livro, uma camada `Linear(5, 6)` seguida de ReLU produz duas linhas com médias 0,1324 e
0,2170 e variâncias 0,0231 e 0,0398; depois da normalização, as duas linhas têm média 0 e variância 1.
No GPT, a LayerNorm opera sobre a última dimensão do tensor `(b, L, d)`: cada um dos `b · L` vetores é
normalizado independentemente.

---

## Logits

**Definição.** Pontuações não normalizadas que o modelo atribui a cada token do vocabulário, para cada
posição da sequência. São números reais quaisquer — podem ser negativos e não somam 1.

**Função no modelo.** São a saída do GPT. O vetor de logits da posição `i` expressa o quanto o modelo
"acredita" em cada token como sucessor do token `i`. Na geração, só interessa a última posição, que
prevê o token seguinte ao fim do texto; no treinamento (Sprint 5), todas as posições são usadas, cada
uma comparada ao seu alvo.

**Relação com outros conceitos.** Produzidos pela
[cabeça de saída](#language-model-head-cabeça-de-saída-out_head). O
[softmax](capitulo-03.md#softmax) os converte em
[probabilidades](#next-token-probabilities-probabilidades-do-próximo-token);
[temperatura](#temperature-temperatura) e [top-k](#top-k-sampling-amostragem-top-k) atuam sobre os
logits *antes* do softmax.

**Exemplo.** Para o lote `(2, 4)` de token IDs, os logits têm forma `(2, 4, 50257)`. Na geração,
`logits[:, -1, :]` extrai a última posição, de forma `(2, 50257)`. No vocabulário de brinquedo da
seção 5.3, os logits `[4.51, 0.89, …, 6.75, …, 6.28, 1.79]` viram, pelo softmax, probabilidades de
0,572 para `"forward"` e 0,358 para `"toward"`.

---

## Multinomial sampling (amostragem multinomial)

**Definição.** Seleção aleatória do próximo token em que cada token é sorteado com probabilidade igual
à que o modelo lhe atribui — `torch.multinomial(probs, num_samples=1)`.

**Função no modelo.** Substitui o `argmax` da [decodificação gulosa](#greedy-decoding-decodificação-gulosa)
para dar variedade ao texto: o token mais provável continua sendo escolhido na maioria das vezes, mas
não sempre. É o mecanismo que torna a geração não determinística. Vem da seção 5.3 do livro; a Sprint
4 o explora junto com a geração.

**Relação com outros conceitos.** Atua sobre as
[probabilidades do próximo token](#next-token-probabilities-probabilidades-do-próximo-token), que podem
ser previamente reformadas pela [temperatura](#temperature-temperatura) e restritas pelo
[top-k](#top-k-sampling-amostragem-top-k). A semente aleatória (`torch.manual_seed`) torna o sorteio
reprodutível.

**Exemplo.** No vocabulário de brinquedo da seção 5.3, 1.000 sorteios com semente 123 dão 582 vezes
`"forward"`, 343 `"toward"`, 73 `"closer"` e 2 `"inches"` — frequências próximas das probabilidades
0,572, 0,358, 0,061 e 0,003.

---

## Next-token probabilities (probabilidades do próximo token)

**Definição.** Distribuição de probabilidade sobre o vocabulário obtida aplicando o
[softmax](capitulo-03.md#softmax) aos [logits](#logits) de uma posição: `V` valores positivos que
somam 1.

**Função no modelo.** Expressa a previsão do modelo para o próximo token de forma interpretável: cada
valor é a chance que o modelo atribui àquele token. É sobre essa distribuição que a estratégia de
seleção age — escolhendo o máximo ou sorteando —, e é ela que o treinamento da Sprint 5 vai tentar
concentrar no token correto.

**Relação com outros conceitos.** Sua forma é controlada pela [temperatura](#temperature-temperatura)
e pelo [top-k](#top-k-sampling-amostragem-top-k). A
[decodificação gulosa](#greedy-decoding-decodificação-gulosa) pega o seu máximo; a
[amostragem multinomial](#multinomial-sampling-amostragem-multinomial) sorteia a partir dela.

**Exemplo.** `probs = torch.softmax(logits[:, -1, :], dim=-1)` tem forma `(b, 50257)` e
`probs.sum(dim=-1)` vale 1 em cada linha. Uma distribuição uniforme daria `1/50.257 ≈ 0,00002` a cada
token; um modelo não treinado espalha a probabilidade por milhares de tokens, sem preferência real.

---

## Parameter count (contagem de parâmetros e memória)

**Definição.** Número total de pesos treináveis de um modelo, obtido somando o número de elementos de
todos os seus tensores de parâmetros (`sum(p.numel() for p in model.parameters())`); multiplicado pelo
tamanho de cada número, dá a memória ocupada pelo modelo.

**Função no modelo.** É a medida usual do tamanho de um LLM — o "124M" do GPT-2 small — e determina a
memória necessária para carregá-lo e o custo de treiná-lo. Decompor a contagem por componente mostra
onde está a capacidade do modelo.

**Relação com outros conceitos.** Depende inteiramente da
[configuração](#gpt-model-configuration-configuração-do-modelo). O
[weight tying](#weight-tying-compartilhamento-de-pesos) explica a diferença entre os 163 milhões
contados e os 124 milhões nominais. No projeto, `contar_parametros` (`src/gpt.py`) faz a soma.

**Exemplo.** GPT-2 small com cabeça de saída separada: embedding de token 38.597.376; embedding
posicional 786.432; 12 blocos com 7.085.568 cada (atenção 2.360.064, feed forward 4.722.432, duas
LayerNorms 3.072), 85.026.816 no total; LayerNorm final 1.536; cabeça de saída 38.597.376. Soma:
**163.009.536** parâmetros. Em `float32` (4 bytes cada), `163.009.536 × 4 / 1024² ≈ 621,83 MB`.
Descontada a cabeça de saída, restam 124.412.160.

---

## Pre-LayerNorm e Post-LayerNorm (posição da normalização)

**Definição.** As duas posições possíveis da LayerNorm em um bloco Transformer. Na **pre-LayerNorm**, a
normalização é aplicada à entrada de cada subcamada, dentro do ramo residual:
`x + Subcamada(LN(x))`. Na **post-LayerNorm**, é aplicada depois da soma residual:
`LN(x + Subcamada(x))`.

**Função no modelo.** O GPT-2, e a implementação do livro, usam pre-LayerNorm. Nessa disposição o
caminho residual — a soma `x + …` — atravessa a rede sem passar por nenhuma normalização, o que mantém
um caminho direto para o gradiente; o livro observa que a disposição post-LayerNorm, do Transformer
original, costuma levar a uma dinâmica de treinamento pior.

**Relação com outros conceitos.** É a combinação entre a [LayerNorm](#layernorm-normalização-de-camada)
e as [conexões residuais](#residual-connection-conexão-residual-shortcut-ou-skip-connection). Como o
fluxo residual não é normalizado dentro dos blocos, a pre-LayerNorm exige a
[LayerNorm final](#final-layernorm-normalização-final) antes da cabeça de saída.

**Exemplo.** Primeira metade do bloco do GPT, em pre-LayerNorm:
`atalho = x; x = norm1(x); x = att(x); x = dropout(x); x = x + atalho`. Em post-LayerNorm seria
`x = norm1(x + dropout(att(x)))`.

---

## ReLU (Rectified Linear Unit)

**Definição.** Função de ativação `ReLU(x) = max(0, x)`: devolve a entrada quando ela é positiva e zero
caso contrário.

**Função no modelo.** Não é usada no GPT. Aparece no capítulo no exemplo da LayerNorm e como contraste
com a [GELU](#gelu-gaussian-error-linear-unit). Por zerar todas as entradas negativas, tem gradiente
nulo nessa região — um neurônio que só recebe valores negativos deixa de aprender —, e o "canto" em
zero pode dificultar a otimização em redes muito profundas.

**Relação com outros conceitos.** Foi a ativação padrão do deep learning por ser simples e eficiente;
em LLMs, ativações suaves como GELU e SwiGLU a substituíram. Fica dentro da
[rede feed forward](#feedforward-network-rede-feed-forward) nas arquiteturas que a utilizam.

**Exemplo.** `ReLU([−2, −0,5, 0, 0,5, 2]) = [0, 0, 0, 0,5, 2]`. A GELU, para as mesmas entradas, dá
aproximadamente `[−0,045, −0,154, 0, 0,346, 1,955]`.

---

## Residual connection (conexão residual, shortcut ou skip connection)

**Definição.** Ligação que soma a entrada de uma camada (ou bloco de camadas) à sua saída:
`y = x + F(x)`. Cria um caminho alternativo, que "pula" a camada.

**Função no modelo.** Mantém o fluxo do gradiente em redes profundas. Na retropropagação, a soma passa
o gradiente adiante sem atenuação, de modo que as primeiras camadas continuam recebendo sinal de
aprendizado mesmo com dezenas de camadas acima delas. No GPT há duas por bloco: uma em torno da atenção
e outra em torno da feed forward. Também significa que cada subcamada aprende uma *correção* a ser
somada à representação, em vez de reescrevê-la por inteiro.

**Relação com outros conceitos.** Foi proposta para redes de visão computacional (ResNet) contra o
[desaparecimento do gradiente](#vanishing-gradient-desaparecimento-do-gradiente). Exige que entrada e
saída tenham a mesma forma — razão pela qual atenção, [feed forward](#feedforward-network-rede-feed-forward)
e o [bloco Transformer](#transformer-block-bloco-transformer) preservam `(b, L, d)`. Combinada com a
[pre-LayerNorm](#pre-layernorm-e-post-layernorm-posição-da-normalização).

**Exemplo.** Na rede de 5 camadas do livro, a média do valor absoluto do gradiente na primeira camada
é cerca de 0,0002 sem conexões residuais e 0,22 com elas — mais de mil vezes maior. Sem os atalhos, o
gradiente encolhe de camada em camada, de 0,005 na última até 0,0002 na primeira.

---

## Scale e shift (parâmetros treináveis da LayerNorm)

**Definição.** Os dois vetores treináveis da LayerNorm, ambos de dimensão `d`: `scale` (γ), que
multiplica o vetor normalizado, e `shift` (β), que é somado a ele. São inicializados com uns e zeros,
respectivamente.

**Função no modelo.** Devolvem ao modelo a liberdade que a normalização retira. Forçar média 0 e
variância 1 em todos os vetores pode ser restritivo demais; com `scale` e `shift`, o treinamento pode
escolher, característica por característica, a escala e o deslocamento mais úteis — inclusive
desfazer a normalização, se for o caso.

**Relação com outros conceitos.** São os únicos parâmetros da [LayerNorm](#layernorm-normalização-de-camada);
o [epsilon](#epsilon-eps-constante-de-estabilidade-numérica) é fixo. Por serem inicializados com uns e
zeros, uma LayerNorm recém-criada produz exatamente o vetor normalizado. Entram na
[contagem de parâmetros](#parameter-count-contagem-de-parâmetros-e-memória) com `2 · d` por LayerNorm.

**Exemplo.** Com `d = 768`, cada LayerNorm tem 1.536 parâmetros. O GPT-2 small tem 25 LayerNorms (2 por
bloco nos 12 blocos, mais a final): 38.400 parâmetros, menos de 0,03% do modelo.

---

## Temperature (temperatura)

**Definição.** Número positivo `T` pelo qual os logits são divididos antes do softmax:
`probs = softmax(logits / T)`.

**Função no modelo.** Controla o quão concentrada é a distribuição do próximo token e, com isso, o
equilíbrio entre previsibilidade e variedade do texto. `T < 1` acentua as diferenças entre os logits e
concentra a probabilidade nos tokens mais prováveis; `T > 1` achata a distribuição e dá mais chance aos
tokens improváveis; `T = 1` mantém a distribuição original. Vem da seção 5.3 do livro; a Sprint 4 a
explora como parâmetro da geração.

**Relação com outros conceitos.** Só tem efeito combinada com a
[amostragem multinomial](#multinomial-sampling-amostragem-multinomial): com `argmax`, dividir os logits
por uma constante positiva não muda o vencedor. Quando `T → 0`, a amostragem se aproxima da
[decodificação gulosa](#greedy-decoding-decodificação-gulosa); na função `gerar` (`src/geracao.py`),
temperatura 0 seleciona diretamente o `argmax`. Costuma ser usada junto com o
[top-k](#top-k-sampling-amostragem-top-k).

**Exemplo.** No vocabulário de brinquedo da seção 5.3, a probabilidade de `"forward"` é 0,572 com
`T = 1`, sobe para 0,991 com `T = 0,1` e cai para 0,242 com `T = 5`. Com `T = 5`, a continuação sem
sentido `"pizza"` passa de 0,0001 para 0,043 — cerca de 4% dos sorteios.

---

## Top-k sampling (amostragem top-k)

**Definição.** Estratégia que restringe a escolha do próximo token aos `k` tokens de maiores logits:
os demais logits são substituídos por `−∞` antes do softmax, recebendo probabilidade exatamente zero, e
a probabilidade se redistribui entre os `k` restantes.

**Função no modelo.** Evita que a amostragem escolha tokens muito improváveis — e, em geral, sem
sentido — sem abrir mão da variedade entre os candidatos plausíveis. Vem da seção 5.3 do livro; a
Sprint 4 a explora como parâmetro da geração.

**Relação com outros conceitos.** Usa o mesmo truque do `−∞` da
[máscara causal](capitulo-03.md#causal-mask-máscara-causal) do Capítulo 3: `exp(−∞) = 0`. Costuma ser
combinada com a [temperatura](#temperature-temperatura): o top-k define *quais* tokens podem ser
escolhidos e a temperatura, *com que probabilidade relativa*. Com `k = 1`, equivale à
[decodificação gulosa](#greedy-decoding-decodificação-gulosa).

**Exemplo.** Com `k = 3` no vocabulário de brinquedo da seção 5.3, sobram `"forward"` (6,75),
`"toward"` (6,28) e `"closer"` (4,51); as probabilidades viram 0,578, 0,361 e 0,061, e as outras seis
são zero. No livro, a geração com o modelo treinado usa `top_k=25` e `temperature=1.4`.

---

## Transformer

**Definição.** Arquitetura de rede neural proposta em 2017, no artigo *Attention Is All You Need*,
baseada inteiramente em mecanismos de atenção, sem recorrência. A versão original tinha um codificador
e um decodificador, cada um formado por blocos empilhados de atenção e redes feed forward.

**Função no modelo.** É a arquitetura sobre a qual o GPT é construído — o "T" de *Generative Pretrained
Transformer*. O GPT usa apenas a parte decodificadora, repetindo o mesmo
[bloco Transformer](#transformer-block-bloco-transformer) `n_layers` vezes. A atenção permite que cada
token consulte diretamente qualquer token anterior, o que viabiliza dependências de longo alcance e o
processamento paralelo de todas as posições no treinamento.

**Relação com outros conceitos.** O GPT é uma variante
[somente decodificador](#decoder-only-arquitetura-somente-decodificador); o BERT, uma variante só com
codificador. A peça central é a
[atenção multi-cabeça](capitulo-03.md#multi-head-attention-atenção-multi-cabeça) do Capítulo 3. Ver
também [Transformer](capitulo-01.md#transformer-arquitetura-transformer) no glossário do Capítulo 1.

**Exemplo.** O GPT-2 small é um Transformer somente decodificador com 12 blocos de largura 768; o GPT-2
XL usa a mesma arquitetura com 48 blocos de largura 1.600.

---

## Transformer block (bloco Transformer)

**Definição.** Unidade que se repete ao longo do GPT, formada por duas subcamadas, cada uma precedida
por uma LayerNorm, seguida de dropout e envolvida por uma conexão residual:

```
x = x + Dropout(MultiHeadAttention(LayerNorm₁(x)))
x = x + Dropout(FeedForward(LayerNorm₂(x)))
```

**Função no modelo.** É onde o modelo processa a sequência. A
[atenção multi-cabeça](capitulo-03.md#multi-head-attention-atenção-multi-cabeça), com máscara causal,
troca informação *entre* posições; a [feed forward](#feedforward-network-rede-feed-forward) transforma
cada posição *individualmente*. A forma do tensor não muda — `(b, L, d)` na entrada e na saída —, o
que permite empilhar quantos blocos se quiser; o conteúdo de cada vetor, porém, passa a incorporar
informação do contexto anterior.

**Relação com outros conceitos.** Integra a [LayerNorm](#layernorm-normalização-de-camada) em
disposição [pre-LayerNorm](#pre-layernorm-e-post-layernorm-posição-da-normalização), as
[conexões residuais](#residual-connection-conexão-residual-shortcut-ou-skip-connection) e o
[dropout](#dropout-no-modelo-gpt). Empilhados, formam o corpo do [`GPTModel`](#gptmodel-modelo-gpt). O
Capítulo 3 o apresentou como destino da atenção
([bloco Transformer](capitulo-03.md#transformer-block-bloco-transformer)); aqui ele é implementado.

**Exemplo.** No GPT-2 small, com uma entrada `(2, 4, 768)`: `LayerNorm₁ → (2, 4, 768)`; atenção com
12 cabeças de dimensão 64 — internamente `(2, 12, 4, 64)` e pesos de atenção `(2, 12, 4, 4)` —, saída
`(2, 4, 768)`; soma residual; `LayerNorm₂`; feed forward `(2, 4, 768) → (2, 4, 3072) → (2, 4, 768)`;
soma residual. Saída `(2, 4, 768)`. Cada bloco tem 7.085.568 parâmetros.

---

## Vanishing gradient (desaparecimento do gradiente)

**Definição.** Fenômeno em que os gradientes ficam progressivamente menores à medida que são
retropropagados da saída para as primeiras camadas de uma rede profunda, até se tornarem pequenos
demais para atualizar os pesos de forma útil. O problema simétrico, gradientes que crescem sem
controle, é a explosão do gradiente.

**Função no modelo.** É o principal obstáculo a treinar redes com muitas camadas, como os LLMs. Se as
camadas iniciais não recebem gradiente, elas param de aprender, e a profundidade da rede deixa de
trazer benefício. A arquitetura do bloco Transformer foi desenhada para evitá-lo.

**Relação com outros conceitos.** As duas defesas do GPT são as
[conexões residuais](#residual-connection-conexão-residual-shortcut-ou-skip-connection), que oferecem
um caminho direto para o gradiente, e a [LayerNorm](#layernorm-normalização-de-camada), que mantém as
ativações em escala controlada. A [GELU](#gelu-gaussian-error-linear-unit), por ter gradiente não nulo
para entradas negativas, também ajuda.

**Exemplo.** Na rede de 5 camadas do livro, sem atalhos, a média do valor absoluto do gradiente cai de
0,0050 na última camada para 0,0002 na primeira — 25 vezes menor depois de apenas cinco camadas.

---

## Weight tying (compartilhamento de pesos)

**Definição.** Reutilização da mesma matriz de pesos na tabela de
[embedding de token](capitulo-02.md#token-embedding-embedding-de-token) e na
[cabeça de saída](#language-model-head-cabeça-de-saída-out_head). As duas têm forma `(V, d)`: a
primeira transforma um ID em vetor; a segunda, um vetor em pontuações para cada ID.

**Função no modelo.** Reduz o número de parâmetros e a memória do modelo, já que as duas matrizes estão
entre as maiores do GPT. O GPT-2 original usa esse compartilhamento — daí o nome "124M". O livro, por
padrão, mantém as duas camadas separadas, com o argumento de que isso costuma dar melhor treinamento e
desempenho, e retoma o compartilhamento no Capítulo 6, ao carregar os pesos pré-treinados da OpenAI.

**Relação com outros conceitos.** Explica a diferença na
[contagem de parâmetros](#parameter-count-contagem-de-parâmetros-e-memória) entre o modelo do livro e o
GPT-2 nominal. Só é possível porque a cabeça de saída não tem viés e tem a mesma forma da tabela de
embedding.

**Exemplo.** No GPT-2 small, a cabeça de saída tem `50.257 × 768 = 38.597.376` parâmetros. Com as duas
matrizes separadas, o modelo tem 163.009.536 parâmetros; compartilhando-as, 124.412.160 — uma
redução de 24%.
