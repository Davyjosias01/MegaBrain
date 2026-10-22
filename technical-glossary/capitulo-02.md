# Capítulo 2 — Working with Text Data

Glossário dos termos introduzidos no Capítulo 2 de RASCHKA, S. *Build a Large Language Model
(From Scratch)*. Entradas em ordem alfabética.

O capítulo cobre a primeira etapa do pipeline — tudo o que acontece antes da atenção:

```
texto bruto → tokens → token IDs → pares entrada/alvo → lotes → embeddings de token + posicionais → (b, L, d)
```

Os números dos exemplos vêm do corpus do projeto, `sprints/the-verdict.txt` (20.479 caracteres), e
do tokenizador BPE do GPT-2.

Notação usada nas fórmulas:

| Símbolo | Significado |
|---|---|
| `N` | número de tokens do corpus depois da tokenização |
| `V` | tamanho do vocabulário (`vocab_size`) |
| `L` | comprimento do contexto (`max_length`), em tokens por sequência |
| `S` | passo da janela deslizante (`stride`) |
| `b` | tamanho do lote (`batch_size`) |
| `d` | dimensão do embedding (`output_dim`; é o `d_in` do Capítulo 3) |
| `x`, `y` | tensores de entrada e de alvo, ambos de forma `(b, L)` |

---

## Batch (lote)

**Definição.** Conjunto de `b` sequências processadas juntas em uma única passagem pelo modelo. No
pipeline do capítulo, um lote é um par de tensores de inteiros `x` e `y`, ambos de forma `(b, L)`;
`b` é o *batch size* (tamanho do lote).

**Função no modelo.** Permite processar várias sequências em paralelo, aproveitando a vetorização da
CPU ou da GPU, e é a unidade sobre a qual a perda e o gradiente são calculados no treinamento. Lotes
pequenos ocupam pouca memória, mas geram atualizações mais ruidosas; lotes grandes estabilizam o
gradiente ao custo de memória. É um hiperparâmetro.

**Relação com outros conceitos.** Montado pelo [DataLoader](#dataloader-carregador-de-dados) a
partir das amostras do [Dataset](#dataset-conjunto-de-dados). Quando o número de amostras não é
múltiplo de `b`, o último lote fica incompleto — ver
[`drop_last`](#drop_last-descarte-do-lote-incompleto). Depois da
[camada de embedding](#embedding-layer-camada-de-embedding), o lote ganha uma terceira dimensão:
`(b, L) → (b, L, d)`.

**Exemplo.** Com `batch_size=8`, `max_length=4` e `stride=4`, o primeiro lote de *The Verdict* tem
`x.shape == (8, 4)`; a primeira linha de `x` é `[40, 367, 2885, 1464]` e a de `y`,
`[367, 2885, 1464, 1807]`.

---

## Byte Pair Encoding (BPE, codificação por pares de bytes)

**Definição.** Algoritmo de tokenização por subpalavras. Começa com um vocabulário de símbolos
elementares — no GPT-2, os 256 valores possíveis de um byte — e, repetidamente, funde o par de
símbolos vizinhos mais frequente no corpus de treinamento em um símbolo novo, até atingir o tamanho
de vocabulário desejado. O resultado é uma tabela de fusões: palavras frequentes viram um único
token; palavras raras são quebradas em pedaços.

**Função no modelo.** Resolve o problema das palavras fora do vocabulário sem recorrer a `<|unk|>`:
qualquer texto pode ser representado, no pior caso byte a byte. Ao mesmo tempo, mantém o vocabulário
em um tamanho fixo e controlável — 50.257 no GPT-2 (256 bytes + 50.000 fusões + `<|endoftext|>`) —,
que define o número de linhas da camada de embedding e de saídas da camada final do modelo.

**Relação com outros conceitos.** É o [tokenizador](#tokenizer-tokenizador) do GPT-2, do GPT-3 e do
ChatGPT original, usado no projeto por meio do [tiktoken](#tiktoken). Produz
[subpalavras](#subword-subpalavra), em contraste com o tokenizador por expressão regular, que
trabalha com palavras inteiras e depende do [token desconhecido](#unknown-token-token-desconhecido).

**Exemplo.** A palavra inventada `"Akwirw ier"` vira os IDs `[33901, 86, 343, 86, 220, 959]`, que
correspondem aos pedaços `"Ak"`, `"w"`, `"ir"`, `"w"`, `" "` e `"ier"` (exercício 2.1); decodificar a
lista devolve o texto original sem perda. *The Verdict* tem 4.690 tokens pelo tokenizador por regex e
5.145 pelo BPE.

---

## Context length (comprimento do contexto, `max_length`)

**Definição.** Número máximo de tokens que o modelo recebe de uma vez — o comprimento `L` de cada
sequência de entrada. No `GPTDatasetV1` é o parâmetro `max_length`; na camada de embedding
posicional, o `context_length`, que fixa quantas posições existem.

**Função no modelo.** Delimita até onde o modelo enxerga: nenhuma previsão pode depender de tokens a
mais de `L` posições de distância. Também dimensiona o custo — o tamanho de cada amostra, o número de
linhas da camada posicional e, no Capítulo 3, a matriz de atenção `L × L`. Textos maiores precisam
ser truncados ou recortados em janelas.

**Relação com outros conceitos.** Junto com o [stride](#stride-passo), determina quantas amostras a
[janela deslizante](#sliding-window-janela-deslizante) extrai do corpus. Fixa o número de posições do
[embedding posicional](#positional-embedding-embedding-posicional). No Capítulo 3, a
[máscara causal](capitulo-03.md#causal-mask-máscara-causal) é criada com esse mesmo tamanho.

**Exemplo.** O livro usa `max_length=4` para ilustrar e recomenda pelo menos 256 em treinamento; o
GPT-2 usa 1.024. Nos 5.145 tokens BPE de *The Verdict*, com janelas sem sobreposição
(`stride = max_length`), `L = 4` gera 1.286 amostras, `L = 256` gera 20 e `L = 1024` apenas 5.

---

## DataLoader (carregador de dados)

**Definição.** Classe `torch.utils.data.DataLoader` do PyTorch, que percorre um
[Dataset](#dataset-conjunto-de-dados) e agrupa suas amostras em lotes, podendo embaralhar a ordem
(`shuffle`) e usar processos paralelos (`num_workers`).

**Função no modelo.** É a interface entre os dados preparados e o laço de treinamento: a cada
iteração entrega um par `(x, y)` já empilhado em tensores. Separa *o que* é uma amostra
(responsabilidade do Dataset) de *como* as amostras são servidas (tamanho do lote, ordem, descarte).

**Relação com outros conceitos.** Monta os [lotes](#batch-lote) e aplica o
[`drop_last`](#drop_last-descarte-do-lote-incompleto). O laço de treinamento da Sprint 5 vai
consumi-lo diretamente.

**Exemplo.** `criar_dataloader(texto, batch_size=1, max_length=4, stride=1, shuffle=False)`
(`src/dados.py`) devolve, no primeiro lote, `x = [[40, 367, 2885, 1464]]` e
`y = [[367, 2885, 1464, 1807]]`; o segundo lote é a mesma janela deslocada uma posição,
`x = [[367, 2885, 1464, 1807]]`.

---

## Dataset (conjunto de dados)

**Definição.** Subclasse de `torch.utils.data.Dataset` que define como obter uma amostra a partir de
um índice. Precisa implementar `__len__` (quantas amostras existem) e `__getitem__` (a amostra de
índice `i`).

**Função no modelo.** No projeto, o `GPTDatasetV1` tokeniza o corpus inteiro uma única vez, percorre
os token IDs com uma janela deslizante e guarda cada par entrada/alvo como tensores. Isso transforma
um texto contínuo em uma coleção indexável de exemplos de treinamento.

**Relação com outros conceitos.** Implementa a [janela deslizante](#sliding-window-janela-deslizante)
e produz os [pares entrada-alvo](#input-target-pairs-pares-entrada-alvo); é consumido pelo
[DataLoader](#dataloader-carregador-de-dados).

**Exemplo.** `GPTDatasetV1(texto, tokenizador, max_length=256, stride=128)` sobre *The Verdict* tem
`len(dataset) == 39`, e `dataset[0]` devolve dois tensores de forma `(256,)` — a entrada e o alvo.

---

## `drop_last` (descarte do lote incompleto)

**Definição.** Parâmetro do `DataLoader` que, quando `True`, descarta o último lote caso ele tenha
menos de `batch_size` amostras.

**Função no modelo.** Garante que todos os lotes tenham a mesma forma `(b, L)`. Um último lote muito
pequeno produz uma estimativa de gradiente ruidosa e pode desestabilizar o treinamento; descartá-lo é
a escolha usual. O custo é deixar essas amostras de fora naquela época.

**Relação com outros conceitos.** Só tem efeito quando o número de amostras do
[Dataset](#dataset-conjunto-de-dados) não é múltiplo do tamanho do [lote](#batch-lote). Com
`shuffle=True`, as amostras descartadas mudam a cada época, de modo que nenhuma fica permanentemente
de fora.

**Exemplo.** Com `max_length=256` e `stride=128`, *The Verdict* gera 39 amostras; com `batch_size=8`
saem 4 lotes completos e as 7 amostras restantes (18%) são descartadas. Com `max_length=4` e
`stride=4`, as 1.286 amostras formam 160 lotes de 8 e sobram 6.

---

## Embedding (representação vetorial)

**Definição.** Mapeamento de objetos discretos — palavras, tokens, imagens, documentos — para pontos
de um espaço vetorial contínuo de dimensão fixa. O vetor resultante também é chamado de embedding.

**Função no modelo.** Redes neurais só operam sobre números contínuos: multiplicações de matrizes,
somas, derivadas. O embedding é a ponte entre o símbolo discreto e essas operações. Em um LLM, os
embeddings são parâmetros aprendidos: o treinamento os ajusta de modo que tokens usados em contextos
parecidos fiquem próximos no espaço.

**Relação com outros conceitos.** Em LLMs são produzidos por uma
[camada de embedding](#embedding-layer-camada-de-embedding) que faz parte do próprio modelo, ao
contrário de embeddings pré-treinados à parte, como os do [Word2Vec](#word2vec). Também existem
embeddings de frases e documentos, usados em *retrieval-augmented generation*, fora do escopo do
projeto. A atenção do Capítulo 3 transforma o embedding de cada token em um
[vetor de contexto](capitulo-03.md#context-vector-vetor-de-contexto).

**Exemplo.** No exemplo do Capítulo 3, *"journey"* é representada pelo vetor de 3 dimensões
`[0.55, 0.87, 0.66]`; no GPT-2 pequeno, cada token é um vetor de 768 dimensões.

---

## Embedding dimension (dimensão do embedding)

**Definição.** Número de componentes `d` do vetor que representa cada token — o `output_dim` (ou
`embedding_dim`) da camada de embedding.

**Função no modelo.** Define a capacidade de representação de cada token e a "largura" do modelo
inteiro: as camadas seguintes recebem e devolvem vetores de dimensão `d`. Dimensões maiores capturam
relações mais sutis, mas aumentam linearmente os parâmetros da camada de embedding (`V × d`) e o
custo de todas as camadas seguintes.

**Relação com outros conceitos.** É a terceira dimensão do tensor de
[embeddings de entrada](#input-embeddings-embeddings-de-entrada) `(b, L, d)` e o `d_in` dos mecanismos
de atenção do Capítulo 3. O [embedding posicional](#positional-embedding-embedding-posicional) precisa
ter a mesma dimensão, para que os dois possam ser somados.

**Exemplo.** O livro usa `d = 3` para ilustrar e `d = 256` para experimentar; o GPT-2 pequeno usa
768 e o maior GPT-3, 12.288. Com o vocabulário BPE de 50.257 tokens, a matriz de embedding tem
12.865.792 parâmetros para `d = 256` e 38.597.376 para `d = 768`.

---

## Embedding layer (camada de embedding)

**Definição.** A camada `torch.nn.Embedding(num_embeddings, embedding_dim)`: uma matriz de pesos
`V × d` em que a linha `i` é o vetor do token de ID `i`. Aplicar a camada a um tensor de IDs é uma
operação de consulta (*lookup*) — ela devolve as linhas correspondentes.

**Função no modelo.** É a primeira camada treinável do LLM. Os pesos começam aleatórios e são
ajustados por retropropagação como os de qualquer outra camada. A consulta por índice calcula o
mesmo que uma codificação *one-hot* seguida de multiplicação por uma camada linear sem bias, mas sem
materializar vetores one-hot de 50.257 posições.

**Relação com outros conceitos.** Recebe [token IDs](#token-id-identificador-de-token) e produz
[embeddings de token](#token-embedding-embedding-de-token); uma segunda instância, consultada pela
posição, produz os [embeddings posicionais](#positional-embedding-embedding-posicional). Seu número de
linhas é o tamanho do [vocabulário](#vocabulary-vocabulário); o de colunas, a
[dimensão do embedding](#embedding-dimension-dimensão-do-embedding).

**Exemplo.** Com `torch.manual_seed(123)` e `nn.Embedding(6, 3)`, a linha 3 da matriz de pesos é
`[-0.4015, 0.9666, -1.1481]`, e `camada(torch.tensor([3]))` devolve exatamente esse vetor. Aplicada a
`torch.tensor([2, 3, 5, 1])`, devolve uma matriz `4 × 3` com as linhas 2, 3, 5 e 1, nessa ordem.

---

## Encode e decode (codificação e decodificação)

**Definição.** Os dois métodos de todo tokenizador. `encode` recebe texto, divide-o em tokens e
converte cada token em seu ID pelo vocabulário; `decode` faz o caminho inverso, de IDs para tokens e
de tokens para texto.

**Função no modelo.** `encode` prepara a entrada do modelo; `decode` torna a saída legível — o modelo
gera IDs, e só o decode os transforma em texto na geração. A propriedade desejável é a ida e volta
sem perda: `decode(encode(texto)) == texto`.

**Relação com outros conceitos.** Dependem do [vocabulário](#vocabulary-vocabulário) nos dois
sentidos (token → ID e ID → token). No tokenizador por regex a ida e volta perde informação — os
espaços são descartados e palavras desconhecidas viram
[`<|unk|>`](#unknown-token-token-desconhecido) —; no
[BPE](#byte-pair-encoding-bpe-codificação-por-pares-de-bytes) ela é exata.

**Exemplo.** Com o tokenizador por regex e `<|unk|>` (o `SimpleTokenizerV2` do notebook da Sprint 2),
`decode(encode("Hello, do you like tea?"))` devolve `"<|unk|>, do you like tea?"`, porque *Hello* não
aparece em *The Verdict*. Com o BPE do GPT-2, o mesmo texto volta idêntico.

---

## Input embeddings (embeddings de entrada)

**Definição.** Soma dos embeddings de token com os embeddings posicionais: o tensor de forma
`(b, L, d)` que efetivamente entra nas camadas principais do LLM.

**Função no modelo.** É o produto final do Capítulo 2 e a entrada do Capítulo 3. Cada vetor carrega
ao mesmo tempo *qual* token está ali e *em que posição* da sequência ele está.

**Relação com outros conceitos.** Combina [embedding de token](#token-embedding-embedding-de-token) e
[embedding posicional](#positional-embedding-embedding-posicional). É o `x` que os mecanismos de
[self-attention](capitulo-03.md#self-attention-autoatenção) recebem; no projeto, é produzido pela
classe `EmbeddingDeEntrada` (`src/embeddings.py`).

**Exemplo.** Lote de token IDs `(8, 4)` → embeddings de token `(8, 4, 256)`; embeddings posicionais
`(4, 256)`; a soma usa *broadcasting* — a mesma matriz de posições é somada a cada uma das 8
sequências — e resulta em `(8, 4, 256)`.

---

## Input-target pairs (pares entrada-alvo)

**Definição.** Pares `(x, y)` de sequências de token IDs em que o alvo `y` é a entrada `x` deslocada
uma posição para a frente: `y[i] = x[i + 1]`.

**Função no modelo.** Codificam a tarefa de pré-treinamento sem nenhum rótulo manual — o próprio texto
fornece as respostas. Uma única janela de `L` tokens contém `L` problemas de previsão: prever o 2º
token a partir do 1º, o 3º a partir dos dois primeiros, e assim por diante.

**Relação com outros conceitos.** Materializam a
[previsão do próximo token](#next-token-prediction-previsão-do-próximo-token); são produzidos pela
[janela deslizante](#sliding-window-janela-deslizante) do [Dataset](#dataset-conjunto-de-dados). A
[máscara causal](capitulo-03.md#causal-mask-máscara-causal) do Capítulo 3 existe justamente para que a
posição `i` não veja `x[i + 1]`, que é a resposta `y[i]`.

**Exemplo.** A partir do 51º token BPE de *The Verdict*: `x = [290, 4920, 2241, 287]` e
`y = [4920, 2241, 287, 257]`, ou seja, *" and"* → *" established"*, *" and established"* →
*" himself"*, *" and established himself"* → *" in"* e *" and established himself in"* → *" a"*.

---

## Next-token prediction (previsão do próximo token)

**Definição.** Tarefa de treinamento em que o modelo recebe uma sequência de tokens e deve prever o
token seguinte. É uma forma de aprendizado autossupervisionado: os rótulos são extraídos do próprio
texto.

**Função no modelo.** É o objetivo do pré-treinamento dos modelos GPT. Na geração, o mesmo mecanismo
é aplicado repetidamente: o token previsto é anexado à entrada e o modelo prevê o seguinte.

**Relação com outros conceitos.** Define o formato dos
[pares entrada-alvo](#input-target-pairs-pares-entrada-alvo) e, por isso, toda a organização do
[Dataset](#dataset-conjunto-de-dados). Exige a
[atenção causal](capitulo-03.md#causal-attention-atenção-causal--masked-attention) do Capítulo 3; a
função de perda que mede o acerto da previsão é assunto da Sprint 5.

**Exemplo.** No notebook da Sprint 2, com o tokenizador por regex: contexto *"in a villa on"* → alvo
*"the"*; contexto *"in a villa"* → alvo *"on"*.

---

## Positional embedding (embedding posicional)

**Definição.** Vetor que representa a posição de um token na sequência e é somado ao embedding do
token. Há duas famílias: **absolutos**, com um vetor próprio para cada posição (1ª, 2ª, …), e
**relativos**, que codificam a distância entre pares de tokens em vez da posição exata. Os absolutos
podem ser fixos, como as senoides do Transformer original, ou aprendidos, como no GPT.

**Função no modelo.** A camada de embedding de token ignora a posição — o mesmo ID gera sempre o
mesmo vetor — e a self-attention também não tem noção de ordem. Sem informação posicional,
*"o gato mordeu o cão"* e *"o cão mordeu o gato"* seriam o mesmo conjunto de vetores. O embedding
posicional é a fonte da noção de ordem do modelo.

**Relação com outros conceitos.** Somado ao [embedding de token](#token-embedding-embedding-de-token),
forma os [embeddings de entrada](#input-embeddings-embeddings-de-entrada); por isso precisa ter a
mesma [dimensão](#embedding-dimension-dimensão-do-embedding). Seu número de posições é o
[comprimento do contexto](#context-length-comprimento-do-contexto-max_length). Os relativos tendem a
generalizar melhor para sequências de comprimentos não vistos no treinamento; o GPT usa absolutos
aprendidos.

**Exemplo.** No GPT, é outra `nn.Embedding(context_length, d)`, consultada com `torch.arange(L)` —
os índices `0, 1, …, L − 1`; para `L = 4` e `d = 256`, produz um tensor `(4, 256)`. No experimento 08
da Sprint 3, uma sequência de vinte cópias de *"the"* dá entropia de atenção 0,991 (quase uniforme)
sem embedding posicional e 0,932 com ele.

---

## Raw text (texto bruto)

**Definição.** O texto como sequência de caracteres, antes de qualquer processamento — por exemplo,
os 20.479 caracteres do conto *The Verdict*, lidos de `sprints/the-verdict.txt`.

**Função no modelo.** É a matéria-prima do pré-treinamento, mas o modelo não pode operá-lo
diretamente: texto é dado categórico, e redes neurais trabalham com multiplicações e derivadas sobre
números contínuos. Todo o Capítulo 2 é o caminho do texto bruto até um tensor numérico.

**Relação com outros conceitos.** É a entrada da [tokenização](#tokenization-tokenização). Decisões
tomadas aqui afetam o resto do pipeline: o livro não converte o texto para minúsculas, porque a
capitalização ajuda o modelo a distinguir nomes próprios e inícios de frase.

**Exemplo.** `texto = Path("sprints/the-verdict.txt").read_text(encoding="utf-8")`, com
`len(texto) == 20479`.

---

## Sliding window (janela deslizante)

**Definição.** Estratégia de amostragem que percorre a sequência de token IDs com uma janela de
tamanho `L`, avançando `S` posições por vez e recortando um par entrada-alvo em cada posição.

**Função no modelo.** Transforma um único texto longo em muitas amostras de tamanho fixo, que é o
formato exigido para montar lotes. O número de amostras é `⌈(N − L) / S⌉`.

**Relação com outros conceitos.** Controlada pelo
[comprimento do contexto](#context-length-comprimento-do-contexto-max_length) e pelo
[stride](#stride-passo); implementada no `__init__` do [Dataset](#dataset-conjunto-de-dados).

**Exemplo.** Nos 5.145 tokens BPE de *The Verdict*: `L = 4, S = 1` gera 5.141 janelas;
`L = 4, S = 4`, 1.286; `L = 256, S = 128`, 39. A função `numero_de_amostras(N, L, S)`
(`src/dados.py`) calcula esse valor sem montar o dataset.

---

## Special tokens (tokens especiais)

**Definição.** Tokens que não vêm do texto, adicionados ao vocabulário para sinalizar algo ao
modelo. Os do capítulo são `<|endoftext|>`, que separa documentos independentes concatenados, e
`<|unk|>`, que substitui palavras fora do vocabulário. Outros esquemas usam `[BOS]` (início de
sequência), `[EOS]` (fim de sequência) e `[PAD]` (preenchimento, para igualar o comprimento das
sequências de um lote).

**Função no modelo.** Dão ao modelo informação estrutural que o texto não carrega explicitamente —
por exemplo, que dois trechos vizinhos no corpus de treinamento não têm relação entre si.

**Relação com outros conceitos.** O GPT usa apenas `<|endoftext|>`, que também faz o papel de
`[EOS]` e de `[PAD]` (as posições de preenchimento são mascaradas, então o token que as ocupa não
importa). Não usa `<|unk|>` porque o
[BPE](#byte-pair-encoding-bpe-codificação-por-pares-de-bytes) representa qualquer palavra — ver
[token desconhecido](#unknown-token-token-desconhecido).

**Exemplo.** No vocabulário por regex de *The Verdict*, os dois tokens especiais entram no fim:
`<|endoftext|>` recebe o ID 1130 e `<|unk|>` o 1131, totalizando 1.132 entradas. No BPE do GPT-2,
`<|endoftext|>` é o ID 50256, o último do vocabulário; o tiktoken só o reconhece no meio do texto
quando autorizado com `allowed_special={"<|endoftext|>"}`.

---

## Stride (passo)

**Definição.** Número de posições que a janela deslizante avança entre uma amostra e a seguinte.

**Função no modelo.** Controla a sobreposição entre amostras. `S = 1` gera o máximo de amostras, cada
uma quase idêntica à anterior; `S = L` usa cada token exatamente uma vez como entrada, sem
sobreposição. O livro observa que sobreposição demais pode favorecer *overfitting*, já que o modelo vê
os mesmos trechos repetidas vezes.

**Relação com outros conceitos.** Junto com o
[comprimento do contexto](#context-length-comprimento-do-contexto-max_length), define o número de
amostras da [janela deslizante](#sliding-window-janela-deslizante). Com `S > L`, parte dos tokens é
pulada e nunca aparece no treinamento.

**Exemplo.** Com `L = 4`, a primeira amostra de *The Verdict* é `[40, 367, 2885, 1464]`. Com `S = 1`,
a segunda é `[367, 2885, 1464, 1807]` — deslocada uma posição; com `S = 4`, é
`[1807, 3619, 402, 271]`, sem nenhum token em comum com a primeira.

---

## Subword (subpalavra)

**Definição.** Unidade de tokenização intermediária entre caractere e palavra: um pedaço de palavra,
como `"token"` + `"ization"`.

**Função no modelo.** Equilibra o tamanho do vocabulário e o comprimento das sequências. Palavras
frequentes ficam inteiras, em um único token; palavras raras, inventadas ou de outros idiomas são
compostas de pedaços conhecidos, sem nunca cair em `<|unk|>`. Pedaços recorrentes, como `"ization"`
ou `"ing"`, também permitem que o que é aprendido sobre uma palavra se aproveite em outras da mesma
família.

**Relação com outros conceitos.** Produzidas pelo
[BPE](#byte-pair-encoding-bpe-codificação-por-pares-de-bytes). Um [token](#token) BPE pode ser uma
palavra inteira, uma subpalavra, um caractere ou um byte.

**Exemplo.** No BPE do GPT-2, `"tokenization"` vira `["token", "ization"]`, enquanto `"tokenização"`
vira `["token", "iza", "ç", "ão"]` — o vocabulário foi construído sobre textos em inglês, então
palavras em português custam mais tokens.

---

## tiktoken

**Definição.** Biblioteca de código aberto da OpenAI que implementa tokenizadores BPE, com núcleo em
Rust. `tiktoken.get_encoding("gpt2")` carrega o tokenizador usado no GPT-2.

**Função no modelo.** Fornece ao projeto um tokenizador BPE pronto e eficiente — implementar o
treinamento do BPE está fora do escopo do livro. Garante que os token IDs sejam os mesmos do GPT-2,
o que permitirá carregar os pesos pré-treinados da OpenAI nos capítulos finais.

**Relação com outros conceitos.** Implementa o
[BPE](#byte-pair-encoding-bpe-codificação-por-pares-de-bytes) e oferece os métodos
[`encode` e `decode`](#encode-e-decode-codificação-e-decodificação), com a mesma interface do
tokenizador por regex. No projeto é encapsulado pela classe `TokenizadorBPE` (`src/tokenizacao.py`).

**Exemplo.** `tiktoken.get_encoding("gpt2").n_vocab == 50257`, e
`encode("Hello, do you like tea?")` devolve `[15496, 11, 466, 345, 588, 8887, 30]`.

---

## Token

**Definição.** Unidade mínima em que o texto é dividido para ser processado pelo modelo. Dependendo
do tokenizador, pode ser uma palavra, um sinal de pontuação, uma subpalavra, um caractere ou um byte.

**Função no modelo.** É a unidade de tudo o que o LLM faz: ele lê uma sequência de tokens, representa
cada um como vetor e gera um token por vez. O comprimento do contexto e o custo da atenção são
medidos em tokens, não em palavras.

**Relação com outros conceitos.** Produzido pela [tokenização](#tokenization-tokenização) e associado
a um [token ID](#token-id-identificador-de-token) pelo [vocabulário](#vocabulary-vocabulário).

**Exemplo.** O tokenizador por regex divide `"Hello, world. Is this-- a test?"` em 10 tokens: `Hello`,
`,`, `world`, `.`, `Is`, `this`, `--`, `a`, `test` e `?`.

---

## Token embedding (embedding de token)

**Definição.** Vetor de dimensão `d` associado a um token ID, obtido pela consulta à camada de
embedding de token.

**Função no modelo.** Representa *o que* o token é — seu conteúdo, independentemente de onde aparece.
É a parte do vetor de entrada que o treinamento ajusta para que tokens de uso semelhante fiquem
próximos no espaço.

**Relação com outros conceitos.** Produzido pela
[camada de embedding](#embedding-layer-camada-de-embedding) e somado ao
[embedding posicional](#positional-embedding-embedding-posicional) para formar os
[embeddings de entrada](#input-embeddings-embeddings-de-entrada). Sozinho, é igual para o mesmo ID em
qualquer posição.

**Exemplo.** `nn.Embedding(50257, 256)` aplicada a um lote `(8, 4)` de IDs produz um tensor
`(8, 4, 256)`. Em `"I saw the cat and the dog"`, as duas ocorrências de `" the"` têm o mesmo ID BPE
(262) e, portanto, exatamente o mesmo embedding de token.

---

## Token ID (identificador de token)

**Definição.** Inteiro único que o vocabulário associa a cada token — o índice da linha
correspondente na matriz de embedding.

**Função no modelo.** É a forma do token que o computador manipula: compacta, sem ambiguidade e
utilizável como índice. O número em si, porém, não carrega significado: o ID 1130 não é "maior" nem
"mais parecido" com o 1129 em nenhum sentido linguístico. Por isso o ID não serve como representação
semântica — serve apenas para buscar o embedding.

**Relação com outros conceitos.** Liga [token](#token) e [embedding](#embedding-representação-vetorial):
token ↔ ID pelo [vocabulário](#vocabulary-vocabulário), ID → vetor pela
[camada de embedding](#embedding-layer-camada-de-embedding).

**Exemplo.** No vocabulário por regex de *The Verdict*, `!` tem ID 0 e `"` tem ID 1, porque o
vocabulário é ordenado e os sinais de pontuação vêm antes das letras; no BPE do GPT-2, `"Hello"` é
15496 e `","` é 11.

---

## Tokenization (tokenização)

**Definição.** Processo de dividir o texto bruto em tokens.

**Função no modelo.** Primeiro passo do pipeline. As escolhas feitas aqui — separar ou não a
pontuação, manter ou descartar os espaços, trabalhar com palavras ou subpalavras — definem o tamanho
do vocabulário, o comprimento das sequências e o que o modelo consegue representar.

**Relação com outros conceitos.** Executada por um [tokenizador](#tokenizer-tokenizador); produz
[tokens](#token), que o [vocabulário](#vocabulary-vocabulário) converte em IDs. O capítulo apresenta
duas abordagens: por expressão regular (palavras e pontuação) e
[BPE](#byte-pair-encoding-bpe-codificação-por-pares-de-bytes) (subpalavras).

**Exemplo.** `re.split(r'([,.:;?_!"()\']|--|\s)', texto)`, seguido da remoção dos itens vazios — a
regra usada por `dividir_em_tokens` (`src/tokenizacao.py`) —, divide *The Verdict* em 4.690 tokens.
Descartar os espaços economiza memória, mas impediria o modelo de reproduzir, por exemplo, a
indentação de um código Python.

---

## Tokenizer (tokenizador)

**Definição.** Componente que implementa a tokenização e a conversão entre tokens e IDs, expondo os
métodos `encode` e `decode`.

**Função no modelo.** É a fronteira entre o mundo do texto e o dos tensores, usada na entrada
(texto → IDs) e na saída (IDs gerados → texto). Um modelo só é compatível com o tokenizador com que
foi treinado: os IDs não fazem sentido em outro vocabulário.

**Relação com outros conceitos.** Usa um [vocabulário](#vocabulary-vocabulário) e os métodos
[encode e decode](#encode-e-decode-codificação-e-decodificação). O projeto tem dois:
`TokenizadorSimples` (regex, vocabulário construído a partir do corpus) e `TokenizadorBPE` (BPE do
GPT-2, via [tiktoken](#tiktoken)), ambos em `src/tokenizacao.py`.

**Exemplo.** Um tokenizador por regex cujo vocabulário não tem `<|unk|>` (o `SimpleTokenizerV1` do
livro) falha com `KeyError: 'Hello'` ao codificar *"Hello, do you like tea?"*; com `<|unk|>` no
vocabulário (`SimpleTokenizerV2`), a mesma frase vira `[1131, 5, 355, 1126, 628, 975, 10]`.

---

## Unknown token (token desconhecido)

**Definição.** O token especial `<|unk|>`, que substitui qualquer palavra ausente do vocabulário
(*out-of-vocabulary*).

**Função no modelo.** Evita que o tokenizador falhe diante de palavras novas, mas a um custo: todas as
palavras desconhecidas viram o mesmo token, e a informação sobre qual palavra era se perde — o decode
não tem como recuperá-la.

**Relação com outros conceitos.** É um dos [tokens especiais](#special-tokens-tokens-especiais).
Necessário em tokenizadores de palavra inteira, como o por regex; dispensável no
[BPE](#byte-pair-encoding-bpe-codificação-por-pares-de-bytes), que decompõe palavras desconhecidas em
[subpalavras](#subword-subpalavra).

**Exemplo.** *"Hello"* e *"palace"* não aparecem em *The Verdict*. Codificar
*"In the sunlit terraces of the palace."* com o tokenizador por regex e decodificar de volta devolve
*"In the sunlit terraces of the <|unk|>."*.

---

## Vocabulary (vocabulário)

**Definição.** Mapeamento entre o conjunto de tokens possíveis e os inteiros que os identificam. No
tokenizador por regex, é construído a partir do corpus: a lista dos tokens únicos, em ordem
alfabética, com cada token associado à sua posição. O mapeamento inverso (ID → token) é necessário
para o decode.

**Função no modelo.** Fixa o universo de símbolos que o modelo conhece. Seu tamanho `V` determina o
número de linhas da camada de embedding e, na Sprint 4, o número de saídas da camada final, que
atribui uma pontuação a cada token possível.

**Relação com outros conceitos.** Liga [token](#token) e [token ID](#token-id-identificador-de-token)
nos dois sentidos e é estendido com [tokens especiais](#special-tokens-tokens-especiais). Construído a
partir de um corpus pequeno, deixa muitas palavras de fora — o problema que o
[BPE](#byte-pair-encoding-bpe-codificação-por-pares-de-bytes) resolve.

**Exemplo.** *The Verdict* tem 4.690 tokens, mas apenas 1.130 distintos; com `<|endoftext|>` e
`<|unk|>`, o vocabulário fica com 1.132 entradas. O do BPE do GPT-2 tem 50.257, construído sobre um
corpus muito maior. No projeto, `construir_vocabulario` (`src/tokenizacao.py`) faz essa construção.

---

## Word2Vec

**Definição.** Método de 2013 para gerar embeddings de palavras treinando uma rede neural rasa a
prever as palavras vizinhas de uma palavra (ou a palavra a partir das vizinhas). Parte da hipótese de
que palavras que aparecem em contextos parecidos têm significados parecidos.

**Função no modelo.** Não é usado no projeto. É a referência histórica de que embeddings capturam
semelhança: em um espaço Word2Vec, nomes de aves ficam próximos entre si e longe de nomes de países.
LLMs preferem aprender seus próprios embeddings junto com o resto do modelo, porque assim eles se
ajustam à tarefa e aos dados do treinamento.

**Relação com outros conceitos.** Produz [embeddings](#embedding-representação-vetorial)
pré-treinados e fixos, um por palavra; a [camada de embedding](#embedding-layer-camada-de-embedding)
do GPT é treinada junto com o modelo. Em ambos os casos o vetor de um token não depende do contexto em
que ele aparece — quem contextualiza é a [self-attention](capitulo-03.md#self-attention-autoatenção)
do Capítulo 3.

**Exemplo.** A propriedade mais citada do Word2Vec é a aritmética de vetores:
`vetor("rei") − vetor("homem") + vetor("mulher")` fica próximo de `vetor("rainha")`.
