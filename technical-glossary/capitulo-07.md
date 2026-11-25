# Capítulo 7 — Fine-tuning to Follow Instructions

Glossário dos termos introduzidos no Capítulo 7 de RASCHKA, S. *Build a Large Language Model (From
Scratch)*: adaptação do GPT para responder a instruções. Entradas em ordem alfabética.

No [Capítulo 6](capitulo-06.md) o modelo pré-treinado virou um classificador: a cabeça de saída foi
trocada e a resposta passou a ser uma de duas classes. Aqui a arquitetura **não muda**. O modelo
continua prevendo o próximo token, com a mesma cabeça de 50.257 saídas e a mesma perda do
[Capítulo 5](capitulo-05.md); o que muda são os **dados**: cada exemplo é um pedido seguido da resposta
certa, escritos num formato fixo. O caminho é:

```
exemplo {instruction, input, output}    1.075 exemplos, 913 no treino
  → prompt formatado                    texto único: pedido + "### Response:" + resposta
  → token IDs                           comprimentos diferentes
  → lote montado por agrupar_lote       padding por lote, alvo deslocado, -100 no padding
  → GPTModel pré-treinado               mesma arquitetura, mesma cabeça
  → entropia cruzada + AdamW            mesmo laço da Sprint 5
  → geração a partir do prompt          para sozinho em <|endoftext|>
  → comparação com a resposta esperada  avaliação das respostas
```

Cada entrada diz a **função no fine-tuning** e a **relação com os demais conceitos**. Os exemplos
numéricos vêm das execuções do projeto (`results-by-sprints/sprint06/`): GPT-2 124M ajustado em CPU,
com os 6 últimos blocos treináveis, sobre a base de instruções do livro.

Notação usada nas fórmulas:

| Símbolo | Significado |
|---|---|
| `b` | tamanho do lote (`batch_size`) |
| `L` | número de tokens da maior sequência **do lote** |
| `V` | tamanho do vocabulário — 50.257 no GPT-2 |
| `P`, `R` | precisão e revocação de palavras entre a resposta gerada e a esperada |

---

## Collate function (função de agrupamento)

**Definição.** Função que recebe uma lista de exemplos do dataset e devolve os tensores de um lote. O
`DataLoader` do PyTorch chama essa função a cada lote; a padrão apenas empilha os exemplos, o que só
funciona quando todos têm o mesmo tamanho.

**Função no fine-tuning.** Os prompts de instrução têm comprimentos muito diferentes, e é a função de
agrupamento que os transforma num tensor retangular. A do projeto (`agrupar_lote`, em
`src/instrucoes.py`) faz quatro coisas com cada exemplo do lote:

1. acrescenta um `<|endoftext|>` ao final;
2. completa com o [token de padding](#padding-token-token-de-preenchimento) até o comprimento do maior
   exemplo **daquele lote**;
3. cria o alvo como a entrada deslocada uma posição para a esquerda;
4. troca, no alvo, o padding por [`-100`](#ignore-index-índice-ignorado), mantendo o primeiro
   `<|endoftext|>`.

**Relação com outros conceitos.** No [Capítulo 6](capitulo-06.md#padding-preenchimento) todas as
mensagens eram completadas até o tamanho da maior do conjunto de treino; aqui o padding é por lote, e
lotes de prompts curtos ficam curtos — menos cálculo desperdiçado. O deslocamento do alvo é o mesmo
da janela deslizante do [Capítulo 2](capitulo-02.md), feito agora exemplo a exemplo.

**Exemplo.** Do teste da task D2 (e do livro), com `P = 50256`:

```
exemplos:  [0, 1, 2, 3, 4]   [5, 6]   [7, 8, 9]

entrada:   [0, 1, 2, 3, 4]        alvo:   [1, 2, 3, 4, P]
           [5, 6, P, P, P]                [6, P, -100, -100, -100]
           [7, 8, 9, P, P]                [8, 9, P, -100, -100]
```

Nos 114 lotes de treino do projeto, o comprimento varia de 54 a 91 tokens.

---

## End-of-text token (token de fim de texto)

**Definição.** O token especial `<|endoftext|>`, de ID 50256 no vocabulário do GPT-2. No pré-treino,
separava um documento do seguinte.

**Função no fine-tuning.** Ensina o modelo a **parar**. Cada exemplo de treino termina com um
`<|endoftext|>` que entra no alvo: depois da última palavra da resposta, o token certo a prever é o de
fim. Na geração, `responder` interrompe o laço quando o modelo produz esse token. Sem isso o modelo
seguiria escrevendo até o limite de tokens, como faz o modelo só pré-treinado.

**Relação com outros conceitos.** É o mesmo token usado como
[padding](#padding-token-token-de-preenchimento); o que distingue os dois papéis é o alvo: o primeiro
`<|endoftext|>` é cobrado na perda, os seguintes recebem o
[índice ignorado](#ignore-index-índice-ignorado). A parada depende da geração autorregressiva do
[Capítulo 4](capitulo-04.md), que ganhou um critério de fim além do número máximo de tokens.

**Exemplo.** No conjunto de teste, com limite de 48 tokens por resposta, o modelo pré-treinado gerou em
média 23,9 palavras por resposta e o ajustado, 8,5 — contra 8,4 das respostas
esperadas (`exp06-avaliacao-das-respostas.csv`).

---

## Ignore index (índice ignorado)

**Definição.** Valor especial do alvo que a entropia cruzada desconsidera: as posições marcadas com ele
não entram na média da perda nem geram gradiente. No PyTorch o padrão é `-100`
(`cross_entropy(..., ignore_index=-100)`).

**Função no fine-tuning.** Impede que o modelo seja treinado para prever padding. Num lote, os exemplos
curtos têm muitas posições de preenchimento; se elas contassem, boa parte da perda viria de acertar
`<|endoftext|>` depois de `<|endoftext|>`, uma tarefa trivial que mascararia o erro nas respostas.

**Relação com outros conceitos.** É aplicado pela
[função de agrupamento](#collate-function-função-de-agrupamento) e consumido pela entropia cruzada do
[Capítulo 5](capitulo-05.md), sem nenhuma mudança em `perda_do_lote`: o `-100` já é o padrão da função
do PyTorch. Não confundir com o padding da **entrada**, que continua sendo o ID 50256 — `-100` não é um
token e não poderia ser consultado na tabela de embeddings.

**Exemplo.** No lote do exemplo anterior há 15 posições de alvo, 5 delas com `-100`. O teste da task D2
confere que `cross_entropy` devolve exatamente a média das perdas nas 10 posições restantes.

---

## Instruction dataset (conjunto de dados de instruções)

**Definição.** Base supervisionada em que cada exemplo tem uma **instrução** (a tarefa, em linguagem
natural), uma **entrada** opcional (o texto sobre o qual a tarefa atua) e a **resposta** esperada.

**Função no fine-tuning.** É o que define o comportamento que o modelo vai aprender: o tipo de pedido,
o tamanho e o tom das respostas. O modelo ajustado responde bem a pedidos parecidos com os da base e
mal aos demais.

**Relação com outros conceitos.** É um [conjunto rotulado](capitulo-06.md#supervised-fine-tuning-fine-tuning-supervisionado)
em que o rótulo é um texto, não uma classe. Cada exemplo vira texto pelo
[formato do prompt](#prompt-format-formato-do-prompt) e é dividido em treino, validação e teste, como
no [Capítulo 6](capitulo-06.md#evaluation-dataset-conjunto-de-avaliação).

**Exemplo.** A base do livro (`sprints/dados/instrucoes/instruction-data.json`) tem 1.100 exemplos de
tarefas curtas em inglês — gramática, sinônimos, conversões, perguntas factuais:

```json
{"instruction": "Identify the correct spelling of the following word.",
 "input": "Ocassion",
 "output": "The correct spelling is 'Occasion.'"}
```

Dos 1.100, 25 repetem o par (instrução, entrada) de outro exemplo. Com a divisão do livro, 2 perguntas
do teste e 4 da validação também estavam no treino; o projeto remove as repetições **antes** de
dividir, ficando com 1.075 exemplos: 913 de treino, 55 de validação e 107 de teste.

---

## Instruction fine-tuning (fine-tuning de instruções)

**Definição.** Continuação do treinamento de um modelo pré-treinado sobre pares (instrução, resposta),
para que ele passe a **atender** a pedidos em vez de apenas continuar o texto.

**Função no fine-tuning.** É a segunda forma de adaptação estudada no livro e a que dá origem aos
assistentes de conversa. O modelo pré-treinado já sabe a língua, mas diante de um pedido a continuação
mais provável é repetir a instrução ou seguir escrevendo sobre outra coisa. Depois do ajuste, o padrão
"pedido → `### Response:` → resposta → fim" passa a ser o mais provável.

**Relação com outros conceitos.** Compare com o
[fine-tuning de classificação](capitulo-06.md#classification-fine-tuning-fine-tuning-para-classificação):

| | Classificação (Cap. 6) | Instruções (Cap. 7) |
|---|---|---|
| Cabeça de saída | trocada por `Linear(768, 2)` | a original, `Linear(768, 50257)` |
| Alvo | uma classe por texto | o próximo token, em toda posição |
| Perda lida em | último token | todas as posições que não são padding |
| Saída | uma classe | texto gerado |
| Avaliação | acurácia | [avaliação das respostas](#response-evaluation-avaliação-das-respostas) |

O laço é o `treinar_modelo` do [Capítulo 5](capitulo-05.md), sem alteração: só mudam os lotes que ele
recebe e o ponto de partida ([pesos pré-treinados](capitulo-06.md#pretrained-weights-pesos-pré-treinados)).

**Exemplo.** `treinar_instrucoes.py`: GPT-2 124M, 2 épocas sobre 913 exemplos, AdamW (`lr = 5e-5`,
`weight_decay = 0,1`), lotes de 8, com os 6 últimos blocos e a norma final treináveis
(42.528.768 de 163.037.184 parâmetros). A perda de validação caiu de 4,10 para 0,78 em
14,4 min de CPU. O livro ajusta o GPT-2 355M inteiro; a redução se deve à máquina usada (CPU,
7,5 GB de RAM).

---

## Padding token (token de preenchimento)

**Definição.** Token acrescentado ao final das sequências curtas para que todas as do lote tenham o
mesmo comprimento. No projeto, o ID 50256 (`<|endoftext|>`).

**Função no fine-tuning.** Torna o lote um tensor retangular `(b, L)`. Não carrega informação: existe
só para preencher.

**Relação com outros conceitos.** Fica à direita, e por causa da máscara causal do
[Capítulo 3](capitulo-03.md) nenhum token real o enxerga — o padding não altera a saída das posições
que importam. No alvo ele é substituído pelo [índice ignorado](#ignore-index-índice-ignorado), exceto o
primeiro, que faz o papel de [fim de texto](#end-of-text-token-token-de-fim-de-texto). O
[Capítulo 6](capitulo-06.md#padding-preenchimento) usa o mesmo token, com comprimento fixo.

**Exemplo.** Ver o lote em [Collate function](#collate-function-função-de-agrupamento): o exemplo
`[5, 6]` recebe três tokens de padding na entrada; no alvo, o primeiro é mantido e os outros dois (mais
a posição final) viram `-100`.

---

## Prompt format (formato do prompt)

**Definição.** O molde de texto que transforma um exemplo `{instruction, input, output}` numa única
sequência. O livro usa o formato **Alpaca**: uma frase de abertura fixa e seções marcadas por
`### Instruction:`, `### Input:` e `### Response:`.

**Função no fine-tuning.** Dá ao modelo marcas estáveis para separar o pedido da resposta. Como o
modelo só sabe continuar texto, é o formato que diz **onde** a resposta começa: tudo o que vem depois
de `### Response:`.

**Relação com outros conceitos.** O formato usado na inferência tem de ser **idêntico** ao do treino;
`montar_prompt` é o prompt de treino cortado logo depois de `### Response:\n`, e o teste da task E2
confere que ele é um prefixo exato, em texto e em tokens, de todos os exemplos de treino. O bloco
`### Input:` é omitido quando a entrada é vazia.

**Exemplo.** `formatar_exemplo` (`src/instrucoes.py`):

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Identify the correct spelling of the following word.

### Input:
Ocassion

### Response:
The correct spelling is 'Occasion.'
```

---

## Response evaluation (avaliação das respostas)

**Definição.** Medição da qualidade das respostas geradas pelo modelo ajustado, comparando-as com as
respostas esperadas de um conjunto que ele não viu no treino.

**Função no fine-tuning.** É o equivalente da acurácia do Capítulo 6, mas sem uma resposta única
certa: "The capital of France is Paris." e "Paris." estão ambas corretas. Por isso não há uma métrica
só. As opções, da mais barata à mais fiel:

- **correspondência exata** — a resposta tem as mesmas palavras da esperada, na mesma ordem
  (ignorando caixa e pontuação). Simples, mas pune qualquer variação de redação;
- **F1 de palavras** — `F1 = 2·P·R / (P + R)`, onde `P` é a fração das palavras geradas que estão na
  resposta esperada e `R`, a fração das esperadas que foram geradas. Mede sobreposição, não correção;
- **leitura humana** — a mais confiável e a que não escala;
- **LLM como juiz** — outro modelo dá uma nota à resposta. É o método do livro (Llama 3 via Ollama).

**Relação com outros conceitos.** Usa o conjunto de teste da
[base de instruções](#instruction-dataset-conjunto-de-dados-de-instruções) e respostas produzidas por
geração *greedy* ([Capítulo 5](capitulo-05.md)), para que o resultado seja reprodutível. A perda de
validação mede outra coisa — a probabilidade dada à resposta esperada, token a token — e pode cair sem
que as respostas geradas fiquem certas.

**Exemplo.** O projeto usa as duas métricas automáticas e uma amostra para leitura
(`exp06_avaliacao_das_respostas.py`); o LLM juiz não foi usado, por depender de um modelo pronto de
terceiros. Nos 107 exemplos de teste:

| Modelo | Correspondência exata | F1 de palavras | Palavras por resposta |
|---|---|---|---|
| GPT-2 124M pré-treinado | 0,0% | 0,203 | 23,9 |
| GPT-2 124M ajustado | 7,5% (8 de 107) | 0,528 | 8,5 |

O ajuste mudou o comportamento — o modelo responde uma vez e para, com o comprimento das respostas
esperadas —, mas não deu o conteúdo: só 8 das 107 respostas coincidem com a esperada. E o F1 engana:
"The chemical formula for ammonium nitrate is NaCl." tem F1 0,88 contra a resposta certa (NH4NO3),
porque 7 das 8 palavras coincidem. Métricas de sobreposição medem semelhança de redação, não correção;
daí a leitura de amostras (`exp06-amostra-para-leitura.txt`) e, no livro, o LLM juiz.
