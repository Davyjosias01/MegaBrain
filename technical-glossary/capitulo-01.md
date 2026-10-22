# Capítulo 1 — Understanding Large Language Models

Glossário dos termos introduzidos no Capítulo 1 de RASCHKA, S. *Build a Large Language Model
(From Scratch)*. Entradas em ordem alfabética.

O Capítulo 1 é conceitual: apresenta o vocabulário que o restante do projeto vai transformar em
código. Por isso, sempre que possível, cada entrada indica em que sprint o conceito passa a ser
implementado.

| Sprint | Capítulo | Onde o conceito vira código |
|---|---|---|
| Sprint 2 | 2 | tokens, token IDs, embeddings e pares entrada/alvo |
| Sprint 3 | 3 | self-attention, causal attention e multi-head attention |
| Sprint 4 | 4 | bloco Transformer e modelo GPT completo |
| Sprint 5 | 5 | pré-treinamento por previsão do próximo token |
| Sprint 6 | 6 e 7 | fine-tuning para classificação e por instruções |

---

## Artificial intelligence (inteligência artificial)

**Definição.** Campo amplo da computação que busca construir sistemas capazes de executar tarefas que
normalmente exigiriam inteligência humana — entender linguagem, reconhecer padrões, tomar decisões.
Inclui o aprendizado de máquina, mas não se resume a ele: sistemas baseados em regras, sistemas
especialistas, algoritmos genéticos, lógica fuzzy e raciocínio simbólico também são IA.

**Função no modelo.** É o guarda-chuva sob o qual o LLM se encaixa. Um LLM é uma aplicação de
[deep learning](#deep-learning-aprendizado-profundo), que é um ramo do
[machine learning](#machine-learning-aprendizado-de-máquina), que por sua vez é um ramo da IA.
Por gerar texto, o LLM também é classificado como **IA generativa** (*generative AI*).

**Relação com outros conceitos.** Contém [machine learning](#machine-learning-aprendizado-de-máquina)
⊃ [deep learning](#deep-learning-aprendizado-profundo) ⊃ [LLMs](#large-language-model-llm-modelo-de-linguagem-de-grande-escala).

**Exemplo.** Um controlador de temperatura baseado em lógica fuzzy é IA sem aprendizado de máquina:
as regras são escritas por um especialista. Um filtro de spam treinado com e-mails rotulados é IA
*com* aprendizado de máquina. O GPT é IA com deep learning.

---

## Autoregressive model (modelo autorregressivo)

**Definição.** Modelo que gera uma sequência um elemento por vez, usando as próprias saídas
anteriores como entrada para prever o próximo elemento.

**Função no modelo.** É o modo de funcionamento do GPT na geração de texto. Cada token gerado é
anexado à entrada, e o modelo é executado de novo para produzir o seguinte. Isso mantém a coerência
do texto, porque cada nova palavra é escolhida condicionada a tudo o que veio antes.

**Relação com outros conceitos.** É a contrapartida, na geração, da
[previsão da próxima palavra](#next-word-prediction-previsão-da-próxima-palavra) usada no treino.
Exige que o modelo só enxergue o passado — o que, no Capítulo 3, é garantido pela máscara causal
(ver [capitulo-03.md](capitulo-03.md#causal-attention-atenção-causal--masked-attention)).

**Exemplo.** Matematicamente, o modelo fatora a probabilidade de uma sequência como
`P(x₁, …, x_T) = Π_t P(x_t | x₁, …, x_{t−1})`. Na prática:
`"O gato subiu no"` → `"telhado"` → `"O gato subiu no telhado"` → `"."` → …

---

## BERT (Bidirectional Encoder Representations from Transformers)

**Definição.** Família de modelos construída sobre o [encoder](#encoder-codificador) do Transformer
original e pré-treinada com **previsão de palavras mascaradas**: palavras escondidas no meio da frase
devem ser recuperadas a partir do contexto dos dois lados.

**Função no modelo.** Não é o modelo deste projeto. Aparece como contraste com o
[GPT](#gpt-generative-pretrained-transformer): por enxergar a frase inteira, nos dois sentidos, o BERT
é forte em tarefas de compreensão e classificação (sentimento, categorização de documentos,
detecção de conteúdo tóxico), mas não foi projetado para gerar texto.

**Relação com outros conceitos.** BERT = encoder; GPT = [decoder](#decoder-decodificador). Ambos
derivam do [Transformer](#transformer-arquitetura-transformer) de 2017.

**Exemplo.** Entrada: `"O [MASK] latiu a noite inteira."` — o BERT deve prever `"cachorro"`, usando
tanto `"O"` quanto `"latiu a noite inteira"`. Um GPT, para a mesma frase, só poderia usar o que vem
antes da lacuna.

---

## Classification fine-tuning (fine-tuning para classificação)

**Definição.** Modalidade de [fine-tuning](#fine-tuning-ajuste-fino) em que o conjunto rotulado é
formado por textos e seus rótulos de classe. O modelo aprende a mapear um texto para uma categoria
de um conjunto fixo.

**Função no modelo.** Adapta um [modelo base](#foundation-model-modelo-de-fundação-modelo-base) a
uma tarefa de decisão — por exemplo, filtrar mensagens. Será implementado no Capítulo 6 (Sprint 6).

**Relação com outros conceitos.** É a alternativa ao
[fine-tuning por instruções](#instruction-fine-tuning-fine-tuning-por-instruções). Ambos partem do
mesmo modelo pré-treinado e usam [dados rotulados](#labeled-and-unlabeled-data-dados-rotulados-e-não-rotulados).

**Exemplo.** Conjunto de treino com pares `(texto, rótulo)`:
`("Você ganhou um prêmio! Clique aqui", spam)`, `("Reunião amanhã às 10h", não-spam)`.

---

## Decoder (decodificador)

**Definição.** Submódulo do Transformer responsável por **gerar** a sequência de saída, um token por
vez. No Transformer original, recebe os vetores produzidos pelo [encoder](#encoder-codificador) e a
parte da saída já gerada.

**Função no modelo.** É a única parte do Transformer que o GPT usa. Sem o encoder, o decoder do GPT
processa o texto da esquerda para a direita e prevê o próximo token a partir dos anteriores — o que o
torna adequado à geração de texto.

**Relação com outros conceitos.** Base do [GPT](#gpt-generative-pretrained-transformer); seu
funcionamento é [autorregressivo](#autoregressive-model-modelo-autorregressivo). O bloco que se
repete dentro dele é o bloco Transformer da Sprint 4.

**Exemplo.** Na tradução da figura 1.4, o decoder já produziu `"Das ist ein"` e precisa gerar apenas
a próxima palavra, `"Beispiel"`, olhando para a frase de entrada codificada e para o que já escreveu.

---

## Deep learning (aprendizado profundo)

**Definição.** Ramo do [machine learning](#machine-learning-aprendizado-de-máquina) que usa
[redes neurais](#neural-network-rede-neural) com três ou mais camadas (redes profundas) para modelar
padrões e abstrações complexas nos dados.

**Função no modelo.** É a técnica que torna os LLMs possíveis. A diferença prática em relação ao
machine learning tradicional é que **não há extração manual de características**: a própria rede
aprende quais aspectos do texto são relevantes. Isso é essencial em linguagem, onde as
características úteis são numerosas demais para serem codificadas à mão.

**Relação com outros conceitos.** Subconjunto do machine learning, que é subconjunto da
[IA](#artificial-intelligence-inteligência-artificial). Os rótulos continuam necessários em tarefas
supervisionadas — o que muda é quem escolhe as características.

**Exemplo.** Em um filtro de spam tradicional, um especialista define atributos como "frequência da
palavra *grátis*", "número de pontos de exclamação" e "presença de links". Em deep learning, a rede
recebe o texto e aprende sozinha quais padrões indicam spam.

---

## Embedding (vetor de representação)

**Definição.** Representação de um token (ou de um texto) como um vetor de números reais em um
espaço de várias dimensões, construída de forma que a geometria do espaço carregue informação —
tokens com papéis parecidos ficam próximos.

**Função no modelo.** Redes neurais operam sobre números contínuos, não sobre texto. O embedding é a
porta de entrada do modelo: converte cada [token](#token-unidade-de-texto) em um vetor que as camadas
seguintes podem processar. No Capítulo 1, aparece como a saída do [encoder](#encoder-codificador);
a construção concreta é assunto do Capítulo 2.

**Relação com outros conceitos.** Vem depois da tokenização; é a entrada da
[self-attention](#self-attention-autoatenção). Implementado na Sprint 2 (embeddings de token e
posicionais, usados por `src/entradas_atencao.py`).

**Exemplo.** `"gato"` → token ID `n` → vetor `[0,12; −0,53; …; 0,08]` com 768 dimensões no GPT-2
pequeno. Em PyTorch: `torch.nn.Embedding(vocab_size, 768)(ids)`.

---

## Emergent behavior (comportamento emergente)

**Definição.** Capacidade que um modelo apresenta sem ter sido treinado explicitamente para ela. Surge
como consequência da escala do modelo e da diversidade dos dados vistos no pré-treinamento.

**Função no modelo.** Explica por que um único LLM, treinado só para prever a próxima palavra,
consegue executar muitas tarefas diferentes — traduzir, resumir, classificar — sem um modelo
específico para cada uma.

**Relação com outros conceitos.** Resultado do [pré-treinamento](#pretraining-pré-treinamento) em um
[corpus](#training-corpus-corpus-de-treinamento) grande e multilíngue; é o que viabiliza o uso
[zero-shot](#zero-shot-learning-aprendizado-sem-exemplos) e [few-shot](#few-shot-learning-aprendizado-com-poucos-exemplos).

**Exemplo.** O GPT não foi treinado para traduzir, ao contrário do Transformer original. Mesmo
assim, por ter visto muito texto em vários idiomas, aprendeu a traduzir — algo que, segundo o
livro, surpreendeu os próprios pesquisadores.

---

## Encoder (codificador)

**Definição.** Submódulo do Transformer original que lê o texto de entrada inteiro e o converte em
uma sequência de vetores que capturam o contexto de cada posição.

**Função no modelo.** Não é usado no GPT. É estudado por contraste: no Transformer original ele
alimenta o [decoder](#decoder-decodificador); sozinho, é a base do [BERT](#bert-bidirectional-encoder-representations-from-transformers).

**Relação com outros conceitos.** Produz [embeddings](#embedding-vetor-de-representação)
contextualizados. Enxerga a frase nos dois sentidos, ao contrário do decoder do GPT.

**Exemplo.** Em uma tradução inglês → alemão, o encoder transforma `"This is an example"` em quatro
vetores contextualizados, que o decoder consulta a cada palavra alemã gerada.

---

## Few-shot learning (aprendizado com poucos exemplos)

**Definição.** Capacidade de executar uma tarefa nova a partir de um pequeno número de exemplos
fornecidos **na própria entrada** do modelo, sem alterar seus pesos.

**Função no modelo.** Permite usar um LLM pré-treinado em tarefas para as quais ele não foi
treinado, apenas mostrando o padrão desejado no prompt. Não há retreinamento: os exemplos orientam a
previsão do próximo token. Por isso também é chamado de aprendizado *em contexto* (*in-context
learning*).

**Relação com outros conceitos.** Contrasta com [zero-shot](#zero-shot-learning-aprendizado-sem-exemplos)
(nenhum exemplo) e com [fine-tuning](#fine-tuning-ajuste-fino) (que altera os pesos). É uma
manifestação de [comportamento emergente](#emergent-behavior-comportamento-emergente). O livro não
fixa uma quantidade de exemplos — "poucos" significa uma quantidade mínima, muito menor que um
conjunto de treino.

**Exemplo.** Prompt: `"gato → cat; cachorro → dog; pássaro →"`. Os dois pares iniciais mostram a
tarefa; o modelo completa com `"bird"`.

---

## Fine-tuning (ajuste fino)

**Definição.** Segunda etapa de treinamento de um LLM: o modelo já pré-treinado continua sendo
treinado, agora com um conjunto de dados **rotulado**, menor e específico de uma tarefa ou domínio.

**Função no modelo.** Especializa o [modelo base](#foundation-model-modelo-de-fundação-modelo-base).
Como o modelo já sabe "linguagem", bastam relativamente poucos dados e pouca computação para
adaptá-lo. Modelos ajustados a um domínio podem superar modelos de uso geral nesse domínio.

**Relação com outros conceitos.** Vem depois do [pré-treinamento](#pretraining-pré-treinamento).
Suas duas modalidades principais são o [fine-tuning para classificação](#classification-fine-tuning-fine-tuning-para-classificação)
e o [fine-tuning por instruções](#instruction-fine-tuning-fine-tuning-por-instruções). Assunto da
Sprint 6.

**Exemplo.** O ChatGPT original foi obtido a partir do GPT-3 por fine-tuning em um grande conjunto
de instruções; o BloombergGPT é um exemplo de modelo especializado em finanças.

---

## Foundation model (modelo de fundação, modelo base)

**Definição.** Modelo resultante do [pré-treinamento](#pretraining-pré-treinamento): treinado em um
corpus grande e diverso, sabe completar texto, mas ainda não foi especializado em nenhuma tarefa.

**Função no modelo.** É o ponto de partida reaproveitável. Pré-treinar é caro; um modelo base pode
ser ajustado várias vezes, para tarefas diferentes, sem repetir essa etapa.

**Relação com outros conceitos.** Saída do pré-treinamento e entrada do
[fine-tuning](#fine-tuning-ajuste-fino). Também chamado de *pretrained model*.

**Exemplo.** O GPT-3 é o modelo base do qual o ChatGPT original foi derivado. Neste projeto, a
Sprint 5 produz um modelo base pequeno, treinado em `the-verdict.txt`, e o livro mostra como carregar
pesos abertos do GPT-2 para não precisar pré-treinar em grande escala.

---

## GPT (Generative Pretrained Transformer)

**Definição.** Família de LLMs da OpenAI, introduzida em 2018, formada **apenas pela parte de
decoder** do Transformer e pré-treinada com previsão da próxima palavra.

**Função no modelo.** É a arquitetura que este projeto constrói. Processa o texto em um único
sentido, da esquerda para a direita, e gera texto de forma
[autorregressiva](#autoregressive-model-modelo-autorregressivo).

**Relação com outros conceitos.** *Generative*: gera texto. *Pretrained*: passa por
[pré-treinamento](#pretraining-pré-treinamento). *Transformer*: herda a arquitetura de 2017, mas só
o [decoder](#decoder-decodificador). Contrasta com o [BERT](#bert-bidirectional-encoder-representations-from-transformers).

**Exemplo.** O GPT-3 (2020) tem 96 camadas Transformer e 175 bilhões de parâmetros — o Transformer
original repetia seus blocos 6 vezes. O modelo deste projeto seguirá a configuração do GPT-2
pequeno, com cerca de 124 milhões de parâmetros (Sprint 4).

---

## Instruction fine-tuning (fine-tuning por instruções)

**Definição.** Modalidade de [fine-tuning](#fine-tuning-ajuste-fino) em que o conjunto rotulado é
formado por pares **instrução → resposta**.

**Função no modelo.** Ensina o modelo base a seguir comandos em vez de apenas continuar o texto. É o
que transforma um modelo que completa frases em um assistente. Será implementado no Capítulo 7
(Sprint 6).

**Relação com outros conceitos.** Alternativa ao
[fine-tuning para classificação](#classification-fine-tuning-fine-tuning-para-classificação); a saída
continua sendo texto livre, não uma classe.

**Exemplo.** `{"instrução": "Traduza para o inglês: 'bom dia'", "resposta": "good morning"}`.

---

## Labeled and unlabeled data (dados rotulados e não rotulados)

**Definição.** Dados **rotulados** acompanham cada exemplo da resposta esperada (uma classe, uma
tradução, uma resposta). Dados **não rotulados** são apenas o conteúdo bruto — no caso de LLMs,
texto comum, sem nenhuma anotação.

**Função no modelo.** Definem as duas etapas de treinamento. O
[pré-treinamento](#pretraining-pré-treinamento) usa texto não rotulado, disponível em enorme
quantidade; o [fine-tuning](#fine-tuning-ajuste-fino) usa um conjunto rotulado, menor e mais caro de
produzir.

**Relação com outros conceitos.** O [aprendizado autossupervisionado](#self-supervised-learning-aprendizado-autossupervisionado)
é o que permite treinar com texto não rotulado: os rótulos são extraídos do próprio texto.

**Exemplo.** Não rotulado: o conteúdo de `sprints/the-verdict.txt`. Rotulado: uma planilha de
e-mails com a coluna `spam / não-spam` preenchida por pessoas.

---

## Language tasks (tarefas de linguagem)

**Definição.** Ações específicas que um sistema executa para processar, entender ou gerar linguagem
humana: tradução, resumo, análise de sentimento, resposta a perguntas, redação, extração de
informação.

**Função no modelo.** São o "para quê" de um LLM. Modelos de NLP anteriores eram construídos para uma
tarefa de cada vez; um LLM executa muitas delas com o mesmo conjunto de pesos.

**Relação com outros conceitos.** Tarefas que exigem compreensão de contexto e geração de texto
coerente eram o ponto fraco das abordagens baseadas em
[reconhecimento de padrões simples](#straightforward-pattern-recognition-reconhecimento-de-padrões-simples).
O LLM as resolve via [zero-shot](#zero-shot-learning-aprendizado-sem-exemplos),
[few-shot](#few-shot-learning-aprendizado-com-poucos-exemplos) ou [fine-tuning](#fine-tuning-ajuste-fino).

**Exemplo.** Escrever um e-mail a partir de uma lista de palavras-chave — tarefa fora do alcance das
gerações anteriores de modelos de linguagem e trivial para um LLM atual.

---

## Large Language Model (LLM, modelo de linguagem de grande escala)

**Definição.** [Rede neural](#neural-network-rede-neural) profunda, baseada na arquitetura
[Transformer](#transformer-arquitetura-transformer), treinada em grandes volumes de texto para
entender, gerar e responder a textos em linguagem humana. O "grande" se refere a duas coisas: ao
número de [parâmetros](#parameters-parâmetros-pesos) — dezenas a centenas de bilhões — e ao tamanho
do conjunto de treinamento.

**Função no modelo.** É o objeto do projeto inteiro. Ao final do semestre, o repositório terá um LLM
pequeno, construído componente por componente.

**Relação com outros conceitos.** Aplicação de [deep learning](#deep-learning-aprendizado-profundo);
treinado por [previsão da próxima palavra](#next-word-prediction-previsão-da-próxima-palavra) e depois
ajustado por [fine-tuning](#fine-tuning-ajuste-fino). "Entender", aqui, significa processar e gerar
texto de forma coerente e contextualmente adequada — não consciência ou compreensão humana.

**Exemplo.** O GPT-3: 175 bilhões de parâmetros, pré-treinado em 300 bilhões de tokens.

---

## Machine learning (aprendizado de máquina)

**Definição.** Ramo da [IA](#artificial-intelligence-inteligência-artificial) que desenvolve algoritmos
capazes de aprender a partir de dados e fazer previsões ou tomar decisões **sem serem programados
explicitamente** com as regras da tarefa.

**Função no modelo.** É o princípio de todo o projeto: em vez de escrever regras de linguagem, o
modelo ajusta seus [parâmetros](#parameters-parâmetros-pesos) minimizando o erro de previsão em um
conjunto de treinamento.

**Relação com outros conceitos.** Contém o [deep learning](#deep-learning-aprendizado-profundo). No
machine learning tradicional, as características dos dados são escolhidas por especialistas.

**Exemplo.** Um filtro de spam recebe e-mails rotulados como spam ou legítimos e, minimizando os erros
de classificação no conjunto de treino, aprende os padrões que distinguem as duas classes.

---

## Natural Language Processing (NLP, processamento de linguagem natural)

**Definição.** Área da computação dedicada a fazer computadores lerem, interpretarem e produzirem
linguagem humana. Em português, também se usa a sigla PLN.

**Função no modelo.** É a área em que os LLMs se inserem — e que eles transformaram. Antes deles, o
NLP se apoiava em regras explícitas, métodos estatísticos mais simples e modelos treinados para uma
única tarefa.

**Relação com outros conceitos.** Os LLMs elevaram o desempenho em praticamente todas as
[tarefas de linguagem](#language-tasks-tarefas-de-linguagem) do NLP, graças ao
[Transformer](#transformer-arquitetura-transformer) e ao uso de grandes corpora.

**Exemplo.** Antes: um modelo para análise de sentimento, outro para tradução, outro para resumo.
Com um LLM: um único modelo executa as três tarefas.

---

## Neural network (rede neural)

**Definição.** Modelo computacional formado por camadas de unidades que aplicam transformações
lineares (pesos e vieses) seguidas de funções não lineares. Com três ou mais camadas, é chamada de
rede neural profunda.

**Função no modelo.** Um LLM *é* uma rede neural profunda. Todas as peças do projeto — embeddings,
atenção, blocos Transformer — são camadas de uma única rede, treinada de ponta a ponta.

**Relação com outros conceitos.** Instrumento do [deep learning](#deep-learning-aprendizado-profundo);
seus pesos ajustáveis são os [parâmetros](#parameters-parâmetros-pesos). Em PyTorch, é construída como
uma subclasse de `torch.nn.Module` (Apêndice A).

**Exemplo.** `y = W₂ · ReLU(W₁ · x + b₁) + b₂`. É a rede `Linear(3, 4) → ReLU → Linear(4, 1)` que
`src/verificar_ambiente.py` monta para validar o ambiente.

---

## Next-word prediction (previsão da próxima palavra)

**Definição.** Tarefa de prever a palavra (ou, mais precisamente, o token) seguinte de uma sequência a
partir das anteriores.

**Função no modelo.** É **a** tarefa de [pré-treinamento](#pretraining-pré-treinamento) do GPT.
Embora pareça simples, obriga o modelo a capturar sintaxe, semântica, contexto e até conhecimento
geral, porque tudo isso ajuda a prever o que vem a seguir. Ela aproveita a natureza sequencial da
linguagem.

**Relação com outros conceitos.** É uma forma de
[aprendizado autossupervisionado](#self-supervised-learning-aprendizado-autossupervisionado). Na
geração, vira o laço [autorregressivo](#autoregressive-model-modelo-autorregressivo). Os pares
entrada/alvo que a implementam são construídos na Sprint 2.

**Exemplo.** Com o início de `the-verdict.txt`: entrada `"I HAD always"` → alvo `"thought"`;
entrada `"I HAD always thought"` → alvo `"Jack"`.

---

## Parameters (parâmetros, pesos)

**Definição.** Valores numéricos ajustáveis de uma [rede neural](#neural-network-rede-neural) —
pesos e vieses — que são otimizados durante o treinamento.

**Função no modelo.** São o que o modelo aprende. No LLM, são ajustados para que a previsão do
próximo token fique cada vez mais correta. A quantidade de parâmetros é uma das duas medidas do
"grande" em *large language model*.

**Relação com outros conceitos.** Ajustados no [pré-treinamento](#pretraining-pré-treinamento) e,
depois, no [fine-tuning](#fine-tuning-ajuste-fino). Não confundir com os *pesos de atenção* do
Capítulo 3, que são recalculados a cada entrada (ver [capitulo-03.md](capitulo-03.md#weight-parameter-parâmetro-de-peso)).

**Exemplo.** Uma camada `Linear(3, 4)` tem `3 × 4 = 12` pesos e `4` vieses: 16 parâmetros. O GPT-3
tem 175 bilhões.

---

## Pretraining (pré-treinamento)

**Definição.** Primeira etapa de treinamento de um LLM: o modelo é treinado em um corpus grande e
diverso de texto **não rotulado**, com a tarefa de
[previsão da próxima palavra](#next-word-prediction-previsão-da-próxima-palavra). O prefixo "pré"
indica que é uma etapa inicial, sobre a qual outras são construídas.

**Função no modelo.** Dá ao modelo um conhecimento geral de linguagem. O resultado é o
[modelo base](#foundation-model-modelo-de-fundação-modelo-base), capaz de completar texto e com
alguma capacidade [few-shot](#few-shot-learning-aprendizado-com-poucos-exemplos).

**Relação com outros conceitos.** Usa [dados não rotulados](#labeled-and-unlabeled-data-dados-rotulados-e-não-rotulados)
via [aprendizado autossupervisionado](#self-supervised-learning-aprendizado-autossupervisionado);
precede o [fine-tuning](#fine-tuning-ajuste-fino). Assunto do Capítulo 5 (Sprint 5).

**Exemplo.** O custo do pré-treinamento do GPT-3 foi estimado em cerca de US$ 4,6 milhões em
computação na nuvem. Neste projeto, o pré-treinamento é feito em escala educacional, em um único
conto, em hardware comum.

---

## Self-attention (autoatenção)

**Definição.** Mecanismo que permite a cada token de uma sequência ponderar a importância de todos os
outros tokens da mesma sequência ao construir sua própria representação.

**Função no modelo.** É o componente central do [Transformer](#transformer-arquitetura-transformer) e,
portanto, dos LLMs. Captura dependências de longo alcance e relações de contexto: o significado de uma
palavra passa a depender das palavras ao redor.

**Relação com outros conceitos.** Apenas apresentada no Capítulo 1; implementada no Capítulo 3
(Sprint 3, `src/attention.py`). Detalhes em
[capitulo-03.md](capitulo-03.md#self-attention-autoatenção).

**Exemplo.** Em `"O animal não atravessou a rua porque ele estava cansado"`, a self-attention permite
que a representação de `"ele"` dê peso alto a `"animal"`, resolvendo a referência.

---

## Self-supervised learning (aprendizado autossupervisionado)

**Definição.** Forma de aprendizado em que os rótulos não são coletados à parte: são gerados
automaticamente a partir da estrutura dos próprios dados. É uma forma de autorrotulagem.

**Função no modelo.** É o que torna o [pré-treinamento](#pretraining-pré-treinamento) viável em escala.
Como o "rótulo" de cada posição é simplesmente a palavra seguinte do texto, qualquer texto bruto vira
dado de treino, sem trabalho humano de anotação.

**Relação com outros conceitos.** Implementado pela
[previsão da próxima palavra](#next-word-prediction-previsão-da-próxima-palavra); permite usar
[dados não rotulados](#labeled-and-unlabeled-data-dados-rotulados-e-não-rotulados).

**Exemplo.** A frase `"o gato subiu no telhado"` gera, sozinha, quatro exemplos de treino:
`"o" → "gato"`, `"o gato" → "subiu"`, `"o gato subiu" → "no"`, `"o gato subiu no" → "telhado"`.

---

## Straightforward pattern recognition (reconhecimento de padrões simples)

**Definição.** Identificação de regularidades nos dados por meio de regras fixas ou comparações
diretas, que podem ser escritas à mão ou capturadas por modelos simples.

**Função no modelo.** Representa o que os métodos anteriores aos LLMs faziam bem — classificação de
spam, detecção de padrões fixos — e marca o limite que os LLMs superaram: tarefas que exigem
compreensão de contexto e geração de texto.

**Relação com outros conceitos.** Contrasta com as [tarefas de linguagem](#language-tasks-tarefas-de-linguagem)
complexas e com o [deep learning](#deep-learning-aprendizado-profundo), que aprende padrões que não
poderiam ser enumerados por regras.

**Exemplo.** A regra "se o e-mail contém *ganhe dinheiro rápido*, marcar como spam", ou uma expressão
regular que detecta endereços de e-mail.

---

## Token (unidade de texto)

**Definição.** Unidade de texto que o modelo lê e gera. No Capítulo 1, de forma aproximada, o número
de tokens de um texto corresponde ao número de palavras e sinais de pontuação; no Capítulo 2, com o
tokenizador BPE, um token pode ser também um pedaço de palavra.

**Função no modelo.** É a unidade básica de tudo: a entrada é uma sequência de tokens, a saída é um
token por vez, e o tamanho dos corpora de treinamento é medido em tokens.

**Relação com outros conceitos.** Obtido pela tokenização; cada token recebe um ID numérico e depois um
[embedding](#embedding-vetor-de-representação). Assunto da Sprint 2.

**Exemplo.** `"Hello, world."` → `["Hello", ",", "world", "."]`: quatro tokens, como no notebook da
Sprint 2.

---

## Training corpus (corpus de treinamento)

**Definição.** Coleção de textos usada para treinar o modelo. No pré-treinamento de LLMs, são corpora
de bilhões de palavras, cobrindo muitos assuntos, idiomas naturais e linguagens de programação.

**Função no modelo.** A escala e a diversidade do corpus determinam o que o modelo consegue aprender —
sintaxe, semântica, contexto e até conhecimento geral.

**Relação com outros conceitos.** Matéria-prima do [pré-treinamento](#pretraining-pré-treinamento);
composto de [dados não rotulados](#labeled-and-unlabeled-data-dados-rotulados-e-não-rotulados).
Medido em [tokens](#token-unidade-de-texto).

**Exemplo.** O GPT-3 foi treinado com 300 bilhões de tokens amostrados de cinco fontes, das quais a
maior é o CommonCrawl filtrado (410 bilhões de tokens, cerca de 570 GB). O corpus deste projeto é
`sprints/the-verdict.txt`, com 20.479 caracteres.

---

## Transformer (arquitetura Transformer)

**Definição.** Arquitetura de rede neural profunda introduzida em 2017 no artigo *Attention Is All
You Need*, originalmente para tradução automática. É formada por um [encoder](#encoder-codificador) e
um [decoder](#decoder-decodificador), ambos construídos com camadas de
[self-attention](#self-attention-autoatenção).

**Função no modelo.** É a base de praticamente todos os LLMs atuais. A self-attention dá ao modelo
acesso seletivo a toda a sequência de entrada ao gerar cada palavra da saída.

**Relação com outros conceitos.** [BERT](#bert-bidirectional-encoder-representations-from-transformers)
usa o encoder; [GPT](#gpt-generative-pretrained-transformer) usa o decoder. Transformer e LLM são
quase sinônimos na literatura, mas não exatamente: há Transformers para visão computacional e há
LLMs baseados em redes recorrentes ou convolucionais.

**Exemplo.** O Transformer original repetia os blocos de encoder e de decoder 6 vezes e traduzia
textos do inglês para o alemão e o francês.

---

## Zero-shot learning (aprendizado sem exemplos)

**Definição.** Capacidade de executar uma tarefa nunca vista, a partir apenas da descrição da tarefa,
sem nenhum exemplo na entrada e sem treinamento adicional.

**Função no modelo.** É a forma mais direta de usar um LLM: a instrução em linguagem natural basta.
Mostra o grau de generalização obtido no pré-treinamento.

**Relação com outros conceitos.** Contrasta com [few-shot](#few-shot-learning-aprendizado-com-poucos-exemplos)
(alguns exemplos na entrada) e com [fine-tuning](#fine-tuning-ajuste-fino) (altera os pesos).
Decorre do [comportamento emergente](#emergent-behavior-comportamento-emergente).

**Exemplo.** Prompt: `"Traduza para o inglês: 'bom dia'"` → `"good morning"`, sem nenhum par de
exemplo.

---

> **Nota.** Este arquivo é o glossário oficial do Capítulo 1 e substitui o
> [`documentation/Glossario.docx`](../documentation/Glossario.docx), que permanece no repositório como
> registro das anotações feitas durante a leitura. As entradas técnicas daquele documento — *few-shot*,
> BERT, GPT, aprendizado autossupervisionado, comportamento emergente, NLP, reconhecimento de padrões
> simples e tarefas de linguagem — foram revisadas e incorporadas aqui; as traduções de palavras comuns
> do inglês não fazem parte do glossário técnico.
