# Capítulo 5 — Pretraining on Unlabeled Data

Glossário dos termos introduzidos no Capítulo 5 de RASCHKA, S. *Build a Large Language Model
(From Scratch)*: treinamento e avaliação de um modelo de linguagem. Entradas em ordem alfabética.

O capítulo transforma o `GPTModel` do Capítulo 4 — que, com pesos aleatórios, só gera tokens sem
relação entre si — em um modelo que prevê o próximo token do seu corpus. O caminho é sempre o mesmo:

```
lote (entrada, alvo)           (b, L) e (b, L)
  → modelo                     logits (b, L, V)
  → entropia cruzada           perda (escalar)          ← avaliação
  → backward                   gradiente de cada peso
  → otimizador (AdamW)         pesos atualizados        ← treinamento
  → repetir por lotes e épocas, medindo treino e validação
```

Além dos conceitos, cada entrada diz a **função no treinamento** e a **relação com os demais
componentes do modelo**, como pede o enunciado da Sprint 5. Os exemplos numéricos vêm das execuções
do projeto (`results-by-sprints/sprint05/`): GPT-2 124M com contexto de 256 tokens, treinado sobre
`the-verdict.txt`. Os termos *greedy decoding*, *temperatura* e *top-k* já estão no
[glossário do Capítulo 4](capitulo-04.md); aqui aparecem apenas as
[estratégias de decodificação](#decoding-strategy-estratégia-de-decodificação) como conjunto.

Notação usada nas fórmulas:

| Símbolo | Significado |
|---|---|
| `b` | tamanho do lote (`batch_size`) |
| `L` | número de tokens por sequência (`max_length`) |
| `V` | tamanho do vocabulário — 50.257 no GPT-2 |
| `θ` | o conjunto de todos os pesos do modelo |
| `η` | taxa de aprendizado (`lr`) |
| `ℒ` | perda (entropia cruzada média) |

---

## AdamW (otimizador Adam com weight decay desacoplado)

**Definição.** Otimizador que, para cada peso, mantém duas médias móveis: a do gradiente (`m`,
"momento") e a do gradiente ao quadrado (`v`). O passo de atualização é `η · m̂ / (√v̂ + ε)`, com
`m̂` e `v̂` corrigidos pelo número de passos já dados. O "W" indica que o
[weight decay](#weight-decay-decaimento-de-pesos) é aplicado diretamente aos pesos, separado do
gradiente.

**Função no treinamento.** É quem efetivamente altera os pesos a partir dos gradientes calculados
pela [retropropagação](#backpropagation-retropropagação). A divisão por `√v̂` dá a cada peso um passo
de tamanho adaptado: pesos com gradiente grande e ruidoso andam menos; pesos com gradiente pequeno e
consistente andam mais.

**Relação com outros conceitos.** É o [otimizador](#optimizer-otimizador) usado no livro e no
projeto. Depende da [taxa de aprendizado](#learning-rate-taxa-de-aprendizado) e do
[weight decay](#weight-decay-decaimento-de-pesos). Como guarda `m`, `v` e o contador de passos, seu
estado precisa entrar no [checkpoint](#checkpoint-ponto-de-salvamento) para que o treino possa ser
retomado.

**Exemplo.** `torch.optim.AdamW(modelo.parameters(), lr=4e-4, weight_decay=0.1)` — configuração da
seção 5.2, registrada em `ConfigTreinamento` (`src/treinamento.py`). Como guarda duas médias por peso,
o estado do AdamW ocupa o dobro da memória dos pesos: o checkpoint do GPT-2 124M tem 1,95 GB, contra
~620 MB só de pesos. No teste da task C2, retomar o treino **sem** esse estado levou a pesos diferentes
(diferença de 7·10⁻³) dos de um treino sem interrupção; **com** o estado, a diferença foi zero.

---

## Backpropagation (retropropagação)

**Definição.** Algoritmo que calcula, pela regra da cadeia, a derivada da perda em relação a cada peso
do modelo, percorrendo o grafo de computação de trás para frente — da perda até os embeddings.

**Função no treinamento.** Responde, para cada peso, "em que direção e com que intensidade mudar este
peso reduziria a perda?". Sem ela, não haveria como saber como ajustar os 162 milhões de parâmetros.
No PyTorch, uma única chamada `perda.backward()` preenche o atributo `.grad` de todos os pesos.

**Relação com outros conceitos.** Parte da [perda](#cross-entropy-loss-perda-de-entropia-cruzada)
e produz os [gradientes](#gradient-gradiente) que o [otimizador](#optimizer-otimizador) usa. É o
passo 3 do [laço de treinamento](#training-loop-laço-de-treinamento). As
[conexões residuais](capitulo-04.md#residual-connection-conexão-residual-shortcut-ou-skip-connection)
do Capítulo 4 existem justamente para que o gradiente chegue às primeiras camadas sem desaparecer.

**Exemplo.** No teste da task A1, `perda_do_lote(...).backward()` sobre um GPT-2 small preencheu o
gradiente da cabeça de saída (`out_head.weight`, forma `(50257, 768)`). Na avaliação
(`perda_do_loader`) a retropropagação é desligada com `torch.no_grad()`: não há o que atualizar.

---

## Checkpoint (ponto de salvamento)

**Definição.** Arquivo que guarda o estado de um treinamento num dado momento: os pesos do modelo
(`state_dict`), o estado do otimizador, a configuração da arquitetura e do treino e, opcionalmente, o
histórico das perdas.

**Função no treinamento.** Preserva o resultado do treino — que custa tempo e energia — para uso
posterior, sem treinar de novo: inferência, comparação entre versões, ou **retomada** de um treino
interrompido.

**Relação com outros conceitos.** Contém o [`state_dict`](#state_dict-dicionário-de-estado) do
modelo e o do [AdamW](#adamw-otimizador-adam-com-weight-decay-desacoplado). Sem a
[configuração do modelo](capitulo-04.md#gpt-model-configuration-configuração-do-modelo) gravada junto,
não seria possível recriar o `GPTModel` antes de carregar os pesos.

**Exemplo.** `salvar_checkpoint` e `carregar_checkpoint` (`src/checkpoint.py`) gravam e leem
`results-by-sprints/sprint05/base-modelo.pt`. O modelo carregado produziu logits idênticos aos do
modelo salvo e, com geração greedy, exatamente a mesma amostra registrada na época 10 do treino (task
D3). Os arquivos `.pt` não são versionados no Git.

---

## Cross-entropy loss (perda de entropia cruzada)

**Definição.** Para cada posição, transforma os [logits](capitulo-04.md#logits) em probabilidades com
softmax e toma o negativo do logaritmo da probabilidade atribuída ao token correto. A perda do lote é a
média sobre todas as `b · L` posições:

```
ℒ = −(1 / (b·L)) · Σ log p(alvo correto)
```

**Função no treinamento.** É o número que o treinamento minimiza e a medida principal de avaliação. É
0 quando o modelo dá probabilidade 1 ao token certo e cresce sem limite quando dá probabilidade
próxima de zero — penaliza fortemente a confiança errada.

**Relação com outros conceitos.** Compara a saída do modelo com o alvo da
[janela deslizante](capitulo-02.md#sliding-window-janela-deslizante) — a entrada deslocada de uma
posição. Sua exponencial é a [perplexidade](#perplexity-perplexidade). É o ponto de partida da
[retropropagação](#backpropagation-retropropagação). Medida separadamente em treino e validação, revela
o [sobreajuste](#overfitting-sobreajuste).

**Exemplo.** Um modelo que espalha a probabilidade uniformemente entre os 50.257 tokens tem perda
`ln(50257) ≈ 10,82`. O GPT-2 124M recém-inicializado do projeto começou com 10,99 (treino) e 10,98
(validação); após 10 épocas, terminou com 0,83 e 6,36. Implementação: `perda_do_lote`, com
`torch.nn.functional.cross_entropy(logits.flatten(0, 1), alvo.flatten())`.

---

## Decoding strategy (estratégia de decodificação)

**Definição.** A regra que escolhe o próximo token a partir das probabilidades produzidas pelo modelo:
[greedy](capitulo-04.md#greedy-decoding-decodificação-gulosa) (o mais provável),
[amostragem](capitulo-04.md#multinomial-sampling-amostragem-multinomial),
[temperatura](capitulo-04.md#temperature-temperatura), [top-k](capitulo-04.md#top-k-sampling-amostragem-top-k)
ou combinações.

**Função no treinamento.** Nenhuma: a estratégia atua só na geração, depois do treino, e não altera
nenhum peso. Por isso é o controle mais barato sobre o texto produzido — e também por isso não
substitui o treino.

**Relação com outros conceitos.** Opera sobre a distribuição que os pesos aprenderam. A mesma
estratégia produz textos muito diferentes antes e depois do [treinamento](#training-loop-laço-de-treinamento);
greedy em um modelo [sobreajustado](#overfitting-sobreajuste) tende a reproduzir o corpus
([memorização](#memorization-memorização)).

**Exemplo.** Experimentos 05 e 06, com o mesmo contexto ("Every effort moves you"): no modelo
treinado, greedy copiou 44% dos 8-gramas do texto de treino, temperatura 0,7 caiu para 7% e
temperatura ≥ 1 para 0%; no mesmo modelo sem treino, **todas** as estratégias produziram tokens soltos,
com diversidade ~1 e cópia 0.

---

## Epoch (época)

**Definição.** Uma passagem completa por todos os lotes do conjunto de treinamento.

**Função no treinamento.** É a unidade de duração do treino. O número de épocas define quantas vezes
cada amostra é vista e, junto com o número de lotes, quantos passos de otimização acontecem.

**Relação com outros conceitos.** Uma época tem `len(loader_treino)` [passos](#training-step-passo-de-treinamento).
Com o [DataLoader](capitulo-02.md#dataloader-carregador-de-dados) em `shuffle=True`, a ordem dos lotes
muda a cada época. Épocas demais em poucos dados levam ao [sobreajuste](#overfitting-sobreajuste);
a [perda de validação](#validation-loss-perda-de-validação) indica quando parar.

**Exemplo.** Com 18 amostras de treino e lote 2, cada época tem 9 passos; 10 épocas = 90 passos e
46.080 tokens vistos. No experimento 01, a validação teve o menor valor (6,07) por volta da época 17;
com 40 épocas o treino foi a 0,07 e a validação subiu para 6,83.

---

## Evaluation frequency (frequência de avaliação)

**Definição.** De quantos em quantos passos as perdas de treino e de validação são medidas durante o
treinamento (`eval_freq` no livro), e sobre quantos lotes (`eval_iter`).

**Função no treinamento.** Produz as curvas que mostram se o modelo está aprendendo e quando começa o
sobreajuste, sem pagar o custo de avaliar todos os dados a cada passo.

**Relação com outros conceitos.** Chama a [perda por loader](#cross-entropy-loss-perda-de-entropia-cruzada)
em modo [`eval()`](capitulo-04.md#eval-e-train-modos-de-execução-do-modelo), sem gradientes. Gera o
histórico gravado no [checkpoint](#checkpoint-ponto-de-salvamento).

**Exemplo.** `treinar_modelo(..., frequencia_avaliacao=5, lotes_avaliacao=5)`: avalia nos passos 0, 5,
10, … sobre os 5 primeiros lotes de cada conjunto. O histórico do treino base está em
`results-by-sprints/sprint05/base-historico.csv`.

---

## Generalization (generalização)

**Definição.** Capacidade do modelo de prever bem dados que não viu no treinamento.

**Função no treinamento.** É o objetivo real: um modelo de linguagem útil precisa prever o próximo
token de textos novos, não só os do corpus. A perda de treino mede ajuste; só a de validação mede
generalização.

**Relação com outros conceitos.** Medida pela [perda de validação](#validation-loss-perda-de-validação).
O oposto é o [sobreajuste](#overfitting-sobreajuste). Melhora com mais dados e com regularização
([dropout](capitulo-04.md#dropout-no-modelo-gpt), [weight decay](#weight-decay-decaimento-de-pesos)).

**Exemplo.** No experimento 04, crescer o modelo de 6,5 M para 162 M parâmetros levou a perda de
treino de 7,26 a 0,48, mas a de validação parou em torno de 6,15 a partir de 49 M: com ~4.600 tokens de
treino, mais capacidade não trouxe mais generalização.

---

## Gradient (gradiente)

**Definição.** Vetor das derivadas parciais da perda em relação a cada peso: indica a direção em que a
perda mais cresce. Andar no sentido oposto reduz a perda.

**Função no treinamento.** É a informação que o otimizador usa para atualizar os pesos.

**Relação com outros conceitos.** Calculado pela [retropropagação](#backpropagation-retropropagação),
zerado por [`zero_grad`](#zero_grad-zerar-gradientes) a cada passo e consumido pelo
[otimizador](#optimizer-otimizador). A [taxa de aprendizado](#learning-rate-taxa-de-aprendizado)
controla o tamanho do passo na sua direção.

**Exemplo.** Depois de `perda.backward()`, `modelo.out_head.weight.grad` tem a mesma forma do peso,
`(50257, 768)`. Na avaliação o gradiente não é calculado: o teste da task A2 confirma que nenhum
`.grad` é preenchido por `perda_do_loader`.

---

## Learning rate (taxa de aprendizado)

**Definição.** O fator `η` que multiplica a atualização dos pesos em cada passo do otimizador.

**Função no treinamento.** Controla a velocidade e a estabilidade da convergência. Pequena demais: o
modelo quase não sai do ponto inicial no tempo disponível. Grande demais: os passos ultrapassam o
mínimo, a perda oscila, e o modelo pode memorizar rapidamente ou divergir.

**Relação com outros conceitos.** Parâmetro do [AdamW](#adamw-otimizador-adam-com-weight-decay-desacoplado)
e registrado na configuração do treino. Interage com o número de [épocas](#epoch-época): uma taxa maior
chega antes ao [sobreajuste](#overfitting-sobreajuste).

**Exemplo.** Experimento 02, 10 épocas: `1e-5` terminou com perda de treino 10,62 (quase ln V);
`4e-4` (livro) com 4,86/6,28 (treino/validação); `1e-3` teve a melhor validação, 6,09; `3e-3` decorou
o treino (0,48) e piorou a validação (6,87); `1e-2` foi instável, com a perda subindo em 4 avaliações.

---

## Memorization (memorização)

**Definição.** Quando o modelo reproduz trechos literais dos dados de treinamento em vez de produzir
texto novo plausível.

**Função no treinamento.** É o sintoma visível, no texto gerado, de que o treino foi além da
generalização. Em modelos grandes treinados em poucos dados, é o resultado esperado.

**Relação com outros conceitos.** Acompanha o [sobreajuste](#overfitting-sobreajuste): perda de treino
perto de zero significa que o modelo prevê o corpus quase com certeza. A
[estratégia de decodificação](#decoding-strategy-estratégia-de-decodificação) pode escondê-la
(temperatura alta) ou expô-la (greedy).

**Exemplo.** No projeto, medida como a fração dos 8-gramas gerados que aparecem literalmente no texto
de treino (`copia_do_treino`, `experiments/sprint05-treinamento/comum.py`). Com o modelo base, greedy
copiou 44%; a amostra de 40 épocas do experimento 01 é uma frase do conto: *"Yes--quite insensible to
the irony. She wanted him vindicated--and by me!"*.

---

## Optimizer (otimizador)

**Definição.** Algoritmo que atualiza os pesos do modelo a partir dos gradientes, segundo uma regra
própria (SGD, Adam, AdamW…).

**Função no treinamento.** Executa o passo de aprendizado: `otimizador.step()` aplica a atualização a
todos os pesos registrados.

**Relação com outros conceitos.** Recebe os [gradientes](#gradient-gradiente) da
[retropropagação](#backpropagation-retropropagação), usa a
[taxa de aprendizado](#learning-rate-taxa-de-aprendizado) e, no caso do
[AdamW](#adamw-otimizador-adam-com-weight-decay-desacoplado), o
[weight decay](#weight-decay-decaimento-de-pesos). Tem estado próprio, salvo no
[checkpoint](#checkpoint-ponto-de-salvamento).

**Exemplo.** `ConfigTreinamento(otimizador="AdamW").criar_otimizador(modelo)` instancia
`torch.optim.AdamW` com a taxa e o weight decay registrados; o nome fica no JSON da configuração.

---

## Overfitting (sobreajuste)

**Definição.** Situação em que o modelo continua melhorando nos dados de treino enquanto piora — ou
para de melhorar — nos de validação: ajusta-se às particularidades do conjunto de treino em vez dos
padrões gerais da língua.

**Função no treinamento.** É o principal risco de treinar um modelo grande com poucos dados e o
critério para decidir quando parar ou regularizar.

**Relação com outros conceitos.** Detectado comparando a [perda de treino](#training-loss-perda-de-treinamento)
com a [perda de validação](#validation-loss-perda-de-validação). Reduzido por
[dropout](capitulo-04.md#dropout-no-modelo-gpt), [weight decay](#weight-decay-decaimento-de-pesos),
menos [épocas](#epoch-época) ou mais dados. Aparece no texto como [memorização](#memorization-memorização).

**Exemplo.** No treino base, a validação parou em torno de 6,3 a partir de ~10 mil tokens vistos,
enquanto o treino seguiu caindo até 0,83 (`base-perdas.png`). No experimento 03, aumentar o dropout de
0 para 0,3 reduziu a lacuna validação − treino de 4,86 para 2,64.

---

## Perplexity (perplexidade)

**Definição.** A exponencial da perda de entropia cruzada: `perplexidade = exp(ℒ)`.

**Função no treinamento.** Traduz a perda numa escala interpretável: o número efetivo de tokens entre
os quais o modelo está "em dúvida" a cada previsão. É mais intuitiva para comparar modelos e relatar
resultados.

**Relação com outros conceitos.** Contém a mesma informação da
[entropia cruzada](#cross-entropy-loss-perda-de-entropia-cruzada), em outra escala. Vale `V` (50.257)
para um modelo uniforme e 1 para um modelo perfeito.

**Exemplo.** `perplexidade(perda)` em `src/treinamento.py`. GPT-2 124M antes do treino: ~59 mil, na
ordem de `V`. Depois de 10 épocas: 2,3 no treino — o modelo hesita entre ~2 tokens — e 581 na
validação, em texto que nunca viu.

---

## `state_dict` (dicionário de estado)

**Definição.** Dicionário do PyTorch que associa o nome de cada parâmetro e buffer de um módulo (ou de
um otimizador) ao seu tensor.

**Função no treinamento.** É a forma portátil de salvar e carregar pesos: `modelo.state_dict()` para
gravar, `modelo.load_state_dict(...)` para restaurar num modelo com a mesma arquitetura.

**Relação com outros conceitos.** Principal conteúdo do [checkpoint](#checkpoint-ponto-de-salvamento).
Só tem tensores: para recriar o modelo é preciso também a
[configuração](capitulo-04.md#gpt-model-configuration-configuração-do-modelo).

**Exemplo.** Em `carregar_checkpoint`: `GPTModel(dados["cfg_modelo"])` cria a arquitetura e
`load_state_dict(dados["modelo"])` coloca os pesos. O `torch.load` usa `weights_only=True`, que aceita
apenas tensores e tipos básicos — por isso a versão do PyTorch é gravada como texto.

---

## Tokens seen (tokens vistos)

**Definição.** Total de tokens de entrada processados pelo treinamento até um dado passo:
`passos · b · L`.

**Função no treinamento.** Eixo natural para comparar curvas de treinos com lotes ou contextos
diferentes, já que mede a quantidade de dados consumida, e não a quantidade de passos.

**Relação com outros conceitos.** Cresce `b · L` a cada [passo](#training-step-passo-de-treinamento) e
`len(loader) · b · L` a cada [época](#epoch-época).

**Exemplo.** Treino base: lote 2 × 256 tokens = 512 tokens por passo; 90 passos = 46.080 tokens.
Registrado em `historico["tokens_vistos"]` e usado no eixo x de `base-perdas.png`.

---

## Train/validation split (divisão treino/validação)

**Definição.** Separação do corpus em um conjunto usado para ajustar os pesos (treinamento) e outro,
nunca usado para atualizá-los, que serve só para medir o desempenho (validação).

**Função no treinamento.** Sem um conjunto separado, não há como distinguir aprendizado de
memorização.

**Relação com outros conceitos.** Alimenta dois [DataLoaders](capitulo-02.md#dataloader-carregador-de-dados):
o de treino embaralha e descarta o lote incompleto; o de validação não. Fornece a
[perda de validação](#validation-loss-perda-de-validação).

**Exemplo.** `dividir_treino_validacao(texto, 0.9)` em `src/dados.py`: os primeiros 90% dos caracteres
(18.431) para treino e os últimos 10% (2.048) para validação, sem embaralhar — 4.612 e 534 tokens,
9 lotes de treino e 1 de validação, como no livro.

---

## Training loop (laço de treinamento)

**Definição.** O procedimento que repete, para cada [época](#epoch-época) e cada lote: zerar os
gradientes, calcular a perda, retropropagar e atualizar os pesos — avaliando periodicamente.

**Função no treinamento.** É o treinamento em si: o único lugar onde os pesos mudam.

**Relação com outros conceitos.** Encadeia [`zero_grad`](#zero_grad-zerar-gradientes),
[entropia cruzada](#cross-entropy-loss-perda-de-entropia-cruzada),
[retropropagação](#backpropagation-retropropagação) e [otimizador](#optimizer-otimizador); usa a
[frequência de avaliação](#evaluation-frequency-frequência-de-avaliação) e o modo
[`train()`](capitulo-04.md#eval-e-train-modos-de-execução-do-modelo), que ativa o dropout.

**Exemplo.** `treinar_modelo` em `src/treinamento.py` (equivalente ao `train_model_simple` do livro).
No teste da task B1, 40 épocas sobre um único lote reduziram a perda de 10,83 para 4,43, prova de que
o laço de fato ajusta os pesos.

---

## Training loss (perda de treinamento)

**Definição.** A entropia cruzada medida nos dados de treino.

**Função no treinamento.** Mostra se o otimizador está conseguindo ajustar o modelo aos dados. Deve
cair; se não cai, há erro no laço, taxa de aprendizado inadequada ou modelo pequeno demais.

**Relação com outros conceitos.** Comparada com a [perda de validação](#validation-loss-perda-de-validação),
diagnostica o [sobreajuste](#overfitting-sobreajuste). Tende a zero quando o modelo
[memoriza](#memorization-memorização) o corpus.

**Exemplo.** Treino base: 10,99 → 0,83 em 10 épocas. Experimento 01: 0,07 com 40 épocas.

---

## Training step (passo de treinamento)

**Definição.** Uma única atualização dos pesos, feita a partir de um lote.

**Função no treinamento.** É a unidade mínima de aprendizado; o número total de passos determina,
junto com a taxa de aprendizado, o quanto os pesos se afastam do ponto inicial.

**Relação com outros conceitos.** Uma [época](#epoch-época) tem `len(loader)` passos; o
[tamanho do lote](capitulo-02.md#batch-lote) define quantos passos cabem numa época.

**Exemplo.** Experimento 03, 20 épocas: lote 1 deu 360 passos (e o maior sobreajuste, validação 6,55);
lote 2, 180; lote 4, 80 (descartando 2 amostras por `drop_last`) e a menor lacuna validação − treino.

---

## Validation loss (perda de validação)

**Definição.** A entropia cruzada medida nos dados de validação, que o modelo nunca usa para ajustar
os pesos.

**Função no treinamento.** É a estimativa de quão bem o modelo
[generaliza](#generalization-generalização); indica a melhor época para parar e compara
configurações.

**Relação com outros conceitos.** Vem da [divisão treino/validação](#trainvalidation-split-divisão-treinovalidação).
A diferença em relação à [perda de treino](#training-loss-perda-de-treinamento) mede o
[sobreajuste](#overfitting-sobreajuste).

**Exemplo.** Treino base: 10,98 → 6,36, com o mínimo atingido bem antes do fim. Experimento 01: mínimo
6,07 por volta da época 17 e 6,83 na época 40.

---

## Weight decay (decaimento de pesos)

**Definição.** Termo que reduz cada peso em uma pequena fração a cada passo (`θ ← θ − η·λ·θ`),
puxando-o para zero.

**Função no treinamento.** É uma regularização: impede que pesos cresçam demais para encaixar
exatamente o conjunto de treino.

**Relação com outros conceitos.** Parâmetro do [AdamW](#adamw-otimizador-adam-com-weight-decay-desacoplado),
que o aplica separado do gradiente. Atua contra o [sobreajuste](#overfitting-sobreajuste), como o
[dropout](capitulo-04.md#dropout-no-modelo-gpt).

**Exemplo.** O livro usa `weight_decay=0.1`. No experimento 03 (20 épocas), 0, 0,1 e 0,5 deram
validação entre 6,10 e 6,13: nessa escala de treino curto, o efeito é pequeno comparado ao do dropout
ou ao do lote.

---

## `zero_grad` (zerar gradientes)

**Definição.** Método do otimizador que apaga os gradientes acumulados nos pesos.

**Função no treinamento.** O PyTorch **soma** os gradientes a cada `backward()`. Sem zerar no início
de cada passo, o gradiente de um lote se acumularia com os dos anteriores e a atualização ficaria
errada.

**Relação com outros conceitos.** Primeiro passo do [laço de treinamento](#training-loop-laço-de-treinamento),
antes da [retropropagação](#backpropagation-retropropagação).

**Exemplo.** Em `treinar_modelo`: `otimizador.zero_grad()` → `perda_do_lote(...)` →
`perda.backward()` → `otimizador.step()`.
