# Sprint 6 — Plano de implementação (Capítulos 6 e 7: pesos pré-treinados e fine-tuning)

Enunciado: [`documentation/instrucoes-sprints/sprint-06-fine-tuning.txt`](../../instrucoes-sprints/sprint-06-fine-tuning.txt).
Entrega: **03/12/2026**, junto com a entrega final do projeto. Branch: `sprint-6`, criada a
partir da `develop`.

## Ponto de partida

O que já existe e será reutilizado (sem reescrever):

| Componente | Onde | Uso na Sprint 6 |
|---|---|---|
| `GPTModel`, `config_gpt2`, `contar_parametros` | `src/gpt.py` | arquitetura que recebe os pesos pré-treinados |
| `TokenizadorBPE` | `src/tokenizacao.py` | mesmo BPE do GPT-2; `<|endoftext|>` (50256) como padding |
| `gerar`, `texto_para_ids`, `ids_para_texto` | `src/geracao.py` | inferência com os pesos carregados e geração das respostas |
| `perda_do_lote`, `perda_do_loader`, `treinar_modelo`, `ConfigTreinamento` | `src/treinamento.py` | o fine-tuning de instruções é o mesmo laço da Sprint 5, com outro loader |
| `salvar_checkpoint`, `carregar_checkpoint` | `src/checkpoint.py` | guardar os modelos ajustados |
| padrão de `validar_*.py`, `comum.py`, `executar_todos.py` | `src/`, `experiments/` | mesma organização |

## Decisões e pendências

1. **Origem dos pesos.** O livro lê os checkpoints do TensorFlow publicados pela OpenAI, o que exigiria
   instalar o TensorFlow só para ler um arquivo (e ele pode não ter build para o Python 3.14 fixado no
   projeto). Proposta: baixar os mesmos pesos no formato `safetensors` (`openai-community/gpt2` no
   Hugging Face Hub) e lê-los com a biblioteca `safetensors`, que devolve apenas um dicionário
   nome → tensor. O mapeamento nome → componente continua sendo escrito por nós, que é o que o
   Entregável 2 pede. Não usar `transformers` para instanciar o modelo.
2. **Configuração para carregar o GPT-2.** `context_length = 1024` e `qkv_bias = True` (a Sprint 5
   treinou com 256 e `False`). No arquivo da OpenAI as camadas lineares são `Conv1D`, com o peso
   transposto em relação a `nn.Linear`, e Q, K e V vêm concatenados em um único `c_attn`: é preciso
   transpor e dividir em três.
3. **Dados.** A pasta do ambiente virtual da UNOESC exige login. Até os arquivos serem colocados em
   `sprints/dados/`, o plano usa as bases do livro: *SMS Spam Collection* (UCI, 5.572 mensagens) para a
   classificação e `instruction-data.json` (1.100 exemplos) para as instruções. O caminho dos dados
   fica parametrizado.
4. **Tamanho do modelo.** Classificação com o GPT-2 124M. Instruções: o livro usa o 355M; começar com
   o 124M e subir para o 355M só se a GPU disponível comportar. **Resultado:** a sprint foi executada
   em CPU com 7,5 GB de RAM; as instruções ficaram no 124M, com os 6 últimos blocos treináveis.
5. **Além dos cinco entregáveis.** O enunciado tem cinco entregáveis. A task E4 (avaliação das
   respostas) vem do termo "Response Evaluation" exigido no glossário, e o bloco G (documentação e
   integração final) vem da introdução do enunciado e da entrega final do cronograma; ambos seguem o
   padrão das sprints anteriores.
6. **Arquivos grandes.** Pesos baixados e checkpoints ficam em `modelos/` (não versionado; `.gitignore`
   já cobre `*.safetensors` e `*.pt`).

## Tasks

Cada task é implementada, testada e commitada separadamente. A coluna **Teste** é o que precisa passar
antes do commit.

### Bloco A — pesos pré-treinados (Entregável 2)

| # | Task | Arquivo | Teste |
|---|---|---|---|
| A1 | `baixar_pesos_gpt2(tamanho, pasta)`: baixa o `safetensors` uma vez e reaproveita o arquivo local; `uv add safetensors` | `src/pesos_pretreinados.py` | segunda chamada não acessa a rede; o arquivo abre e contém `wte.weight` com shape `(50257, 768)` |
| A2 | `mapa_de_parametros(cfg)`: tabela nome do arquivo → parâmetro do `GPTModel` (transposição das `Conv1D`, divisão de `c_attn` em Q/K/V) | `src/pesos_pretreinados.py` | todo parâmetro do modelo aparece exatamente uma vez; nenhum tensor do arquivo fica sem destino (exceto os buffers de máscara) |
| A3 | `carregar_pesos_gpt2(modelo, pesos)`: copia com conferência de shape, erro claro em caso de divergência | `src/pesos_pretreinados.py` | shape errado levanta `ValueError`; após carregar, `out_head.weight` é igual ao embedding de tokens (*weight tying*) |
| A4 | Inferência com os pesos carregados | `src/pesos_pretreinados.py` | perda em `the-verdict.txt` muito abaixo de `ln(50257) ≈ 10,8` (o modelo aleatório fica em ≈ 10,8); geração greedy de um contexto fixo é texto em inglês coerente |
| A5 | Experimento aleatório × pré-treinado: perda, perplexidade e amostras com o mesmo contexto e as mesmas estratégias de geração | `experiments/sprint06-fine-tuning/exp01_aleatorio_vs_pretreinado.py` | gera o CSV, a figura e o arquivo de amostras |

### Bloco B — dados e modelo de classificação (Entregável 3)

| # | Task | Arquivo | Teste |
|---|---|---|---|
| B1 | Carregar a base de spam, balancear as classes (subamostragem de *ham*), dividir 70/10/20 com semente fixa e gravar os CSV | `src/classificacao.py` | classes 50/50; partições sem interseção; soma dos tamanhos igual ao total balanceado |
| B2 | `DatasetDeClassificacao`: tokeniza, trunca no contexto e completa com o token 50256 até o maior comprimento do treino; `criar_loaders_classificacao` | `src/classificacao.py` | todos os tensores do lote com o mesmo comprimento; rótulos em {0, 1}; validação e teste usam o comprimento do treino |
| B3 | `adaptar_para_classificacao(modelo, num_classes, blocos_treinaveis=1)`: troca `out_head` por `Linear(emb_dim, num_classes)` e congela tudo, exceto a nova cabeça, os últimos blocos e a `final_norm` | `src/classificacao.py` | saída com shape `(lote, tokens, 2)`; contagem de parâmetros treináveis confere com a soma manual; pesos congelados não mudam após um passo do otimizador |
| B4 | `perda_classificacao_do_lote`, `perda_classificacao_do_loader` e `acuracia_do_loader`, usando o logit do **último token** | `src/classificacao.py` | perda igual à entropia cruzada manual; acurácia 1,0 e 0,0 em casos construídos; antes do ajuste a acurácia fica próxima de 50% |
| B5 | `treinar_classificador(...)`: épocas × lotes, avaliação periódica, histórico de perda e acurácia | `src/classificacao.py` | em um subconjunto pequeno a perda cai e a acurácia de treino sobe; histórico com o tamanho esperado |
| B6 | `classificar_texto(texto, modelo, ...)` → rótulo e probabilidades | `src/classificacao.py` | devolve "spam"/"não spam"; trunca textos maiores que o contexto sem erro |

### Bloco C — treino e avaliação do classificador (Entregável 4)

| # | Task | Arquivo | Teste |
|---|---|---|---|
| C1 | Treino base reprodutível (5 épocas, AdamW `lr = 5e-5`, `weight_decay = 0.1`, lote 8): grava configuração, histórico, curvas e checkpoint | `experiments/sprint06-fine-tuning/treinar_classificador.py` | executa até o fim; acurácia de validação acima de 90%; arquivos em `results-by-sprints/sprint06/` |
| C2 | `metricas_de_classificacao(loader, modelo)`: matriz de confusão, precisão, revocação e F1, implementadas sem scikit-learn | `src/classificacao.py` | valores iguais aos calculados à mão em um exemplo pequeno; divisão por zero tratada |
| C3 | Avaliação no conjunto de **teste**, antes × depois do fine-tuning, com exemplos de erros | `experiments/sprint06-fine-tuning/exp02_antes_e_depois.py` | gera a tabela comparativa, a matriz de confusão e a lista de mensagens classificadas errado |
| C4 | Experimento do congelamento: só a cabeça × último bloco × 6 últimos blocos (o modelo inteiro não coube na RAM da máquina usada) — acurácia, parâmetros treináveis e tempo | `experiments/sprint06-fine-tuning/exp03_camadas_treinaveis.py` | gera o CSV e a figura |
| C5 | Experimento do token usado na classificação: primeiro × último | `experiments/sprint06-fine-tuning/exp04_token_de_classificacao.py` | gera o CSV; a análise liga o resultado à máscara causal da Sprint 3 |

### Bloco D — dados de instruções (Entregável 5)

| # | Task | Arquivo | Teste |
|---|---|---|---|
| D1 | Carregar o JSON de instruções e `formatar_entrada(exemplo)` no formato Alpaca (instrução, entrada opcional, resposta); divisão 85/5/10 | `src/instrucoes.py` | exemplo com e sem `input` formatado como esperado; partições sem interseção |
| D2 | `DatasetDeInstrucoes` e `agrupar_lote(...)` (*collate*): padding até o maior exemplo **do lote**, alvo deslocado em uma posição, padding do alvo trocado por `-100` (mantendo um `<|endoftext|>`), truncamento opcional | `src/instrucoes.py` | no lote de três sequências do livro, entradas e alvos iguais aos esperados; a entropia cruzada ignora as posições `-100` |
| D3 | `criar_loaders_instrucoes(...)` | `src/instrucoes.py` | lotes com comprimentos diferentes entre si e iguais dentro de cada lote |

### Bloco E — fine-tuning de instruções (Entregável 5)

| # | Task | Arquivo | Teste |
|---|---|---|---|
| E1 | Treino de instruções reaproveitando `treinar_modelo` da Sprint 5 (2 épocas, AdamW `lr = 5e-5`, lote 8), com checkpoint | `experiments/sprint06-fine-tuning/treinar_instrucoes.py` | executa até o fim; perda de validação final menor que a inicial |
| E2 | `responder(instrucao, entrada, modelo)`: gera até `<|endoftext|>` e devolve só o trecho depois de `### Response:` | `src/instrucoes.py` | a resposta não contém o prompt; a geração para no token de fim |
| E3 | Respostas do conjunto de teste gravadas em JSON, antes × depois do fine-tuning | `experiments/sprint06-fine-tuning/exp05_respostas_de_teste.py` | um registro por exemplo de teste, com instrução, resposta esperada e as duas respostas do modelo |
| E4 | Avaliação das respostas (*Response Evaluation*): correspondência exata, sobreposição de tokens (F1) e amostra para análise manual. O livro usa um LLM local (Ollama) como juiz; fica como extensão opcional, pois o projeto não usa modelos prontos além dos pesos do GPT-2 | `src/instrucoes.py`, `experiments/sprint06-fine-tuning/exp06_avaliacao_das_respostas.py` | métricas iguais às calculadas à mão em pares construídos; gera o CSV e o resumo |

### Bloco F — validação

| # | Task | Arquivo | Teste |
|---|---|---|---|
| F1 | `validar_fine_tuning.py`: reúne as verificações de A, B, C2, D e E2 no formato dos outros `validar_*.py`. As que dependem dos pesos baixados são puladas, com aviso, se o arquivo não existir | `src/validar_fine_tuning.py` | todas as verificações aprovadas |
| F2 | `comum.py` e `executar_todos.py` dos experimentos | `experiments/sprint06-fine-tuning/` | executa todos os experimentos em sequência |

### Bloco G — documentação e integração final (Entregável 1 e entrega final)

| # | Task | Arquivo | Teste |
|---|---|---|---|
| G1 | Glossário dos Capítulos 6 e 7, com os 14 termos exigidos no enunciado | `technical-glossary/capitulo-06.md` e `capitulo-07.md` (um arquivo por capítulo, como pede o README do glossário) | os 14 termos presentes, no formato do README do glossário; links válidos |
| G2 | Notebook didático da Sprint 6 | `notebook/sprint06-fine-tuning.ipynb` | executa do zero sem erro (`uv run jupyter nbconvert --execute`) |
| G3 | Relatório da Sprint 6: mapeamento dos pesos, aleatório × pré-treinado, classificação (antes × depois, métricas, congelamento), instruções, limitações | `reports/sprint06-relatorio.md` | todos os links e figuras apontam para arquivos existentes |
| G4 | READMEs: raiz (situação, entregáveis, comandos), `src/`, `experiments/`, `results-by-sprints/`, `technical-glossary/` | vários | links válidos; todos os `validar_*.py` passam |
| G5 | Integração final: script que percorre o pipeline inteiro (texto → tokens → embeddings → atenção → GPT → treino → geração → fine-tuning) com os módulos de `src/` | `src/demonstracao_pipeline.py` | executa do início ao fim em CPU com um modelo pequeno |
| G6 | Entrega final: glossário consolidado e relatório técnico final do projeto | `technical-glossary/`, `reports/relatorio-final.md` | todas as entradas dos capítulos 1 a 7 presentes; links válidos |

## Ordem e dependências

A1 → A2 → A3 → A4 → A5 → B1 … B6 → C1 … C5 → D1 → D2 → D3 → E1 … E4 → F1 → F2 → G1 … G6.

Os blocos B–C (classificação) e D–E (instruções) só dependem do bloco A e são independentes entre si.
G1 pode ser feito em paralelo a qualquer momento.

## Fora do escopo desta sprint

LoRA (Apêndice E do livro) e avaliação das respostas com um LLM juiz ficam como extensões opcionais.
