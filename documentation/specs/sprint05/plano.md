# Sprint 5 — Plano de implementação (Capítulo 5: treinamento e avaliação)

Enunciado: [`documentation/instrucoes-sprints/sprint-05-treinamento.txt`](../../instrucoes-sprints/sprint-05-treinamento.txt).
Entrega: **12/11/2026**. Branch: `sprint-5`, criada a partir da `develop`.

## Ponto de partida

O que já existe e será reutilizado (sem reescrever):

| Componente | Onde | Uso na Sprint 5 |
|---|---|---|
| `TokenizadorBPE`, `GPTDatasetV1`, `criar_dataloader` | `src/tokenizacao.py`, `src/dados.py` | lotes de treino e validação |
| `GPTModel`, `config_gpt2`, `contar_parametros` | `src/gpt.py` | o modelo a ser treinado |
| `gerar`, `gerar_texto_simples`, `ajustar_logits`, `texto_para_ids`, `ids_para_texto` | `src/geracao.py` | greedy, temperatura e top-k (já implementados na Sprint 4) |
| padrão de `validar_*.py`, `comum.py`, `executar_todos.py` | `src/`, `experiments/` | mesma organização para a Sprint 5 |

**Pendência — dados:** o enunciado aponta para uma pasta no ambiente virtual da UNOESC, que exige
login. Até os arquivos serem colocados no repositório, o plano usa `sprints/the-verdict.txt` (o corpus
do capítulo). O carregamento do corpus fica parametrizado, então trocar os dados não muda o código.

## Configuração base do treinamento

GPT-2 124M reduzido, como no capítulo: `context_length = 256`, `emb_dim = 768`, `n_heads = 12`,
`n_layers = 12`, `drop_rate = 0.1`, `qkv_bias = False`; divisão **90% treino / 10% validação**;
`batch_size = 2`, `max_length = stride = 256`; **AdamW** com `lr = 4e-4` e `weight_decay = 0.1`;
**10 épocas**; avaliação a cada 5 passos; semente 123. Todos os valores ficam registrados no log e no
checkpoint.

## Tasks

Cada task é implementada, testada e commitada separadamente. A coluna **Teste** é o que precisa passar
antes do commit.

### Bloco A — avaliação (Entregável 2)

| # | Task | Arquivo | Teste |
|---|---|---|---|
| A1 | `perda_do_lote(entrada, alvo, modelo, dispositivo)`: logits → `cross_entropy` com `flatten` | `src/treinamento.py` | igual a `-mean(log_softmax[alvo])` calculado à mão; modelo sem treino dá perda ≈ `ln(50257) ≈ 10,8` |
| A2 | `perda_do_loader(loader, modelo, dispositivo, num_lotes=None)`, em `eval()` e `no_grad()` | `src/treinamento.py` | média igual à média manual dos lotes; respeita `num_lotes`; loader vazio dá `nan` |
| A3 | `perplexidade(perda)` = `exp(perda)` | `src/treinamento.py` | `exp(ln 50257) = 50257`; perplexidade inicial ≈ tamanho do vocabulário |
| A4 | `dividir_treino_validacao(texto, proporcao=0.9)` e `criar_loaders_treino_validacao(...)` | `src/dados.py` | divisão sem sobreposição e sem perda de caracteres; número de lotes de cada loader confere com `numero_de_amostras` |

### Bloco B — treinamento (Entregável 3)

| # | Task | Arquivo | Teste |
|---|---|---|---|
| B1 | `treinar_modelo(...)`: épocas × lotes, `zero_grad` → perda → `backward` → `step`; avaliação periódica; devolve o histórico (perdas, tokens vistos, passo) | `src/treinamento.py` | em um lote fixo pequeno a perda cai mais de 50% (sobreajuste proposital); os pesos mudam; o histórico tem o tamanho esperado |
| B2 | Registro da configuração: `ConfigTreinamento` (épocas, lote, lr, otimizador, weight decay, semente, dispositivo) serializável em JSON | `src/treinamento.py` | ida e volta pelo JSON preserva todos os campos |
| B3 | Amostra de texto ao fim de cada época, com o mesmo contexto inicial | `src/treinamento.py` | a amostra é gerada sem alterar o modo `train()` do modelo |

### Bloco C — salvamento e carregamento (Entregável 6)

| # | Task | Arquivo | Teste |
|---|---|---|---|
| C1 | `salvar_checkpoint(caminho, modelo, otimizador, cfg_modelo, cfg_treino, historico)` e `carregar_checkpoint(caminho)` | `src/checkpoint.py` | após carregar, os logits são idênticos; a geração greedy produz o mesmo texto |
| C2 | Retomada do treinamento a partir do checkpoint (estado do AdamW incluído) | `src/checkpoint.py` | treinar N passos de uma vez == treinar k, salvar, carregar e treinar N−k (mesma semente, sem dropout) |

### Bloco D — validação e treinamento base

| # | Task | Arquivo | Teste |
|---|---|---|---|
| D1 | `validar_treinamento.py`: reúne as verificações de A, B e C, no formato dos outros `validar_*.py` | `src/validar_treinamento.py` | todas as verificações aprovadas |
| D2 | Treinamento base reprodutível: treina com a configuração base e grava curvas de perda, perplexidade, amostras por época, log e o checkpoint (`*.pt`, não versionado) | `experiments/sprint05-treinamento/treinar_base.py` | executa até o fim; perda de treino final < perda inicial; arquivos gerados em `results-by-sprints/sprint05/` |
| D3 | Script de inferência: carrega o checkpoint e gera texto | `experiments/sprint05-treinamento/gerar_com_checkpoint.py` | gera texto a partir do checkpoint de D2, sem treinar |

### Bloco E — experimentos (Entregáveis 4 e 5)

Infraestrutura: `comum.py` e `executar_todos.py` no padrão das sprints anteriores. Cada experimento é
uma task, com teste "executa e gera o CSV e a figura".

| # | Experimento | O que varia | O que mede |
|---|---|---|---|
| E1 | Número de épocas | 1, 5, 10, 20 | perda treino × validação, ponto de sobreajuste, tempo |
| E2 | Taxa de aprendizado | 1e-5 … 1e-2 | convergência, instabilidade, perda final |
| E3 | Configuração de treinamento | tamanho do lote, `weight_decay`, `drop_rate` | generalização (lacuna treino–validação), custo |
| E4 | Tamanho do modelo | `n_layers` / `emb_dim` reduzidos | perda × parâmetros × tempo |
| E5 | Estratégias de geração | greedy, amostragem pura, temperatura (0,1 … 5), top-k (1 … 50) com o **mesmo contexto** | diversidade (distinct-n), repetição, coerência qualitativa |
| E6 | Geração × treinamento | as mesmas estratégias no modelo sem treino e no treinado | separa o efeito da decodificação do efeito dos pesos |

### Bloco F — documentação (Entregáveis 1 e 7)

| # | Task | Arquivo | Teste |
|---|---|---|---|
| F1 | Glossário do Capítulo 5 (cross-entropy, perplexity, época, lote, AdamW, learning rate, weight decay, overfitting, greedy, temperatura, top-k, checkpoint, `state_dict`…) com definição, função no treinamento e relações | `technical-glossary/capitulo-05.md` | todas as entradas no formato do README; links válidos |
| F2 | Notebook didático da Sprint 5 | `notebook/sprint05-treinamento.ipynb` | executa do zero sem erro (`uv run jupyter nbconvert --execute`) |
| F3 | Relatório da Sprint 5: configurações, resultados quantitativos e qualitativos, sobreajuste, limitações | `reports/sprint05-relatorio.md` | todos os links e figuras apontam para arquivos existentes |
| F4 | READMEs: raiz (situação e entregáveis), `src/`, `experiments/`, `results-by-sprints/`, `technical-glossary/`, procedimento de salvamento e carregamento | vários | links válidos; todos os `validar_*.py` passam no ambiente uv |

## Ordem e dependências

A1 → A2 → A3 → A4 → B1 → B2 → B3 → C1 → C2 → D1 → D2 → D3 → E1…E6 → F1…F4.
F1 pode ser feito em paralelo a qualquer momento.

## Fora do escopo desta sprint

Carregar os pesos pré-treinados da OpenAI (seção 5.5 do livro) não é pedido no enunciado; fica como
extensão opcional, se sobrar tempo.
