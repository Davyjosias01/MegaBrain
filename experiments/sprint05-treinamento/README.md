# `sprint05-treinamento` — Treinamento, avaliação e geração

Experimentos da Sprint 5 (Entregáveis 3 a 6), sobre o treinamento implementado em
[`src/treinamento.py`](../../src/treinamento.py) e os checkpoints de [`src/checkpoint.py`](../../src/checkpoint.py).
A interpretação dos resultados está em [`reports/sprint05-relatorio.md`](../../reports/sprint05-relatorio.md).

## Execução

Todos os comandos a partir da raiz do repositório, com o ambiente do uv (`uv sync`).

```bash
uv run python experiments/sprint05-treinamento/treinar_base.py
```

```bash
cd experiments/sprint05-treinamento && uv run python executar_todos.py
```

Ou um experimento isolado (a partir deste diretório):

```bash
uv run python exp02_taxa_de_aprendizado.py
```

`executar_todos.py` também aceita filtros (`uv run python executar_todos.py exp05 exp06`) e roda
`treinar_base.py` antes, se o checkpoint base ainda não existir. Na GTX 1650 o treino base leva ~50 s e
o conjunto E1–E6 ~7,5 min; em CPU, ordens de grandeza a mais.

Todos os artefatos vão para [`results-by-sprints/sprint05/`](../../results-by-sprints/sprint05/).

## Os scripts

| Script | O que faz | Modelo |
|---|---|---|
| `treinar_base.py` | Treino da seção 5.2: 10 épocas, AdamW `lr = 4e-4`, `wd = 0,1`, lote 2, contexto 256 | GPT-2 124M |
| `gerar_com_checkpoint.py` | Carrega o checkpoint e gera texto, sem treinar | o do checkpoint |
| `exp01_numero_de_epocas.py` | épocas ∈ {1, 3, 5, 10, 20, 40} | reduzido |
| `exp02_taxa_de_aprendizado.py` | `lr` ∈ {1e-5, 1e-4, 4e-4, 1e-3, 3e-3, 1e-2} | reduzido |
| `exp03_configuracao_de_treino.py` | lote 1/2/4, weight decay 0/0,1/0,5, dropout 0/0,1/0,3 (20 épocas) | reduzido |
| `exp04_tamanho_do_modelo.py` | `(emb_dim, n_layers, n_heads)` de (64, 2, 2) a (768, 12, 12) | vários |
| `exp05_estrategias_de_geracao.py` | greedy, temperatura, top-k — mesmo contexto | 124M treinado |
| `exp06_geracao_vs_treinamento.py` | as mesmas estratégias com pesos aleatórios × treinados | 124M |

`comum.py` concentra o que é compartilhado: caminhos, semente, dispositivo, o GPT reduzido
(`emb_dim` 256, 4 blocos, 4 cabeças, 29 M parâmetros), o treino rápido parametrizável, as métricas de
texto e a gravação de tabelas e figuras.

## Métricas

**Perda** — entropia cruzada média (`perda_do_loader`), medida no conjunto de treino e no de validação
(90/10 do corpus). **Perplexidade** — `exp(perda)`. **Lacuna** — perda de validação − perda de treino:
quanto maior, maior o sobreajuste.

Sobre os tokens gerados (média de 5 sementes):

- **distinct-n** — fração de n-gramas distintos (1 = nenhuma repetição);
- **repetição** — fração de trigramas que repetem um trigrama anterior do mesmo texto;
- **cópia do treino** — fração dos 8-gramas gerados que aparecem literalmente no texto de treinamento
  (memorização).

## Salvamento e carregamento dos pesos

`treinar_base.py` grava o checkpoint em `results-by-sprints/sprint05/base-modelo.pt` (1,95 GB). Os
arquivos `*.pt` **não são versionados** (`.gitignore`); para obtê-lo, rode o treino base.

O checkpoint contém:

| Chave | Conteúdo |
|---|---|
| `modelo` | `state_dict` do `GPTModel` — os pesos |
| `otimizador` | `state_dict` do AdamW — médias móveis e contador de passos |
| `cfg_modelo` | configuração da arquitetura (`vocab_size`, `context_length`, `emb_dim`, `n_heads`, `n_layers`, `drop_rate`, `qkv_bias`) |
| `cfg_treino` | `ConfigTreinamento`: épocas, lote, contexto, stride, divisão, otimizador, `lr`, weight decay, avaliação, semente, dispositivo, corpus |
| `historico` | perdas por avaliação, tokens vistos e amostras por época |
| `versoes` | Python e PyTorch que gravaram o arquivo |

A mesma configuração, em texto, fica em `results-by-sprints/sprint05/base-config.json`, que é
versionado. Ela identifica o modelo e permite reproduzi-lo mesmo sem o `.pt`.

**Gerar texto com o modelo salvo:**

```bash
uv run python experiments/sprint05-treinamento/gerar_com_checkpoint.py --contexto "Every effort moves you" --tokens 50 --temperatura 1.0 --top-k 10
```

**Em código:**

```python
from checkpoint import carregar_checkpoint, retomar_treinamento, salvar_checkpoint

modelo, dados = carregar_checkpoint("results-by-sprints/sprint05/base-modelo.pt", "cuda")  # pronto para inferência
modelo, otimizador, dados = retomar_treinamento("results-by-sprints/sprint05/base-modelo.pt")  # continuar treinando
salvar_checkpoint("novo.pt", modelo, otimizador, dados["cfg_treino"], historico)
```

A validação ([`src/validar_treinamento.py`](../../src/validar_treinamento.py)) confere que o modelo
carregado produz logits e texto idênticos e que um treino retomado chega aos mesmos pesos de um treino
sem interrupção.

## Reprodutibilidade

- Semente 123 antes dos loaders e antes do modelo em cada treino. Repetir `executar_todos.py` reproduz
  todas as perdas e métricas; só os tempos variam.
- Ambiente registrado em `execucao.log` e `base-execucao.log` (Python, PyTorch, GPU, sistema).
- Corpus local `sprints/the-verdict.txt`, sem downloads. Para usar outro corpus, altere
  `ConfigTreinamento.corpus`.
