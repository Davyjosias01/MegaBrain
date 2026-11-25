# `sprint06-fine-tuning` — Pesos pré-treinados e fine-tuning

Experimentos da Sprint 6 (Entregáveis 2 a 5), sobre o carregamento de pesos de
[`src/pesos_pretreinados.py`](../../src/pesos_pretreinados.py), o fine-tuning de classificação de
[`src/classificacao.py`](../../src/classificacao.py) e o de instruções de
[`src/instrucoes.py`](../../src/instrucoes.py). A interpretação dos resultados está em
[`reports/sprint06-relatorio.md`](../../reports/sprint06-relatorio.md).

## Execução

Todos os comandos a partir da raiz do repositório, com o ambiente do uv (`uv sync`).

Os dois treinos (cada um grava o seu checkpoint em `modelos/`, que não é versionada):

```bash
uv run python experiments/sprint06-fine-tuning/treinar_classificador.py
```

```bash
uv run python experiments/sprint06-fine-tuning/treinar_instrucoes.py
```

Todos os experimentos em sequência, ou só alguns:

```bash
cd experiments/sprint06-fine-tuning && uv run python executar_todos.py
```

```bash
cd experiments/sprint06-fine-tuning && uv run python executar_todos.py exp05 exp06
```

`executar_todos.py` roda antes o treino cujo checkpoint estiver faltando. Na primeira execução, os
pesos do GPT-2 124M (548 MB) são baixados de `openai-community/gpt2`, no Hugging Face Hub, para
`modelos/gpt2-small.safetensors`.

Todos os artefatos vão para [`results-by-sprints/sprint06/`](../../results-by-sprints/sprint06/).

## Os scripts

| Script | O que faz | Tempo em CPU |
|---|---|---|
| `exp01_aleatorio_vs_pretreinado.py` | mesma arquitetura com pesos aleatórios × pesos do GPT-2: perda, perplexidade, próximo token e amostras de geração | minutos |
| `treinar_classificador.py` | treino base do classificador de spam: 5 épocas, AdamW `lr = 5e-5`, `wd = 0,1`, lote 8, último bloco + norma final + cabeça | 40 min |
| `exp02_antes_e_depois.py` | conjunto de teste antes × depois do ajuste: matriz de confusão, precisão, revocação, F1 e mensagens erradas | minutos |
| `exp03_camadas_treinaveis.py` | só a cabeça × último bloco × 6 últimos blocos | 32 + 54 min |
| `exp04_token_de_classificacao.py` | classe lida no último × no primeiro token | 34 min |
| `treinar_instrucoes.py` | fine-tuning de instruções: 2 épocas, AdamW `lr = 5e-5`, `wd = 0,1`, lote 8, 6 últimos blocos + norma final | ver `instrucoes-config.json` |
| `exp05_respostas_de_teste.py` | gera as respostas às 107 instruções de teste com o modelo pré-treinado e com o ajustado | minutos |
| `exp06_avaliacao_das_respostas.py` | correspondência exata, F1 de palavras, comprimento das respostas e amostra para leitura | segundos |

`comum.py` concentra o que é compartilhado: caminhos, semente, dispositivo, as configurações dos dois
treinos (`TREINO_CLASSIFICADOR`, `TREINO_INSTRUCOES`), o treino parametrizável do classificador e a
gravação de tabelas e figuras. Os experimentos 03 e 04 reaproveitam o resultado do treino base em vez
de refazê-lo.

## Dados

| Base | Arquivos | Tamanho |
|---|---|---|
| *SMS Spam Collection* (UCI, CC BY 4.0) | `sprints/dados/sms-spam/` | 5.574 mensagens → 5.171 únicas → 1.306 balanceadas → 914 / 130 / 262 |
| Instruções do livro | `sprints/dados/instrucoes/instruction-data.json` | 1.100 exemplos → 1.075 sem repetição → 913 / 55 / 107 |

Nas duas bases as repetições são removidas **antes** da divisão, para que nenhum exemplo de teste
apareça também no treino. O livro não faz essa remoção; por isso os tamanhos diferem dos dele.

## Métricas

**Classificação** (spam é a classe positiva): **acurácia** — fração de acertos; **precisão** —
VP / (VP + FP); **revocação** — VP / (VP + FN); **F1** — média harmônica das duas. Todas implementadas
em `src/classificacao.py`, sem scikit-learn.

**Instruções**: **correspondência exata** — a resposta gerada tem as mesmas palavras da esperada, na
mesma ordem, ignorando caixa e pontuação; **F1 de palavras** — sobreposição de palavras entre as duas;
**palavras por resposta** — mostra se o modelo aprendeu a parar. As duas primeiras medem semelhança de
texto, não correção: por isso o experimento 06 também grava uma amostra para leitura.

## Limitações da execução

Tudo foi executado em CPU, com 7,5 GB de RAM. Consequências:

- o experimento 03 compara com **6 blocos** treináveis, e não com o modelo inteiro;
- o fine-tuning de instruções usa o GPT-2 **124M** com os 6 últimos blocos treináveis; o livro ajusta o
  355M inteiro. Para repetir como no livro numa GPU, basta trocar `tamanho` para `"medium"` e
  `blocos_treinaveis` para `None` em `TREINO_INSTRUCOES` (`comum.py`);
- a avaliação das respostas com um LLM juiz (seção 7.8 do livro) não foi feita.
