# `sprint04-gpt` — Experimentos da arquitetura GPT

Experimentos da Sprint 4 (Entregável 5), sobre o modelo implementado em [`src/gpt.py`](../../src/gpt.py) e a
geração em [`src/geracao.py`](../../src/geracao.py). Cada script varia **um** parâmetro da arquitetura ou da
geração e mede o efeito sobre o custo computacional e sobre o comportamento do modelo.

A interpretação dos resultados está em [`reports/sprint04-relatorio.md`](../../reports/sprint04-relatorio.md).

## Execução

```bash
cd experiments/sprint04-gpt
python executar_todos.py
```

Ou um experimento isolado:

```bash
python exp07_residuais_e_layernorm.py
```

Todos os artefatos vão para [`results-by-sprints/sprint04/`](../../results-by-sprints/sprint04/) — figuras `.png`,
tabelas `.csv` e o registro completo do console em `execucao.log`, com o ambiente e a configuração-base.

## Configuração-base

Para que as varreduras caibam em minutos de CPU, os experimentos 01 a 07 partem de um GPT reduzido, com o
vocabulário do GPT-2 e escala menor; cada experimento altera só o parâmetro sob teste.

| `vocab_size` | `context_length` | `emb_dim` | `n_heads` | `n_layers` | `drop_rate` | `qkv_bias` |
|---:|---:|---:|---:|---:|---:|:---:|
| 50.257 | 256 | 256 | 4 | 4 | 0,1 | não |

O experimento 08 usa o GPT-2 small completo (124M), como no livro.

## Os experimentos

| Script | O que varia | O que mede |
|---|---|---|
| `exp01_numero_de_blocos.py` | `n_layers` ∈ {1 … 12} | parâmetros, tempo dos blocos e total, tempo por token gerado, escala do fluxo residual |
| `exp02_numero_de_heads.py` | `n_heads` ∈ {1 … 16}, `emb_dim` fixo | parâmetros, tamanho das matrizes de atenção, tempo |
| `exp03_dimensao_embedding.py` | `emb_dim` ∈ {64 … 768} | parâmetros e onde estão (vocabulário × blocos), tempo dos blocos × cabeça de saída |
| `exp04_dimensao_feedforward.py` | `ff_dim` = 1×, 2×, 4×, 8× `emb_dim` | parâmetros e tempo da feed forward contra a atenção |
| `exp05_dropout.py` | `drop_rate` ∈ {0 … 0,5}, `train()` × `eval()` | variação dos logits entre passagens, geração em modo de treino, tempo |
| `exp06_tamanho_do_contexto.py` | tokens na sequência (16 … 1.024); passo da geração | custo quadrático da atenção; custo de cada passo da geração sem cache |
| `exp07_residuais_e_layernorm.py` | bloco completo, sem residual, sem LayerNorm, sem os dois; 2 … 24 blocos | escala das ativações e norma do gradiente por bloco, perda inicial |
| `exp08_estrategias_de_geracao.py` | greedy, temperatura, top-k | distribuições do exemplo da seção 5.3; entropia, diversidade e determinismo da geração no GPT-2 |

`comum.py` concentra o que é compartilhado: caminhos, semente, configuração-base, o lote real do corpus,
medidas de tempo e de comportamento (entropia do próximo token) e o estilo dos gráficos.

## Reprodutibilidade

- Semente 123 para a inicialização de todos os modelos; o experimento 08 usa as sementes 0 a 4 para os sorteios.
- Lotes reais de `sprints/the-verdict.txt`, pelo pipeline da Sprint 2.
- **Nenhum modelo é treinado.** As medidas de comportamento usam pesos de inicialização: descrevem a
  arquitetura, não o que ela aprende. O experimento 07 faz uma única retropropagação, sem atualizar pesos.
- Tempos de CPU, dependentes da máquina: leia-os como comparações relativas.
