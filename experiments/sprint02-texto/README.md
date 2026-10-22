# `sprint02-texto` — Experimentos do processamento de texto

Experimentos da Sprint 2, sobre o pipeline implementado em [`src/tokenizacao.py`](../../src/tokenizacao.py),
[`src/dados.py`](../../src/dados.py) e [`src/embeddings.py`](../../src/embeddings.py). Cada script varia
**uma** dimensão do problema e mede o efeito sobre os dados produzidos, a memória ou o tempo.

A interpretação dos resultados está em [`reports/sprint02-relatorio.md`](../../reports/sprint02-relatorio.md).

## Execução

```bash
cd experiments/sprint02-texto
python executar_todos.py
```

Ou um experimento isolado:

```bash
python exp08_embeddings_posicionais.py
```

Todos os artefatos vão para [`results-by-sprints/sprint02/`](../../results-by-sprints/sprint02/) —
8 figuras `.png`, 8 tabelas `.csv` e o registro completo do console em `execucao.log`, que inclui o
ambiente de execução (versões do PyTorch e do tiktoken, número de threads, processador).

## Os experimentos

Os sete primeiros correspondem, na ordem, às sugestões de experimentação da Sprint 2; o oitavo é a
demonstração experimental pedida no item 3.5 (*Positional Embeddings*).

| Script | O que varia | O que mede |
|---|---|---|
| `exp01_tamanho_de_contexto.py` | `max_length` ∈ {4 … 1024} | amostras, corpus aproveitado, memória do lote e da matriz de atenção |
| `exp02_tamanho_de_lote.py` | `batch_size` ∈ {1 … 64} | lotes por época, amostras descartadas por `drop_last`, tempo por amostra |
| `exp03_dimensao_embedding.py` | `d_emb` ∈ {2 … 1600} | parâmetros e memória das tabelas, tempo, similaridade entre tokens |
| `exp04_sequencias_de_entrada.py` | a frase de entrada | como cada tokenizador fatia o texto, `<\|unk\|>`, reconstrução |
| `exp05_tokens_por_texto.py` | inglês, português, código, números, palavras raras | caracteres por token, tokens por palavra, fração de `<\|unk\|>` |
| `exp06_contexto_e_amostras.py` | `max_length` × `stride` | amostras (fórmula × dataset real), sobreposição, reuso dos alvos |
| `exp07_dimensoes_do_pipeline.py` | configuração (livro, Sprint 3, GPT-2 small) | forma, tipo e memória de cada estrutura, do texto à entrada do modelo |
| `exp08_embeddings_posicionais.py` | presença do embedding posicional | se o mesmo token em posições diferentes, e frases com ordem trocada, se distinguem |

`comum.py` concentra o que é compartilhado: caminhos, semente, textos de teste, estilo dos gráficos e
medidas de tempo e memória. `executar_todos.py` roda os oito na ordem e registra o ambiente no log.

## Medidas

**Corpus aproveitado** — fração dos tokens do corpus que aparece como alvo de previsão em uma época,
depois que o `DataLoader` descarta o lote incompleto (`drop_last=True`).

**Reuso médio dos alvos** — quantas vezes, em média, cada token que aparece como alvo é previsto em uma
época. Vale 1 sem sobreposição; com `stride = 1` vale aproximadamente `max_length`.

**Similaridade de cosseno** — `cos(u, v) = u·v / (‖u‖‖v‖)`, em `[-1, 1]`. Mede se dois vetores apontam
na mesma direção, independentemente do tamanho. É 1 para vetores idênticos.

## Reprodutibilidade

- Semente fixa em 123, reaplicada antes de instanciar cada camada de embedding.
- Corpus local `sprints/the-verdict.txt` — nenhum experimento depende de download.
- Textos de teste fixos em `comum.py`, para que os resultados não mudem se outros arquivos forem editados.
- **Nenhum módulo é treinado.** As medidas sobre embeddings usam pesos na inicialização: descrevem a
  estrutura do pipeline, não relações semânticas aprendidas.
- Tempos dependem da máquina e variam entre execuções; leia-os como comparações relativas.
