# `experiments/` — Experimentos

Scripts e configurações dos experimentos executados em cada sprint: variações de parâmetros,
configurações de arquitetura ou conjuntos de dados, com o objetivo de observar o **impacto** dessas
alterações sobre o comportamento, o desempenho ou o custo computacional do modelo.

Exemplos do tipo de pergunta investigada aqui: como o tamanho do vocabulário afeta a tokenização, qual
o efeito da dimensão dos *embeddings*, como o número de cabeças de atenção altera o custo de
processamento, ou qual a influência da taxa de aprendizado sobre a curva de perda.

Um experimento reutiliza os componentes de [`src/`](../src/) e varia apenas o que está sob teste.

## Experimentos realizados

| Diretório | Sprint | Assunto |
|---|---|---|
| [`sprint02-texto/`](sprint02-texto/) | 2 | Oito experimentos sobre o processamento de texto: contexto, lote, dimensão do embedding, sequências e textos diferentes, contexto × amostras, dimensões do pipeline e embeddings posicionais |
| [`sprint03-atencao/`](sprint03-atencao/) | 3 | Oito experimentos sobre os mecanismos de atenção: dimensões, número e tamanho das cabeças, escala, máscara causal e sequências de entrada |
| [`sprint04-gpt/`](sprint04-gpt/) | 4 | Oito experimentos sobre a arquitetura GPT e a geração: blocos, heads, dimensão dos embeddings e da FeedForward, dropout, contexto, residuais e LayerNorm, temperatura e top-k |

## Convenções

- Cada experimento em seu próprio arquivo ou subdiretório, nomeado como `sprintNN-assunto`
- Deixar registrado o que foi variado, quais valores foram testados e por quê
- Fixar a semente aleatória, para que os resultados sejam reproduzíveis
- Os números e gráficos gerados vão para [`results-by-sprints/`](../results-by-sprints/); a
  interpretação deles vai para [`reports/`](../reports/)
