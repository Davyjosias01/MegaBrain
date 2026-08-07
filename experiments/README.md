# `experiments/` — Experimentos

Scripts e configurações dos experimentos executados em cada sprint: variações de parâmetros,
configurações de arquitetura ou conjuntos de dados, com o objetivo de observar o **impacto** dessas
alterações sobre o comportamento, o desempenho ou o custo computacional do modelo.

Exemplos do tipo de pergunta investigada aqui: como o tamanho do vocabulário afeta a tokenização, qual
o efeito da dimensão dos *embeddings*, como o número de cabeças de atenção altera o custo de
processamento, ou qual a influência da taxa de aprendizado sobre a curva de perda.

Um experimento reutiliza os componentes de [`src/`](../src/) e varia apenas o que está sob teste.

## Convenções

- Cada experimento em seu próprio arquivo ou subdiretório, nomeado como `sprintNN-assunto`
- Deixar registrado o que foi variado, quais valores foram testados e por quê
- Fixar a semente aleatória, para que os resultados sejam reproduzíveis
- Os números e gráficos gerados vão para [`results-by-sprints/`](../results-by-sprints/); a
  interpretação deles vai para [`reports/`](../reports/)
