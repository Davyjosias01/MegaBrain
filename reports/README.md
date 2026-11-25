# `reports/` — Relatórios técnicos

Relatórios de análise de cada sprint. É aqui que os resultados são **interpretados**, e não apenas
apresentados.

O critério de avaliação da disciplina é explícito neste ponto: um relatório não pode se limitar a
exibir códigos, gráficos ou valores numéricos. Os resultados precisam ser discutidos tecnicamente e
relacionados aos conceitos estudados no capítulo correspondente.

## Estrutura sugerida de um relatório

1. **Objetivo** — o que a sprint se propôs a implementar
2. **Implementação** — decisões técnicas tomadas e como os componentes funcionam
3. **Experimentos** — o que foi variado e sob quais condições
4. **Resultados** — dados obtidos, referenciando os artefatos em [`results-by-sprints/`](../results-by-sprints/)
5. **Análise** — interpretação dos resultados à luz dos conceitos do capítulo
6. **Limitações e próximos passos**

## Relatórios

| Arquivo | Sprint | Assunto |
|---|---|---|
| [`sprint01-relatorio.md`](sprint01-relatorio.md) | 1 | Introdução aos LLMs: síntese do capítulo e mapa conceitual da arquitetura GPT |
| [`sprint02-relatorio.md`](sprint02-relatorio.md) | 2 | Processamento de texto: implementação, oito experimentos e respostas às perguntas da sprint |
| [`sprint03-relatorio.md`](sprint03-relatorio.md) | 3 | Mecanismos de atenção: implementação, oito experimentos e análise |
| [`sprint04-relatorio.md`](sprint04-relatorio.md) | 4 | Arquitetura GPT: Transformer Block, modelo, inferência e geração, oito experimentos e análise |
| [`sprint05-relatorio.md`](sprint05-relatorio.md) | 5 | Treinamento e avaliação: perda, perplexidade, treino base, seis experimentos, sobreajuste, estratégias de geração e checkpoints |
| [`sprint06-relatorio.md`](sprint06-relatorio.md) | 6 | Pesos pré-treinados e fine-tuning: carga dos pesos do GPT-2, classificação de spam (antes × depois, métricas, congelamento), instruções (treino, respostas e avaliação) e limitações |
| [`relatorio-final.md`](relatorio-final.md) | Entrega final | Relatório técnico do projeto: o pipeline completo, o que cada sprint construiu, principais resultados, limitações e como reproduzir |

## Convenções

- Um arquivo por sprint, nomeado como `sprintNN-relatorio.md`
- O relatório técnico consolidado do projeto é entregue em 03/12, junto ao repositório final
