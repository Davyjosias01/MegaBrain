# `results-by-sprints/` — Resultados por sprint

Artefatos produzidos pela execução do código e dos experimentos de cada sprint: métricas, gráficos,
logs de treinamento, amostras de texto gerado e checkpoints do modelo.

Guarda o **registro histórico** da evolução do projeto. Preservar os resultados de cada sprint permite
comparar versões do modelo ao longo do semestre e demonstrar o desenvolvimento incremental exigido
pela avaliação.

## Organização

Um subdiretório por sprint:

```
results-by-sprints/
├── sprint01/
├── sprint02/
└── ...
```

## Convenções

- Registrar, junto aos resultados, a configuração que os produziu (hiperparâmetros, semente, hardware)
- Não sobrescrever resultados de sprints anteriores — cada sprint tem seu próprio diretório
- Checkpoints grandes de modelo não devem ser versionados no Git; manter apenas as métricas, os
  gráficos e os logs
