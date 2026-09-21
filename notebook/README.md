# `notebook/` — Notebooks Jupyter

Notebooks de exploração e demonstração. Servem para acompanhar passo a passo o comportamento de cada
componente do modelo — inspecionar tensores intermediários, visualizar matrizes de atenção, plotar
curvas de perda e demonstrar o funcionamento de uma etapa antes de consolidá-la em código.

A lógica definitiva mora em [`src/`](../src/); os notebooks **importam** esses módulos em vez de
duplicar a implementação. Isso mantém uma única fonte de verdade para o código do modelo.

## Convenções

- Nomear como `sprintNN-assunto.ipynb` (ex.: `sprint02-tokenizacao.ipynb`)
- Salvar o notebook com as saídas executadas, para que seja legível sem reexecução
- Manter células de texto explicando o que cada trecho demonstra — o notebook é material de estudo,
  não apenas código

## Notebooks

| Arquivo | Sprint | Assunto |
|---|---|---|
| [`sprint03-mecanismos-de-atencao.ipynb`](sprint03-mecanismos-de-atencao.ipynb) | 3 | Os quatro mecanismos de atenção do Capítulo 3, passo a passo, com os exercícios 3.1, 3.2 e 3.3 |

## Execução

Com o ambiente virtual ativo, a partir da raiz do repositório:

```bash
jupyter lab
```
