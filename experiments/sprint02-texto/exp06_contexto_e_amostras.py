"""Experimento 06 — relação entre tamanho do contexto e quantidade de amostras.

**O que varia:** ``max_length`` em {4, 16, 64, 256, 1024} e ``stride`` como
fração do contexto: 1 token, ¼, ½, 1 e 2 vezes o contexto.
**O que fica fixo:** corpus ``the-verdict.txt`` (5.145 tokens BPE).

**Pergunta:** quantas amostras de treinamento um corpus produz, e quanto da
informação delas é nova?

A janela deslizante produz ``ceil((N - max_length) / stride)`` amostras
(``numero_de_amostras`` em ``src/dados.py``). O experimento confere a fórmula
contra o dataset real e mede três consequências:

- **sobreposição** — fração de cada janela repetida da janela anterior;
- **reuso dos alvos** — quantas vezes, em média, cada token aparece como alvo
  numa época. Acima de 1 há redundância; o modelo vê os mesmos exemplos várias
  vezes, o que aumenta o risco de sobreajuste;
- **tokens nunca usados como alvo** — com ``stride > max_length`` a janela
  pula trechos do corpus.

A conclusão esperada é que o número de amostras é controlado sobretudo pelo
``stride``: com ``stride = 1`` ele quase não depende do contexto (``≈ N``),
mas as amostras passam a ser cópias deslocadas umas das outras.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from comum import (
    CATEGORICA,
    MARCADORES,
    TINTA_SECUNDARIA,
    cabecalho,
    estilizar_linhas,
    imprimir_tabela,
    salvar_figura,
    salvar_tabela,
)

from dados import GPTDatasetV1, carregar_texto, numero_de_amostras
from tokenizacao import TokenizadorBPE

CONTEXTOS = [4, 16, 64, 256, 1024]
FRACOES_DE_STRIDE = [("1 token", None), ("¼ contexto", 0.25), ("½ contexto", 0.5),
                     ("= contexto", 1.0), ("2 × contexto", 2.0)]


def executar() -> list[dict]:
    cabecalho("Experimento 06 — relação entre contexto e quantidade de amostras")
    texto = carregar_texto()
    bpe = TokenizadorBPE()
    n = len(bpe.encode(texto))
    print(f"  corpus={n} tokens BPE")

    linhas: list[dict] = []
    for contexto in CONTEXTOS:
        for rotulo, fracao in FRACOES_DE_STRIDE:
            stride = 1 if fracao is None else max(1, int(contexto * fracao))
            dataset = GPTDatasetV1(texto, bpe, contexto, stride)
            amostras = len(dataset)

            # Posições do corpus que aparecem como alvo em alguma amostra.
            inicios = torch.arange(0, n - contexto, stride)
            alvos = (inicios[:, None] + torch.arange(1, contexto + 1)).flatten()
            alvos_distintos = int(torch.unique(alvos).numel())

            linhas.append(
                {
                    "max_length": contexto,
                    "stride": stride,
                    "stride_relativo": rotulo,
                    "amostras": amostras,
                    "formula_confere": amostras == numero_de_amostras(n, contexto, stride),
                    "sobreposicao_pct": 100 * max(0.0, 1 - stride / contexto),
                    "reuso_medio_dos_alvos": amostras * contexto / max(alvos_distintos, 1),
                    "tokens_nunca_alvo_pct": 100 * (n - 1 - alvos_distintos) / (n - 1),
                }
            )

    imprimir_tabela(linhas)
    salvar_tabela(linhas, "exp06-contexto-e-amostras")
    _graficos(linhas)
    return linhas


def _graficos(linhas: list[dict]) -> None:
    fig, eixos = plt.subplots(1, 2, figsize=(10.5, 3.9))
    # Quinta série em cinza tracejado: a paleta categórica tem quatro cores.
    cores = CATEGORICA + [TINTA_SECUNDARIA]
    marcadores = MARCADORES + ["v"]

    for i, (rotulo, _) in enumerate(FRACOES_DE_STRIDE):
        serie = [l for l in linhas if l["stride_relativo"] == rotulo]
        estilo = {"color": cores[i], "marker": marcadores[i],
                  "linestyle": "--" if i == 4 else "-", "label": f"stride {rotulo}"}
        eixos[0].plot([l["max_length"] for l in serie], [l["amostras"] for l in serie], **estilo)
        eixos[1].plot(
            [l["max_length"] for l in serie], [l["reuso_medio_dos_alvos"] for l in serie], **estilo
        )

    for ax in eixos:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
    eixos[0].legend(fontsize=7)
    estilizar_linhas(eixos[0], "Amostras produzidas", "max_length (tokens)", "amostras")
    estilizar_linhas(
        eixos[1], "Quantas vezes cada token é alvo por época",
        "max_length (tokens)", "reuso médio dos alvos",
    )

    fig.suptitle("Experimento 06 — contexto, stride e quantidade de amostras", y=1.04)
    fig.tight_layout()
    salvar_figura(fig, "exp06-contexto-e-amostras")


if __name__ == "__main__":
    executar()
