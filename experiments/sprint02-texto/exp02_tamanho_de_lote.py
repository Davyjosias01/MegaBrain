"""Experimento 02 — diferentes tamanhos de lote.

**O que varia:** ``batch_size`` em {1, 2, 4, 8, 16, 32, 64}.
**O que fica fixo:** ``max_length = stride = 64`` (80 amostras do corpus),
``d_emb = 256``, ``drop_last=True``, semente 123.

**Pergunta:** o que o tamanho do lote muda no carregamento dos dados?

O lote não altera *quais* amostras existem — isso é decidido pela janela
deslizante —, mas altera como elas são entregues ao modelo:

1. **Lotes por época.** ``amostras // batch_size``, com ``drop_last``.
2. **Amostras descartadas.** ``amostras % batch_size``. Com um corpus pequeno,
   lotes grandes desperdiçam uma fração relevante dos dados.
3. **Tempo por amostra.** Processar várias amostras em uma única operação
   tensorial amortiza o custo fixo de cada chamada; espera-se que o tempo por
   amostra caia com o lote, até saturar.
4. **Memória por lote.** Cresce linearmente com ``batch_size``. Na Sprint 5,
   somada às ativações e aos gradientes, é ela que limita o lote máximo.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from comum import (
    CATEGORICA,
    MARCADORES,
    bytes_de,
    cabecalho,
    cronometrar,
    estilizar_linhas,
    imprimir_tabela,
    salvar_figura,
    salvar_tabela,
    semear,
)

from dados import carregar_texto, criar_dataloader
from embeddings import EmbeddingDeEntrada
from tokenizacao import TokenizadorBPE

LOTES = [1, 2, 4, 8, 16, 32, 64]
CONTEXTO = 64
D_EMB = 256


def executar() -> list[dict]:
    cabecalho("Experimento 02 — diferentes tamanhos de lote")
    texto = carregar_texto()
    bpe = TokenizadorBPE()

    semear()
    camada = EmbeddingDeEntrada(bpe.tamanho_vocabulario, D_EMB, CONTEXTO).eval()

    linhas: list[dict] = []
    for lote in LOTES:
        carregador = criar_dataloader(
            texto, batch_size=lote, max_length=CONTEXTO, stride=CONTEXTO, tokenizador=bpe
        )
        amostras = len(carregador.dataset)
        usadas = len(carregador) * lote

        def uma_epoca() -> None:
            for entradas, _ in carregador:
                camada(entradas)

        tempo = cronometrar(uma_epoca, repeticoes=10, aquecimento=2)
        linhas.append(
            {
                "batch_size": lote,
                "amostras": amostras,
                "lotes_por_epoca": len(carregador),
                "amostras_descartadas": amostras - usadas,
                "descartadas_pct": 100 * (amostras - usadas) / amostras,
                "bytes_lote_embeddings": bytes_de((lote, CONTEXTO, D_EMB), torch.float32),
                "tempo_epoca_ms": tempo,
                "tempo_por_amostra_us": 1000 * tempo / usadas,
            }
        )

    imprimir_tabela(linhas)
    salvar_tabela(linhas, "exp02-tamanho-de-lote")
    _graficos(linhas)
    return linhas


def _graficos(linhas: list[dict]) -> None:
    lote = [l["batch_size"] for l in linhas]
    fig, eixos = plt.subplots(1, 3, figsize=(12, 3.7))

    eixos[0].plot(
        lote, [l["lotes_por_epoca"] for l in linhas],
        color=CATEGORICA[0], marker=MARCADORES[0],
    )
    eixos[0].set_xscale("log", base=2)
    eixos[0].set_yscale("log")
    estilizar_linhas(eixos[0], "Lotes por época", "batch_size", "lotes")

    eixos[1].bar(
        [str(b) for b in lote], [l["descartadas_pct"] for l in linhas], color=CATEGORICA[1]
    )
    eixos[1].grid(axis="x", visible=False)
    estilizar_linhas(
        eixos[1], "Amostras descartadas por drop_last", "batch_size", "% das amostras"
    )

    eixos[2].plot(
        lote, [l["tempo_por_amostra_us"] for l in linhas],
        color=CATEGORICA[2], marker=MARCADORES[2],
    )
    eixos[2].set_xscale("log", base=2)
    eixos[2].set_yscale("log")
    estilizar_linhas(
        eixos[2], "Tempo por amostra\n(DataLoader + embedding)", "batch_size", "microssegundos"
    )

    fig.suptitle(
        f"Experimento 02 — efeito do tamanho do lote (max_length {CONTEXTO}, "
        f"{linhas[0]['amostras']} amostras)",
        y=1.04,
    )
    fig.tight_layout()
    salvar_figura(fig, "exp02-tamanho-de-lote")


if __name__ == "__main__":
    executar()
