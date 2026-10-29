"""Experimento 04 — dimensão da rede FeedForward.

**O que varia:** ``ff_dim`` = ``m · emb_dim``, com ``m`` em {1, 2, 4, 8}
(o GPT-2 usa ``m = 4``).
**O que fica fixo:** ``emb_dim = 256``, 4 cabeças, 4 blocos, contexto 256,
lote real ``(8, 128)``, semente 123, ``eval()``.

**Pergunta:** qual o peso da feed forward dentro do bloco e o que acontece
quando ela é alargada?

Por bloco, a atenção tem ``4d² + d`` parâmetros (Q, K, V e a projeção de saída)
e a feed forward ``2 · d · ff_dim + ff_dim + d``. A razão entre as duas é
``≈ ff_dim / 2d``: com o fator 4 do GPT-2, a feed forward tem o dobro de
parâmetros da atenção (exercício 4.1). O experimento confirma a razão e mede
como ela aparece no tempo.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from comum import (
    CATEGORICA,
    cabecalho,
    cronometrar,
    imprimir_tabela,
    lote_do_corpus,
    novo_modelo,
    salvar_figura,
    salvar_tabela,
)

from gpt import contar_parametros, parametros_por_componente

MULTIPLICADORES = [1, 2, 4, 8]
EMB_DIM = 256


def executar() -> list[dict]:
    cabecalho("Experimento 04 — dimensão da rede FeedForward")
    entradas, _ = lote_do_corpus()

    linhas: list[dict] = []
    for m in MULTIPLICADORES:
        modelo = novo_modelo(ff_dim=m * EMB_DIM)
        partes = parametros_por_componente(modelo)
        atencao = partes["atenção (todos os blocos)"]
        feed_forward = partes["feed forward (todos os blocos)"]
        bloco = modelo.trf_blocks[0]
        with torch.no_grad():
            x = modelo.drop_emb(modelo.embedding(entradas))
            normalizado = bloco.norm1(x)

        linhas.append(
            {
                "ff_dim": m * EMB_DIM,
                "multiplicador": m,
                "parametros": contar_parametros(modelo),
                "parametros_atencao": atencao,
                "parametros_feedforward": feed_forward,
                "razao_ff_atencao": feed_forward / atencao,
                "tempo_atencao_ms": cronometrar(lambda: bloco.att(normalizado), repeticoes=10),
                "tempo_feedforward_ms": cronometrar(lambda: bloco.ff(normalizado), repeticoes=10),
                "tempo_blocos_ms": cronometrar(lambda: modelo.trf_blocks(x)),
            }
        )

    imprimir_tabela(linhas)
    salvar_tabela(linhas, "exp04-dimensao-feedforward")
    _graficos(linhas)
    return linhas


def _graficos(linhas: list[dict]) -> None:
    rotulos = [f"{l['ff_dim']}\n({l['multiplicador']}×)" for l in linhas]
    pos = np.arange(len(linhas))
    largura = 0.38
    fig, eixos = plt.subplots(1, 2, figsize=(10.5, 3.9))

    eixos[0].bar(pos - largura / 2, [l["parametros_atencao"] / 1e6 for l in linhas],
                 width=largura, color=CATEGORICA[0], label="atenção")
    eixos[0].bar(pos + largura / 2, [l["parametros_feedforward"] / 1e6 for l in linhas],
                 width=largura, color=CATEGORICA[1], label="feed forward")
    eixos[0].set_xticks(pos, rotulos)
    eixos[0].grid(axis="x", visible=False)
    eixos[0].legend(fontsize=7)
    eixos[0].set_title("Parâmetros nos 4 blocos", pad=8)
    eixos[0].set_xlabel("ff_dim")
    eixos[0].set_ylabel("milhões")

    eixos[1].bar(pos - largura / 2, [l["tempo_atencao_ms"] for l in linhas],
                 width=largura, color=CATEGORICA[0], label="atenção")
    eixos[1].bar(pos + largura / 2, [l["tempo_feedforward_ms"] for l in linhas],
                 width=largura, color=CATEGORICA[1], label="feed forward")
    eixos[1].set_xticks(pos, rotulos)
    eixos[1].grid(axis="x", visible=False)
    eixos[1].legend(fontsize=7)
    eixos[1].set_title("Tempo de cada subcamada em um bloco\n(lote 8 × 128 tokens)", pad=8)
    eixos[1].set_xlabel("ff_dim")
    eixos[1].set_ylabel("ms")

    fig.suptitle(f"Experimento 04 — efeito da dimensão da FeedForward (emb_dim {EMB_DIM})", y=1.04)
    fig.tight_layout()
    salvar_figura(fig, "exp04-dimensao-feedforward")


if __name__ == "__main__":
    executar()
