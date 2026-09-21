"""Experimento 03 — diferentes dimensões de cabeça, com ``num_heads`` fixo.

**O que varia:** ``head_dim`` em {4, 8, 16, 32, 64, 128}, com ``d_out =
num_heads · head_dim``.
**O que fica fixo:** ``num_heads = 4``, ``d_in = 128``, 128 tokens, lote de 8.

**Pergunta:** qual é o papel isolado de ``head_dim``?

O experimento 02 repartiu um ``d_out`` fixo entre mais cabeças; aqui o caminho é
o oposto: o número de cabeças é constante e cada uma delas fica maior. ``d_out``
cresce junto, e com ele a contagem de parâmetros.

``head_dim`` é o ``d_k`` da fórmula ``softmax(Q·Kᵀ / √d_k)·V``: é nele que os
produtos escalares entre query e key acontecem e é ele que aparece no divisor da
escala. Duas leituras interessam:

1. uma cabeça estreita compara queries e keys num espaço de pouquíssimas
   dimensões, o que limita quantas relações distintas ela consegue representar;
2. a entropia da atenção deve permanecer estável apesar do crescimento de
   ``d_k``, porque o divisor ``√d_k`` acompanha — de novo, o argumento da p. 84.

A comparação entre este experimento e o 02 isola os dois efeitos: lá muda como o
orçamento é dividido; aqui muda o tamanho do orçamento.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from comum import (
    CATEGORICA,
    MARCADORES,
    cabecalho,
    contar_parametros,
    cronometrar,
    dissimilaridade_entre_cabecas,
    entropia_normalizada,
    estilizar_linhas,
    imprimir_tabela,
    salvar_figura,
    salvar_tabela,
    semear,
)

from attention import MultiHeadAttention
from entradas_atencao import lote_de_embeddings

D_ENTRADA = 128
NUM_HEADS = 4
DIMENSOES_DE_HEAD = [4, 8, 16, 32, 64, 128]
NUM_TOKENS = 128
BATCH = 8


def executar() -> list[dict]:
    cabecalho("Experimento 03 — diferentes dimensões de cabeça (num_heads fixo)")
    print(
        f"  d_in={D_ENTRADA}  num_heads={NUM_HEADS}  num_tokens={NUM_TOKENS}  "
        f"batch={BATCH}"
    )

    dados = lote_de_embeddings(
        d_in=D_ENTRADA, max_length=NUM_TOKENS, batch_size=BATCH
    )
    x = dados["embeddings"]

    linhas: list[dict] = []
    for head_dim in DIMENSOES_DE_HEAD:
        d_out = NUM_HEADS * head_dim
        semear()
        mha = MultiHeadAttention(
            d_in=D_ENTRADA,
            d_out=d_out,
            context_length=NUM_TOKENS,
            dropout=0.0,
            num_heads=NUM_HEADS,
        ).eval()

        tempo = cronometrar(lambda: mha(x))
        _, pesos = mha(x, return_attn_weights=True)

        linhas.append(
            {
                "head_dim": head_dim,
                "d_out": d_out,
                "parametros": contar_parametros(mha),
                "tempo_ms": tempo,
                "entropia_normalizada": entropia_normalizada(pesos),
                "dissimilaridade_entre_cabecas": dissimilaridade_entre_cabecas(pesos),
            }
        )

    imprimir_tabela(linhas)
    salvar_tabela(linhas, "exp03-dimensao-de-head")
    _graficos(linhas)
    return linhas


def _graficos(linhas: list[dict]) -> None:
    hd = [l["head_dim"] for l in linhas]

    fig, eixos = plt.subplots(1, 3, figsize=(11.5, 3.6))

    eixos[0].plot(
        hd, [l["parametros"] for l in linhas], color=CATEGORICA[0], marker=MARCADORES[0]
    )
    eixos[0].set_xscale("log", base=2)
    eixos[0].set_yscale("log")
    estilizar_linhas(eixos[0], "Parâmetros do módulo", "head_dim", "parâmetros")

    eixos[1].plot(
        hd, [l["tempo_ms"] for l in linhas], color=CATEGORICA[1], marker=MARCADORES[1]
    )
    eixos[1].set_xscale("log", base=2)
    estilizar_linhas(eixos[1], "Tempo da passagem direta", "head_dim", "milissegundos")

    eixos[2].plot(
        hd,
        [l["entropia_normalizada"] for l in linhas],
        color=CATEGORICA[2],
        marker=MARCADORES[2],
    )
    eixos[2].set_xscale("log", base=2)
    eixos[2].set_ylim(0, 1.05)
    estilizar_linhas(
        eixos[2], "Dispersão da atenção", "head_dim", "entropia normalizada"
    )

    fig.suptitle(
        f"Experimento 03 — dimensão de cabeça com num_heads = {NUM_HEADS} fixo "
        f"(d_out = {NUM_HEADS} × head_dim)",
        y=1.04,
    )
    fig.tight_layout()
    salvar_figura(fig, "exp03-dimensao-de-head")


if __name__ == "__main__":
    executar()
