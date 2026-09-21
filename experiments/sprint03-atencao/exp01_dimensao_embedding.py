"""Experimento 01 — diferentes dimensões de embedding.

**O que varia:** ``d_in = d_out`` em {16, 32, 64, 128, 256, 512}.
**O que fica fixo:** ``num_heads = 4``, 128 tokens por sequência, lote de 8,
entradas reais de ``the-verdict.txt``, semente 123.

**Pergunta:** como o custo do mecanismo e o formato da distribuição de atenção
respondem ao aumento da dimensão do embedding?

Duas grandezas devem ser separadas. Os parâmetros do módulo vivem nas quatro
projeções lineares (Q, K, V e a projeção de saída) e crescem com ``4·d²``. Já a
matriz de atenção tem sempre ``num_tokens²`` elementos, qualquer que seja ``d``.
São dois regimes de custo diferentes, e o experimento mede os dois.

A terceira grandeza medida é a entropia normalizada da atenção. Como o núcleo
divide os scores por ``√d_k`` (seção 3.4.1), a variância dos scores não deve
explodir com ``d`` — logo, a distribuição de atenção deve permanecer
aproximadamente igual. É a verificação empírica do argumento da página 84.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from comum import (
    CATEGORICA,
    MARCADORES,
    cabecalho,
    contar_parametros,
    cronometrar,
    entropia_normalizada,
    estilizar_linhas,
    imprimir_tabela,
    salvar_figura,
    salvar_tabela,
    semear,
)

from attention import MultiHeadAttention
from entradas_atencao import lote_de_embeddings

DIMENSOES = [16, 32, 64, 128, 256, 512]
NUM_HEADS = 4
NUM_TOKENS = 128
BATCH = 8


def executar() -> list[dict]:
    cabecalho("Experimento 01 — diferentes dimensões de embedding")
    print(
        f"  num_heads={NUM_HEADS}  num_tokens={NUM_TOKENS}  batch={BATCH}  "
        f"corpus=the-verdict.txt"
    )

    linhas: list[dict] = []
    for d in DIMENSOES:
        dados = lote_de_embeddings(d_in=d, max_length=NUM_TOKENS, batch_size=BATCH)
        x = dados["embeddings"]

        semear()
        mha = MultiHeadAttention(
            d_in=d,
            d_out=d,
            context_length=NUM_TOKENS,
            dropout=0.0,
            num_heads=NUM_HEADS,
        ).eval()

        tempo = cronometrar(lambda: mha(x))
        _, pesos = mha(x, return_attn_weights=True)

        linhas.append(
            {
                "d_in = d_out": d,
                "head_dim": d // NUM_HEADS,
                "parametros": contar_parametros(mha),
                "elementos_matriz_atencao": BATCH * NUM_HEADS * NUM_TOKENS**2,
                "tempo_ms": tempo,
                "entropia_normalizada": entropia_normalizada(pesos),
            }
        )

    imprimir_tabela(linhas)
    salvar_tabela(linhas, "exp01-dimensao-embedding")
    _graficos(linhas)
    return linhas


def _graficos(linhas: list[dict]) -> None:
    d = [l["d_in = d_out"] for l in linhas]

    # Parâmetros e tempo têm unidades incomparáveis: dois painéis, nunca dois
    # eixos y no mesmo painel.
    fig, eixos = plt.subplots(1, 3, figsize=(11.5, 3.6))

    eixos[0].plot(
        d,
        [l["parametros"] for l in linhas],
        color=CATEGORICA[0],
        marker=MARCADORES[0],
    )
    eixos[0].set_xscale("log", base=2)
    eixos[0].set_yscale("log")
    estilizar_linhas(
        eixos[0], "Parâmetros do módulo", "dimensão do embedding", "parâmetros"
    )

    eixos[1].plot(
        d, [l["tempo_ms"] for l in linhas], color=CATEGORICA[1], marker=MARCADORES[1]
    )
    eixos[1].set_xscale("log", base=2)
    estilizar_linhas(
        eixos[1], "Tempo da passagem direta", "dimensão do embedding", "milissegundos"
    )

    eixos[2].plot(
        d,
        [l["entropia_normalizada"] for l in linhas],
        color=CATEGORICA[2],
        marker=MARCADORES[2],
    )
    eixos[2].set_xscale("log", base=2)
    eixos[2].set_ylim(0, 1.05)
    estilizar_linhas(
        eixos[2],
        "Dispersão da atenção",
        "dimensão do embedding",
        "entropia normalizada",
    )

    fig.suptitle(
        "Experimento 01 — efeito da dimensão do embedding "
        f"(num_heads={NUM_HEADS}, {NUM_TOKENS} tokens)",
        y=1.04,
    )
    fig.tight_layout()
    salvar_figura(fig, "exp01-dimensao-embedding")


if __name__ == "__main__":
    executar()
