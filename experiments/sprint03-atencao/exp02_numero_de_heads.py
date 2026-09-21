"""Experimento 02 — diferentes números de cabeças, com ``d_out`` fixo.

**O que varia:** ``num_heads`` em {1, 2, 3, 4, 6, 8, 12, 16}.
**O que fica fixo:** ``d_in = d_out = 96``, 128 tokens, lote de 8, semente 123.

**Pergunta:** o que realmente muda quando se acrescenta cabeças sem mexer na
dimensão de saída?

Esta é a variação que o livro adota na ``MultiHeadAttention`` da listagem 3.5:
``head_dim = d_out // num_heads``, de modo que o orçamento de 96 dimensões é
*repartido* entre as cabeças em vez de multiplicado. A consequência é que a
contagem de parâmetros não muda — as mesmas matrizes ``W`` de ``96×96`` são
apenas lidas em blocos diferentes.

O que muda é o número de matrizes de atenção calculadas (``num_heads`` por
sequência, cada uma ``num_tokens × num_tokens``) e a dimensão em que cada uma
delas compara queries e keys. Medimos:

- parâmetros e tempo, para confirmar o custo;
- entropia normalizada, que reage à queda de ``head_dim``;
- dissimilaridade média entre as cabeças, para verificar se elas de fato olham a
  sequência de maneiras diferentes ou se são redundantes.

``d_out = 96`` foi escolhido por ser divisível por todos os valores testados.
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

D_MODELO = 96
NUMERO_DE_CABECAS = [1, 2, 3, 4, 6, 8, 12, 16]
NUM_TOKENS = 128
BATCH = 8


def executar() -> list[dict]:
    cabecalho("Experimento 02 — diferentes números de cabeças (d_out fixo)")
    print(f"  d_in = d_out = {D_MODELO}  num_tokens={NUM_TOKENS}  batch={BATCH}")

    dados = lote_de_embeddings(
        d_in=D_MODELO, max_length=NUM_TOKENS, batch_size=BATCH
    )
    x = dados["embeddings"]

    linhas: list[dict] = []
    for h in NUMERO_DE_CABECAS:
        semear()
        mha = MultiHeadAttention(
            d_in=D_MODELO,
            d_out=D_MODELO,
            context_length=NUM_TOKENS,
            dropout=0.0,
            num_heads=h,
        ).eval()

        tempo = cronometrar(lambda: mha(x))
        _, pesos = mha(x, return_attn_weights=True)

        linhas.append(
            {
                "num_heads": h,
                "head_dim": mha.head_dim,
                "parametros": contar_parametros(mha),
                "matrizes_atencao": BATCH * h,
                "tempo_ms": tempo,
                "entropia_normalizada": entropia_normalizada(pesos),
                "dissimilaridade_entre_cabecas": dissimilaridade_entre_cabecas(pesos),
            }
        )

    imprimir_tabela(linhas)
    salvar_tabela(linhas, "exp02-numero-de-heads")
    _graficos(linhas)
    return linhas


def _graficos(linhas: list[dict]) -> None:
    h = [l["num_heads"] for l in linhas]

    fig, eixos = plt.subplots(1, 3, figsize=(11.5, 3.6))

    eixos[0].plot(
        h, [l["tempo_ms"] for l in linhas], color=CATEGORICA[0], marker=MARCADORES[0]
    )
    estilizar_linhas(
        eixos[0], "Tempo da passagem direta", "número de cabeças", "milissegundos"
    )

    eixos[1].plot(
        h,
        [l["entropia_normalizada"] for l in linhas],
        color=CATEGORICA[1],
        marker=MARCADORES[1],
    )
    eixos[1].set_ylim(0, 1.05)
    estilizar_linhas(
        eixos[1], "Dispersão da atenção", "número de cabeças", "entropia normalizada"
    )

    # Com uma única cabeça não existe par a comparar; a série começa em 2.
    com_pares = [l for l in linhas if l["num_heads"] >= 2]
    eixos[2].plot(
        [l["num_heads"] for l in com_pares],
        [l["dissimilaridade_entre_cabecas"] for l in com_pares],
        color=CATEGORICA[2],
        marker=MARCADORES[2],
    )
    eixos[2].set_ylim(0, None)
    estilizar_linhas(
        eixos[2],
        "Especialização das cabeças\n(indefinida com uma só cabeça)",
        "número de cabeças",
        "distância L1 média entre pares",
    )

    # Anota o fato central: a contagem de parâmetros não muda.
    parametros = {l["parametros"] for l in linhas}
    nota = (
        f"parâmetros constantes em {next(iter(parametros)):,} "
        "para todas as configurações"
        if len(parametros) == 1
        else "parâmetros variam entre as configurações"
    )
    fig.suptitle(
        f"Experimento 02 — número de cabeças com d_out = {D_MODELO} fixo ({nota})",
        y=1.04,
    )
    fig.tight_layout()
    salvar_figura(fig, "exp02-numero-de-heads")


if __name__ == "__main__":
    executar()
