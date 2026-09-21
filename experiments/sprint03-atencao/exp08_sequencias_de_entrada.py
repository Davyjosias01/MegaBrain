"""Experimento 08 — diferentes sequências de entrada.

Duas variações independentes, sobre o mesmo módulo de atenção:

**(a) Conteúdo.** Seis sequências de naturezas diferentes passam pelo mesmo
``MultiHeadAttention``, com ``d_in = d_out = 64`` e 4 cabeças:

- o exemplo do livro;
- dois trechos reais de ``the-verdict.txt``;
- uma sequência de um único token repetido;
- uma sequência de tokens distintos e sem relação entre si;
- uma sequência com pontuação pesada.

A sequência de token repetido é o caso de controle mais informativo: como todos
os tokens têm o mesmo embedding de conteúdo, tudo o que resta para diferenciá-los
é o embedding posicional da Sprint 2. Ela mostra que o mecanismo de atenção, por
si só, é invariante a permutações — a ordem só entra pelo vetor de posição.

**(b) Comprimento.** A mesma configuração com ``num_tokens`` em {8, 16, 32, 64,
128, 256}. O custo da atenção cresce com ``num_tokens²``, porque a matriz de
scores tem esse tamanho: é a razão pela qual o comprimento de contexto de um LLM
é caro de aumentar. O experimento mede o tempo e o número de elementos da matriz.

**Métricas por sequência:** entropia normalizada (a atenção está espalhada ou
concentrada?), fração da atenção dirigida ao primeiro token e atenção média ao
próprio token (diagonal).
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from comum import (
    CATEGORICA,
    MARCADORES,
    cabecalho,
    cronometrar,
    desenhar_matriz_atencao,
    entropia_normalizada,
    estilizar_linhas,
    imprimir_tabela,
    massa_no_primeiro_token,
    salvar_figura,
    salvar_tabela,
    semear,
)

from attention import MultiHeadAttention
from entradas_atencao import carregar_texto, embutir_sequencia, lote_de_embeddings

D_MODELO = 64
NUM_HEADS = 4
COMPRIMENTOS = [8, 16, 32, 64, 128, 256]
BATCH = 8


TOKEN_REPETIDO = "the " * 20


def _sequencias() -> list[tuple[str, str, bool]]:
    """``(nome, texto, usar_embedding_posicional)``."""
    corpus = carregar_texto()
    return [
        ("exemplo do livro", "Your journey starts with one step.", True),
        ("narrativa (corpus)", " ".join(corpus.split()[:20]), True),
        ("diálogo (corpus)", " ".join(corpus.split()[300:320]), True),
        ("token repetido", TOKEN_REPETIDO, True),
        # Controle: mesmo texto, sem o embedding posicional da Sprint 2. Todos os
        # tokens ficam com vetores idênticos e a atenção não tem como
        # distingui-los — cada linha vira uniforme sobre o próprio passado.
        ("token repetido (sem posicional)", TOKEN_REPETIDO, False),
        (
            "tokens sem relação",
            "quartz velvet igloo thunder cactus opal mango zebra fjord lantern "
            "amber glacier",
            True,
        ),
        (
            "pontuação pesada",
            'Yes -- no; "perhaps," he said. And then: nothing, nothing at all!',
            True,
        ),
    ]


def executar() -> tuple[list[dict], list[dict]]:
    cabecalho("Experimento 08 — diferentes sequências de entrada")

    linhas_a = _variacao_de_conteudo()
    linhas_b = _variacao_de_comprimento()
    return linhas_a, linhas_b


def _variacao_de_conteudo() -> list[dict]:
    print(f"\n  (a) conteúdo — d_in = d_out = {D_MODELO}, num_heads = {NUM_HEADS}")

    linhas: list[dict] = []
    guardados: list[tuple[str, object, list[str]]] = []

    for nome, texto, usar_posicional in _sequencias():
        dados = embutir_sequencia(
            texto, d_in=D_MODELO, usar_posicional=usar_posicional
        )
        x = dados["embeddings"]
        n = x.shape[1]

        semear()
        mha = MultiHeadAttention(D_MODELO, D_MODELO, n, 0.0, NUM_HEADS).eval()
        _, pesos = mha(x, return_attn_weights=True)
        p = pesos[0].detach()

        linhas.append(
            {
                "sequencia": nome,
                "num_tokens": n,
                "entropia_normalizada": entropia_normalizada(pesos),
                "atencao_ao_primeiro_token": massa_no_primeiro_token(pesos),
                "atencao_ao_proprio_token": float(
                    p.diagonal(dim1=-2, dim2=-1)[:, 1:].mean()
                ),
                # A linha 0 vale sempre 1 sob máscara causal e é excluída.
                "peso_maximo_fora_da_linha_0": float(p[:, 1:, :].max()),
            }
        )
        guardados.append((nome, p[0], dados["tokens"]))

    imprimir_tabela(linhas)
    salvar_tabela(linhas, "exp08-sequencias-conteudo")
    _grafico_conteudo(linhas, guardados)
    return linhas


def _variacao_de_comprimento() -> list[dict]:
    print(f"\n  (b) comprimento — lote de {BATCH} sequências de the-verdict.txt")

    linhas: list[dict] = []
    for n in COMPRIMENTOS:
        dados = lote_de_embeddings(d_in=D_MODELO, max_length=n, batch_size=BATCH)
        x = dados["embeddings"]

        semear()
        mha = MultiHeadAttention(D_MODELO, D_MODELO, n, 0.0, NUM_HEADS).eval()
        tempo = cronometrar(lambda: mha(x))
        _, pesos = mha(x, return_attn_weights=True)

        linhas.append(
            {
                "num_tokens": n,
                "elementos_matriz_atencao": BATCH * NUM_HEADS * n * n,
                "tempo_ms": tempo,
                "tempo_por_token_us": tempo * 1000 / (BATCH * n),
                "entropia_normalizada": entropia_normalizada(pesos),
                "atencao_ao_primeiro_token": massa_no_primeiro_token(pesos),
            }
        )

    imprimir_tabela(linhas)
    salvar_tabela(linhas, "exp08-sequencias-comprimento")
    _grafico_comprimento(linhas)
    return linhas


def _grafico_conteudo(linhas: list[dict], guardados) -> None:
    fig, eixos = plt.subplots(1, 2, figsize=(12.5, 4.0))

    nomes = [l["sequencia"] for l in linhas]
    posicoes = list(range(len(nomes)))
    largura = 0.38

    # Duas métricas na mesma unidade (fração / índice em [0,1]): barras
    # agrupadas com 2px de folga entre elas, um eixo só.
    eixos[0].barh(
        [p - largura / 2 - 0.012 for p in posicoes],
        [l["entropia_normalizada"] for l in linhas],
        height=largura,
        color=CATEGORICA[0],
        label="entropia normalizada",
    )
    eixos[0].barh(
        [p + largura / 2 + 0.012 for p in posicoes],
        [l["atencao_ao_primeiro_token"] for l in linhas],
        height=largura,
        color=CATEGORICA[1],
        label="atenção ao primeiro token",
    )
    eixos[0].set_yticks(posicoes, nomes, fontsize=8)
    eixos[0].invert_yaxis()
    eixos[0].grid(axis="y", visible=False)
    # Folga à direita para a legenda não cobrir nenhuma barra.
    eixos[0].set_xlim(0, 1.34)
    eixos[0].legend(loc="center right", fontsize=8)
    eixos[0].set_title("Como a atenção se distribui em cada sequência", pad=8)
    eixos[0].set_xlabel("fração (0 a 1)")

    # O par "token repetido" com e sem embedding posicional é o resultado mais
    # informativo do experimento, e ganha o segundo painel.
    nome, pesos, rotulos = next(
        g for g in guardados if g[0] == "token repetido (sem posicional)"
    )
    desenhar_matriz_atencao(
        eixos[1],
        pesos,
        "token repetido, sem embedding posicional — cabeça 0\n"
        "(vetores idênticos: cada linha é uniforme sobre o próprio passado)",
        rotulos,
        marcar_mascarados=True,
    )

    fig.suptitle("Experimento 08 (a) — efeito do conteúdo da sequência", y=1.03)
    fig.tight_layout()
    salvar_figura(fig, "exp08-sequencias-conteudo")


def _grafico_comprimento(linhas: list[dict]) -> None:
    n = [l["num_tokens"] for l in linhas]

    fig, eixos = plt.subplots(1, 3, figsize=(11.5, 3.6))

    eixos[0].plot(
        n,
        [l["elementos_matriz_atencao"] for l in linhas],
        color=CATEGORICA[0],
        marker=MARCADORES[0],
    )
    eixos[0].set_xscale("log", base=2)
    eixos[0].set_yscale("log")
    estilizar_linhas(
        eixos[0],
        "Tamanho da matriz de atenção",
        "tokens por sequência",
        "elementos (escala log)",
    )

    eixos[1].plot(
        n, [l["tempo_ms"] for l in linhas], color=CATEGORICA[1], marker=MARCADORES[1]
    )
    eixos[1].set_xscale("log", base=2)
    eixos[1].set_yscale("log")
    estilizar_linhas(
        eixos[1], "Tempo da passagem direta", "tokens por sequência", "milissegundos"
    )

    eixos[2].plot(
        n,
        [l["entropia_normalizada"] for l in linhas],
        color=CATEGORICA[2],
        marker=MARCADORES[2],
    )
    eixos[2].set_xscale("log", base=2)
    eixos[2].set_ylim(0, 1.05)
    estilizar_linhas(
        eixos[2], "Dispersão da atenção", "tokens por sequência", "entropia normalizada"
    )

    fig.suptitle(
        "Experimento 08 (b) — o custo da atenção cresce com o quadrado do "
        "comprimento da sequência",
        y=1.04,
    )
    fig.tight_layout()
    salvar_figura(fig, "exp08-sequencias-comprimento")


if __name__ == "__main__":
    executar()
