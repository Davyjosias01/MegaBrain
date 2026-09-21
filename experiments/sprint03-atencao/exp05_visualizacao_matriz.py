"""Experimento 05 — visualização das matrizes de atenção.

A matriz de atenção é o objeto que o Capítulo 3 constrói passo a passo, e é
também o único lugar onde o comportamento do mecanismo pode ser *visto*. Em
todos os mapas de calor deste experimento:

- cada **linha** é uma *query* — o token que está olhando;
- cada **coluna** é uma *key* — o token que está sendo olhado;
- a **cor** é o peso de atenção, de claro (pouca) a escuro (muita);
- **cada linha soma 1**, porque é a saída de um softmax.

São geradas três figuras:

``exp05-matriz-livro``
    Os três mecanismos aplicados ao exemplo de seis tokens do livro
    (*"Your journey starts with one step"*), com os valores anotados em cada
    célula. Permite conferir número a número contra as páginas 77, 92 e 95.

``exp05-matriz-corpus``
    Os mesmos mecanismos aplicados a uma sequência real de 24 tokens de
    ``the-verdict.txt``, em escala maior.

``exp05-matriz-por-cabeca``
    As quatro cabeças de uma ``MultiHeadAttention`` sobre a mesma sequência,
    lado a lado, na mesma escala de cor.

**Ressalva.** Os pesos mostrados vêm de matrizes ``W`` aleatórias: são o
comportamento do mecanismo *na inicialização*, não padrões linguísticos
aprendidos. Padrões interpretáveis só aparecem depois do treinamento (Sprint 5).
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from comum import (
    ENTRADAS_LIVRO,
    TOKENS_LIVRO,
    cabecalho,
    desenhar_matriz_atencao,
    salvar_figura,
    semear,
)

from attention import (
    CausalAttention,
    MultiHeadAttention,
    SelfAttentionV2,
    atencao_simples,
)
from entradas_atencao import lote_de_embeddings

D_MODELO = 64
NUM_HEADS = 4
NUM_TOKENS_CORPUS = 24


def executar() -> None:
    cabecalho("Experimento 05 — visualização das matrizes de atenção")

    _figura_livro()
    _figura_corpus()
    _figura_por_cabeca()


def _figura_livro() -> None:
    """Exemplo de seis tokens do livro, com os valores anotados."""
    x = ENTRADAS_LIVRO
    lote = x.unsqueeze(0)

    _, pesos_simples = atencao_simples(x)

    semear(789)  # mesma semente da seção 3.4.2, p. 89
    sa = SelfAttentionV2(3, 2).eval()
    _, pesos_self = sa(x, return_attn_weights=True)

    semear(789)
    ca = CausalAttention(3, 2, 6, 0.0).eval()
    _, pesos_causal = ca(lote, return_attn_weights=True)

    fig, eixos = plt.subplots(1, 3, figsize=(13.5, 4.3))
    desenhar_matriz_atencao(
        eixos[0],
        pesos_simples,
        "atencao_simples\n(sem pesos treináveis, seção 3.3)",
        TOKENS_LIVRO,
        anotar=True,
        vmax=1.0,
    )
    desenhar_matriz_atencao(
        eixos[1],
        pesos_self,
        "SelfAttentionV2\n(scaled dot-product, sem máscara)",
        TOKENS_LIVRO,
        anotar=True,
        vmax=1.0,
    )
    desenhar_matriz_atencao(
        eixos[2],
        pesos_causal[0],
        "CausalAttention\n(cinza = escondido pela máscara)",
        TOKENS_LIVRO,
        marcar_mascarados=True,
        anotar=True,
        vmax=1.0,
    )
    fig.suptitle(
        'Experimento 05 — "Your journey starts with one step" '
        "(exemplo da seção 3.3.1)",
        y=1.02,
    )
    fig.tight_layout()
    salvar_figura(fig, "exp05-matriz-livro")


def _figura_corpus() -> None:
    """Mesma comparação sobre uma sequência real de the-verdict.txt."""
    dados = lote_de_embeddings(
        d_in=D_MODELO, max_length=NUM_TOKENS_CORPUS, batch_size=1
    )
    x = dados["embeddings"]
    rotulos = dados["tokens"]

    _, pesos_simples = atencao_simples(x)

    semear()
    sa = SelfAttentionV2(D_MODELO, D_MODELO).eval()
    _, pesos_self = sa(x, return_attn_weights=True)

    semear()
    ca = CausalAttention(D_MODELO, D_MODELO, NUM_TOKENS_CORPUS, 0.0).eval()
    _, pesos_causal = ca(x, return_attn_weights=True)

    fig, eixos = plt.subplots(1, 3, figsize=(14, 4.8))
    desenhar_matriz_atencao(
        eixos[0], pesos_simples[0], "atencao_simples", rotulos
    )
    desenhar_matriz_atencao(eixos[1], pesos_self[0], "SelfAttentionV2", rotulos)
    desenhar_matriz_atencao(
        eixos[2],
        pesos_causal[0],
        "CausalAttention",
        rotulos,
        marcar_mascarados=True,
    )
    fig.suptitle(
        f"Experimento 05 — {NUM_TOKENS_CORPUS} tokens de the-verdict.txt "
        f"(d_in = {D_MODELO}, pesos na inicialização)",
        y=1.02,
    )
    fig.tight_layout()
    salvar_figura(fig, "exp05-matriz-corpus")


def _figura_por_cabeca() -> None:
    """Uma matriz por cabeça, todas na mesma escala de cor."""
    dados = lote_de_embeddings(
        d_in=D_MODELO, max_length=NUM_TOKENS_CORPUS, batch_size=1
    )
    x = dados["embeddings"]
    rotulos = dados["tokens"]

    semear()
    mha = MultiHeadAttention(
        D_MODELO, D_MODELO, NUM_TOKENS_CORPUS, 0.0, NUM_HEADS
    ).eval()
    _, pesos = mha(x, return_attn_weights=True)
    pesos = pesos[0].detach()

    vmax = float(pesos.max())
    fig, eixos = plt.subplots(1, NUM_HEADS, figsize=(4.4 * NUM_HEADS, 4.8))
    for k in range(NUM_HEADS):
        desenhar_matriz_atencao(
            eixos[k],
            pesos[k],
            f"cabeça {k}  (head_dim = {mha.head_dim})",
            rotulos,
            marcar_mascarados=True,
            vmax=vmax,
        )
    fig.suptitle(
        f"Experimento 05 — as {NUM_HEADS} cabeças de uma MultiHeadAttention "
        "sobre a mesma sequência",
        y=1.02,
    )
    fig.tight_layout()
    salvar_figura(fig, "exp05-matriz-por-cabeca")


if __name__ == "__main__":
    executar()
