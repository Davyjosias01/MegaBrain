"""Experimento 03 — diferentes dimensões de embedding.

**O que varia:** ``d_emb`` em {2, 8, 32, 64, 128, 256, 512, 768, 1024, 1280,
1600} — os quatro últimos são as dimensões dos GPT-2 small, medium, large e XL.
**O que fica fixo:** vocabulário do GPT-2 (50.257), ``context_length = 1024``,
lote ``(8, 256)`` tirado do corpus, semente 123.

**Pergunta:** qual o impacto da dimensão do embedding sobre as estruturas do
modelo?

Três efeitos são medidos:

1. **Parâmetros e memória das tabelas.** A tabela de tokens tem
   ``vocab_size · d_emb`` pesos e a posicional ``context_length · d_emb``:
   ambas lineares em ``d_emb``, mas a de tokens é 49 vezes maior.
2. **Tensor de entrada do modelo.** ``(b, num_tokens, d_emb)``: tudo o que vem
   depois na arquitetura (atenção, blocos Transformer) opera sobre ele.
3. **Espaço disponível.** Com a inicialização aleatória, vetores de tokens
   diferentes são quase ortogonais, e tanto mais quanto maior ``d_emb``: o
   desvio da similaridade de cosseno entre pares cai com ``1/√d_emb``. Em
   dimensão alta há "espaço" para que o treinamento posicione milhares de
   tokens sem que eles se sobreponham.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from comum import (
    CATEGORICA,
    MARCADORES,
    TINTA_SECUNDARIA,
    bytes_de,
    cabecalho,
    cronometrar,
    estilizar_linhas,
    imprimir_tabela,
    salvar_figura,
    salvar_tabela,
    semear,
    similaridade_cosseno,
)

from dados import carregar_texto, criar_dataloader
from embeddings import EmbeddingDeEntrada
from tokenizacao import TokenizadorBPE

DIMENSOES = [2, 8, 32, 64, 128, 256, 512, 768, 1024, 1280, 1600]
CONTEXT_LENGTH = 1024
BATCH = 8
NUM_TOKENS = 256
TOKENS_AMOSTRADOS = 2000  # pares de tokens distintos para a similaridade


def executar() -> list[dict]:
    cabecalho("Experimento 03 — diferentes dimensões de embedding")
    bpe = TokenizadorBPE()
    lote, _ = next(
        iter(
            criar_dataloader(
                carregar_texto(), batch_size=BATCH, max_length=NUM_TOKENS,
                stride=NUM_TOKENS, tokenizador=bpe,
            )
        )
    )
    print(
        f"  vocab_size={bpe.tamanho_vocabulario}  context_length={CONTEXT_LENGTH}  "
        f"lote={tuple(lote.shape)}"
    )

    linhas: list[dict] = []
    for d in DIMENSOES:
        semear()
        camada = EmbeddingDeEntrada(bpe.tamanho_vocabulario, d, CONTEXT_LENGTH).eval()
        with torch.no_grad():
            saida = camada(lote)

        # Similaridade entre tokens distintos escolhidos ao acaso.
        semear(7)
        ids = torch.randperm(bpe.tamanho_vocabulario)[:TOKENS_AMOSTRADOS]
        vetores = camada.token_emb.weight.detach()[ids]
        sim = similaridade_cosseno(vetores)
        fora_da_diagonal = sim[~torch.eye(len(ids), dtype=torch.bool)]

        p_token = camada.token_emb.weight.numel()
        p_pos = camada.pos_emb.weight.numel()
        linhas.append(
            {
                "d_emb": d,
                "parametros_token": p_token,
                "parametros_posicao": p_pos,
                "memoria_tabelas_mib": (p_token + p_pos) * 4 / 1024**2,
                "forma_saida": "x".join(map(str, saida.shape)),
                "bytes_saida": bytes_de(tuple(saida.shape), saida.dtype),
                "norma_media": float(vetores.norm(dim=-1).mean()),
                "cosseno_desvio": float(fora_da_diagonal.std()),
                "cosseno_max_abs": float(fora_da_diagonal.abs().max()),
                "tempo_ms": cronometrar(lambda: camada(lote), repeticoes=20, aquecimento=3),
            }
        )

    imprimir_tabela(linhas)
    salvar_tabela(linhas, "exp03-dimensao-embedding")
    _graficos(linhas)
    return linhas


def _graficos(linhas: list[dict]) -> None:
    d = [l["d_emb"] for l in linhas]
    fig, eixos = plt.subplots(1, 3, figsize=(12, 3.7))

    eixos[0].plot(
        d, [l["parametros_token"] for l in linhas],
        color=CATEGORICA[0], marker=MARCADORES[0], label="tabela de tokens (50.257 × d)",
    )
    eixos[0].plot(
        d, [l["parametros_posicao"] for l in linhas],
        color=CATEGORICA[1], marker=MARCADORES[1], label="tabela posicional (1.024 × d)",
    )
    eixos[0].set_xscale("log", base=2)
    eixos[0].set_yscale("log")
    eixos[0].legend(fontsize=7)
    estilizar_linhas(eixos[0], "Parâmetros das tabelas", "d_emb", "parâmetros")

    eixos[1].plot(
        d, [l["cosseno_desvio"] for l in linhas],
        color=CATEGORICA[2], marker=MARCADORES[2], label="medido",
    )
    eixos[1].plot(
        d, [x**-0.5 for x in d],
        color=TINTA_SECUNDARIA, linestyle="--", linewidth=1.2, label="1/√d",
    )
    eixos[1].set_xscale("log", base=2)
    eixos[1].set_yscale("log")
    eixos[1].legend(fontsize=7)
    estilizar_linhas(
        eixos[1], "Similaridade entre tokens distintos\n(inicialização)",
        "d_emb", "desvio do cosseno",
    )

    eixos[2].plot(
        d, [l["tempo_ms"] for l in linhas], color=CATEGORICA[3], marker=MARCADORES[3]
    )
    eixos[2].set_xscale("log", base=2)
    estilizar_linhas(
        eixos[2], f"Tempo do embedding\n(lote {BATCH} × {NUM_TOKENS})", "d_emb", "milissegundos"
    )

    fig.suptitle("Experimento 03 — efeito da dimensão do embedding", y=1.04)
    fig.tight_layout()
    salvar_figura(fig, "exp03-dimensao-embedding")


if __name__ == "__main__":
    executar()
