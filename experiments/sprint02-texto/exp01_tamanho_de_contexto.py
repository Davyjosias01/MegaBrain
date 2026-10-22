"""Experimento 01 — diferentes tamanhos de contexto.

**O que varia:** ``max_length`` em {4, 8, 16, …, 1024}, com ``stride = max_length``
(janelas encostadas, sem sobreposição).
**O que fica fixo:** corpus ``the-verdict.txt`` (5.145 tokens BPE), lote de 8,
``d_emb = 256``, semente 123.

**Pergunta:** o que acontece com o conjunto de treinamento e com a memória de
cada lote quando o contexto aumenta?

O tamanho do contexto é o número de tokens que o modelo enxerga de uma vez. Ele
aparece em três lugares do pipeline, cada um com uma curva de custo diferente:

1. **Número de amostras.** Sem sobreposição, o corpus é fatiado em pedaços de
   ``max_length`` tokens: dobrar o contexto divide as amostras pela metade.
2. **Memória do lote.** Os IDs ``(b, max_length)`` e os embeddings
   ``(b, max_length, d_emb)`` crescem linearmente com o contexto.
3. **Memória da atenção.** A matriz de atenção da Sprint 3 tem
   ``max_length²`` elementos por cabeça: cresce com o quadrado.

O experimento mede as três e registra também quanto do corpus uma época
realmente aproveita depois que o ``DataLoader`` descarta o lote incompleto.
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

from dados import carregar_texto, criar_dataloader, numero_de_amostras
from embeddings import EmbeddingDeEntrada
from tokenizacao import TokenizadorBPE

CONTEXTOS = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
BATCH = 8
D_EMB = 256


def executar() -> list[dict]:
    cabecalho("Experimento 01 — diferentes tamanhos de contexto")
    texto = carregar_texto()
    bpe = TokenizadorBPE()
    n = len(bpe.encode(texto))
    print(f"  corpus={n} tokens BPE  batch={BATCH}  d_emb={D_EMB}  stride=max_length")

    linhas: list[dict] = []
    for contexto in CONTEXTOS:
        carregador = criar_dataloader(
            texto, batch_size=BATCH, max_length=contexto, stride=contexto, tokenizador=bpe
        )
        semear()
        camada = EmbeddingDeEntrada(bpe.tamanho_vocabulario, D_EMB, contexto).eval()

        def uma_epoca() -> None:
            for entradas, _ in carregador:
                camada(entradas)

        previsoes = len(carregador) * BATCH * contexto
        linhas.append(
            {
                "max_length": contexto,
                "amostras": len(carregador.dataset),
                "amostras_stride_1": numero_de_amostras(n, contexto, 1),
                "lotes_por_epoca": len(carregador),
                "previsoes_por_epoca": previsoes,
                # Cada token, exceto o primeiro, pode ser alvo de uma previsão.
                "corpus_aproveitado_pct": 100 * previsoes / (n - 1),
                "bytes_lote_ids": bytes_de((BATCH, contexto), torch.int64),
                "bytes_lote_embeddings": bytes_de((BATCH, contexto, D_EMB), torch.float32),
                "bytes_matriz_atencao": bytes_de((BATCH, contexto, contexto), torch.float32),
                "tempo_epoca_ms": cronometrar(uma_epoca, repeticoes=5, aquecimento=1),
            }
        )

    imprimir_tabela(linhas)
    salvar_tabela(linhas, "exp01-tamanho-de-contexto")
    _graficos(linhas)
    return linhas


def _graficos(linhas: list[dict]) -> None:
    contexto = [l["max_length"] for l in linhas]
    fig, eixos = plt.subplots(1, 3, figsize=(12, 3.7))

    eixos[0].plot(
        contexto, [l["amostras_stride_1"] for l in linhas],
        color=CATEGORICA[1], marker=MARCADORES[1], label="stride = 1",
    )
    eixos[0].plot(
        contexto, [l["amostras"] for l in linhas],
        color=CATEGORICA[0], marker=MARCADORES[0], label="stride = max_length",
    )
    eixos[0].set_xscale("log", base=2)
    eixos[0].set_yscale("log")
    eixos[0].legend()
    estilizar_linhas(eixos[0], "Amostras de treinamento", "max_length (tokens)", "amostras")

    eixos[1].plot(
        contexto, [l["corpus_aproveitado_pct"] for l in linhas],
        color=CATEGORICA[2], marker=MARCADORES[2],
    )
    eixos[1].set_xscale("log", base=2)
    eixos[1].set_ylim(0, 105)
    estilizar_linhas(
        eixos[1], "Corpus aproveitado por época\n(lote 8, drop_last)",
        "max_length (tokens)", "% dos tokens usados como alvo",
    )

    series = [
        ("IDs (b, L) int64", "bytes_lote_ids"),
        ("embeddings (b, L, 256) float32", "bytes_lote_embeddings"),
        ("matriz de atenção (b, L, L)", "bytes_matriz_atencao"),
    ]
    for i, (rotulo, chave) in enumerate(series):
        eixos[2].plot(
            contexto, [l[chave] / 1024 for l in linhas],
            color=CATEGORICA[i], marker=MARCADORES[i], label=rotulo,
        )
    eixos[2].set_xscale("log", base=2)
    eixos[2].set_yscale("log")
    eixos[2].legend(fontsize=7)
    estilizar_linhas(eixos[2], "Memória por lote", "max_length (tokens)", "KiB")

    fig.suptitle(
        f"Experimento 01 — efeito do tamanho do contexto (lote {BATCH}, d_emb {D_EMB})",
        y=1.04,
    )
    fig.tight_layout()
    salvar_figura(fig, "exp01-tamanho-de-contexto")


if __name__ == "__main__":
    executar()
