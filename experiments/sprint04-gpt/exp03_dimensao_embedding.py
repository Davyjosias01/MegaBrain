"""Experimento 03 — dimensão dos embeddings.

**O que varia:** ``emb_dim`` em {64, 128, 256, 384, 512, 768}.
**O que fica fixo:** 4 cabeças, 4 blocos, contexto 256, ``ff_dim = 4 · emb_dim``,
lote real ``(8, 128)``, semente 123, ``eval()``.

**Pergunta:** como a dimensão dos embeddings se distribui pelo modelo, e onde
está o custo?

``emb_dim`` aparece em todas as camadas, mas de duas formas diferentes:

- nas camadas ligadas ao vocabulário — tabela de tokens e cabeça de saída,
  ``2 · V · d`` parâmetros — o custo é **linear** em ``d``;
- dentro dos blocos — atenção ``≈ 4d²`` e feed forward ``≈ 8d²`` por bloco — o
  custo é **quadrático** em ``d``.

Com um vocabulário de 50.257 tokens, o termo linear domina enquanto ``d`` é
pequeno. O experimento mede em que ponto a balança muda e como isso aparece no
tempo de execução.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from comum import (
    CATEGORICA,
    MARCADORES,
    PROMPT,
    cabecalho,
    cronometrar,
    estilizar_linhas,
    imprimir_tabela,
    lote_do_corpus,
    ms_por_token_gerado,
    novo_modelo,
    salvar_figura,
    salvar_tabela,
)

from geracao import texto_para_ids
from gpt import contar_parametros, parametros_por_componente, tamanho_em_mb

DIMENSOES = [64, 128, 256, 384, 512, 768]


def executar() -> list[dict]:
    cabecalho("Experimento 03 — dimensão dos embeddings")
    entradas, _ = lote_do_corpus()
    prompt = texto_para_ids(PROMPT)

    linhas: list[dict] = []
    for d in DIMENSOES:
        modelo = novo_modelo(emb_dim=d)
        partes = parametros_por_componente(modelo)
        total = contar_parametros(modelo)
        vocabulario = partes["embedding de token"] + partes["cabeça de saída"]
        blocos = partes["atenção (todos os blocos)"] + partes["feed forward (todos os blocos)"]
        with torch.no_grad():
            embeddings = modelo.drop_emb(modelo.embedding(entradas))
            fluxo = modelo.trf_blocks(embeddings)

        linhas.append(
            {
                "emb_dim": d,
                "parametros": total,
                "memoria_mb": tamanho_em_mb(total),
                "camadas_do_vocabulario_pct": 100 * vocabulario / total,
                "blocos_pct": 100 * blocos / total,
                "tempo_blocos_ms": cronometrar(lambda: modelo.trf_blocks(embeddings)),
                "tempo_cabeca_ms": cronometrar(lambda: modelo.out_head(modelo.final_norm(fluxo))),
                "ms_por_token_gerado": ms_por_token_gerado(modelo, prompt),
            }
        )

    imprimir_tabela(linhas)
    salvar_tabela(linhas, "exp03-dimensao-embedding")
    _graficos(linhas)
    return linhas


def _graficos(linhas: list[dict]) -> None:
    d = [l["emb_dim"] for l in linhas]
    fig, eixos = plt.subplots(1, 3, figsize=(12, 3.7))

    eixos[0].plot(d, [l["parametros"] / 1e6 for l in linhas], color=CATEGORICA[0], marker=MARCADORES[0])
    estilizar_linhas(eixos[0], "Parâmetros do modelo", "emb_dim", "milhões")

    eixos[1].plot(d, [l["camadas_do_vocabulario_pct"] for l in linhas], color=CATEGORICA[1],
                  marker=MARCADORES[1], label="embedding de token + cabeça")
    eixos[1].plot(d, [l["blocos_pct"] for l in linhas], color=CATEGORICA[2],
                  marker=MARCADORES[2], label="atenção + feed forward")
    eixos[1].set_ylim(0, 100)
    eixos[1].legend(fontsize=7)
    estilizar_linhas(eixos[1], "Onde estão os parâmetros", "emb_dim", "% do total")

    eixos[2].plot(d, [l["tempo_blocos_ms"] for l in linhas], color=CATEGORICA[2],
                  marker=MARCADORES[2], label="4 blocos")
    eixos[2].plot(d, [l["tempo_cabeca_ms"] for l in linhas], color=CATEGORICA[1],
                  marker=MARCADORES[1], label="cabeça de saída")
    eixos[2].legend(fontsize=7)
    estilizar_linhas(eixos[2], "Tempo por componente\n(lote 8 × 128 tokens)", "emb_dim", "ms")

    fig.suptitle("Experimento 03 — efeito da dimensão dos embeddings (4 blocos, 4 cabeças)", y=1.04)
    fig.tight_layout()
    salvar_figura(fig, "exp03-dimensao-embedding")


if __name__ == "__main__":
    executar()
