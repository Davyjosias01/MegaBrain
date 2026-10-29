"""Experimento 02 — número de heads.

**O que varia:** ``n_heads`` em {1, 2, 4, 8, 16}.
**O que fica fixo:** ``emb_dim = 256`` (logo ``head_dim = 256 / n_heads``),
4 blocos, contexto 256, lote real ``(8, 128)``, semente 123, ``eval()``.

**Pergunta:** o número de cabeças muda o tamanho do modelo, o seu custo ou o
seu comportamento?

Na Sprint 3 o experimento equivalente foi feito em um único módulo de
atenção. Aqui ele é repetido dentro do modelo completo, para ver se a
conclusão — cabeças não custam parâmetros, custam execução — se mantém quando a
atenção é só uma parte do bloco. Também é registrado o tamanho das matrizes de
pesos de atenção, ``b · n_heads · L²`` elementos por bloco, que cresce com o
número de cabeças.
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
    entropia_proximo_token,
    estilizar_linhas,
    imprimir_tabela,
    lote_do_corpus,
    ms_por_token_gerado,
    novo_modelo,
    salvar_figura,
    salvar_tabela,
)

from geracao import texto_para_ids
from gpt import contar_parametros

HEADS = [1, 2, 4, 8, 16]
EMB_DIM = 256


def executar() -> list[dict]:
    cabecalho("Experimento 02 — número de heads")
    entradas, _ = lote_do_corpus()
    prompt = texto_para_ids(PROMPT)
    b, L = entradas.shape

    linhas: list[dict] = []
    for h in HEADS:
        modelo = novo_modelo(n_heads=h)
        with torch.no_grad():
            embeddings = modelo.drop_emb(modelo.embedding(entradas))
            logits = modelo(entradas[:, :32])[:, -16:]
        linhas.append(
            {
                "n_heads": h,
                "head_dim": EMB_DIM // h,
                "parametros": contar_parametros(modelo),
                "elementos_atencao_por_bloco": b * h * L * L,
                "tempo_blocos_ms": cronometrar(lambda: modelo.trf_blocks(embeddings)),
                "ms_por_token_gerado": ms_por_token_gerado(modelo, prompt),
                "entropia_proximo_token": entropia_proximo_token(logits),
            }
        )

    imprimir_tabela(linhas)
    salvar_tabela(linhas, "exp02-numero-de-heads")
    _graficos(linhas)
    return linhas


def _graficos(linhas: list[dict]) -> None:
    h = [l["n_heads"] for l in linhas]
    fig, eixos = plt.subplots(1, 3, figsize=(12, 3.7))

    eixos[0].plot(h, [l["parametros"] / 1e6 for l in linhas], color=CATEGORICA[0], marker=MARCADORES[0])
    eixos[0].set_xscale("log", base=2)
    eixos[0].set_ylim(0, max(l["parametros"] for l in linhas) / 1e6 * 1.15)
    estilizar_linhas(eixos[0], "Parâmetros do modelo", "n_heads", "milhões")

    eixos[1].plot(h, [l["elementos_atencao_por_bloco"] / 1e6 for l in linhas],
                  color=CATEGORICA[1], marker=MARCADORES[1])
    eixos[1].set_xscale("log", base=2)
    eixos[1].set_yscale("log", base=2)
    estilizar_linhas(eixos[1], "Pesos de atenção por bloco\n(b · n_heads · L²)", "n_heads", "milhões de elementos")

    eixos[2].plot(h, [l["tempo_blocos_ms"] for l in linhas], color=CATEGORICA[2], marker=MARCADORES[2])
    eixos[2].set_xscale("log", base=2)
    estilizar_linhas(eixos[2], "Tempo dos 4 blocos\n(lote 8 × 128 tokens)", "n_heads", "ms")

    fig.suptitle(f"Experimento 02 — efeito do número de heads (emb_dim {EMB_DIM} fixo)", y=1.04)
    fig.tight_layout()
    salvar_figura(fig, "exp02-numero-de-heads")


if __name__ == "__main__":
    executar()
