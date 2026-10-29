"""Experimento 01 — número de Transformer Blocks.

**O que varia:** ``n_layers`` em {1, 2, 4, 6, 8, 12}.
**O que fica fixo:** a configuração-base (``emb_dim = 256``, 4 cabeças,
contexto 256, vocabulário do GPT-2), lote real ``(8, 128)``, semente 123,
modelo em ``eval()``.

**Pergunta:** o que a profundidade custa e o que ela muda no comportamento do
modelo antes do treinamento?

A profundidade é o principal eixo de crescimento de um GPT: o GPT-2 vai de 12
a 48 blocos entre o small e o XL. Cada bloco acrescenta a mesma quantidade de
parâmetros e de computação, então o custo dos blocos deve crescer linearmente.
O experimento separa o tempo dos blocos do tempo da cabeça de saída, que não
depende da profundidade, e mede como o fluxo residual — a soma acumulada que
atravessa os blocos — se comporta à medida que mais blocos somam suas saídas a
ele.
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
from gpt import contar_parametros, parametros_por_componente, tamanho_em_mb

BLOCOS = [1, 2, 4, 6, 8, 12]


def executar() -> list[dict]:
    cabecalho("Experimento 01 — número de Transformer Blocks")
    entradas, _ = lote_do_corpus()
    prompt = texto_para_ids(PROMPT)
    print(f"  lote={tuple(entradas.shape)}  emb_dim=256  n_heads=4  contexto=256")

    linhas: list[dict] = []
    for n in BLOCOS:
        modelo = novo_modelo(n_layers=n)
        partes = parametros_por_componente(modelo)
        total = contar_parametros(modelo)
        dos_blocos = (partes["atenção (todos os blocos)"] + partes["feed forward (todos os blocos)"]
                      + partes["LayerNorm (blocos + final)"])

        with torch.no_grad():
            embeddings = modelo.drop_emb(modelo.embedding(entradas))
            fluxo = modelo.trf_blocks(embeddings)
            logits = modelo.out_head(modelo.final_norm(fluxo[:, -16:]))

        linhas.append(
            {
                "n_layers": n,
                "parametros": total,
                "parametros_dos_blocos_pct": 100 * dos_blocos / total,
                "memoria_mb": tamanho_em_mb(total),
                "tempo_blocos_ms": cronometrar(lambda: modelo.trf_blocks(embeddings)),
                "tempo_total_ms": cronometrar(lambda: modelo(entradas), repeticoes=3),
                "ms_por_token_gerado": ms_por_token_gerado(modelo, prompt),
                "desvio_fluxo_residual": float(fluxo.std()),
                "entropia_proximo_token": entropia_proximo_token(logits),
            }
        )

    imprimir_tabela(linhas)
    salvar_tabela(linhas, "exp01-numero-de-blocos")
    _graficos(linhas)
    return linhas


def _graficos(linhas: list[dict]) -> None:
    n = [l["n_layers"] for l in linhas]
    fig, eixos = plt.subplots(1, 3, figsize=(12, 3.7))

    eixos[0].plot(n, [l["parametros"] / 1e6 for l in linhas],
                  color=CATEGORICA[0], marker=MARCADORES[0])
    estilizar_linhas(eixos[0], "Parâmetros do modelo", "n_layers", "milhões")

    eixos[1].plot(n, [l["tempo_blocos_ms"] for l in linhas],
                  color=CATEGORICA[1], marker=MARCADORES[1], label="só os blocos")
    eixos[1].plot(n, [l["tempo_total_ms"] for l in linhas],
                  color=CATEGORICA[2], marker=MARCADORES[2], label="modelo inteiro")
    eixos[1].legend(fontsize=7)
    estilizar_linhas(eixos[1], "Tempo da passagem direta\n(lote 8 × 128 tokens)", "n_layers", "ms")

    eixos[2].plot(n, [l["desvio_fluxo_residual"] for l in linhas],
                  color=CATEGORICA[3], marker=MARCADORES[3])
    estilizar_linhas(eixos[2], "Escala do fluxo residual\nna saída do último bloco",
                     "n_layers", "desvio-padrão")

    fig.suptitle("Experimento 01 — efeito do número de blocos (emb_dim 256, 4 cabeças)", y=1.04)
    fig.tight_layout()
    salvar_figura(fig, "exp01-numero-de-blocos")


if __name__ == "__main__":
    executar()
