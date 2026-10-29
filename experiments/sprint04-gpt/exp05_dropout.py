"""Experimento 05 — dropout.

**O que varia:** ``drop_rate`` em {0.0, 0.1, 0.3, 0.5} e o modo do modelo
(``train()`` ou ``eval()``).
**O que fica fixo:** configuração-base, lote real ``(2, 64)``, prompt
"Every effort moves you", semente 123.

**Pergunta:** o que o dropout faz com as saídas do modelo, e por que a
inferência precisa de ``eval()``?

O GPT aplica dropout em três lugares: sobre a soma dos embeddings, sobre os
pesos de atenção e sobre a saída de cada subcamada, antes da conexão residual.
Em ``train()`` cada passagem sorteia uma máscara nova, então a mesma entrada
produz logits diferentes; em ``eval()`` o dropout é desligado e a saída é
determinística. O experimento mede:

- a **variação dos logits** entre 10 passagens da mesma entrada (desvio entre
  passagens, relativo à escala dos logits);
- se a **geração greedy** de 10 tokens muda de uma execução para outra quando o
  modelo é esquecido em modo de treino;
- o **custo** do dropout: tempo em ``train()`` contra ``eval()``.
"""

from __future__ import annotations

import warnings

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
    novo_modelo,
    salvar_figura,
    salvar_tabela,
)

from geracao import gerar, texto_para_ids

TAXAS = [0.0, 0.1, 0.3, 0.5]
PASSAGENS = 10
NOVOS_TOKENS = 10


def _variacao_relativa(modelo, entradas) -> float:
    with torch.no_grad():
        saidas = torch.stack([modelo(entradas) for _ in range(PASSAGENS)])
    return float(saidas.std(dim=0).mean() / saidas.abs().mean())


def executar() -> list[dict]:
    cabecalho("Experimento 05 — dropout")
    entradas, _ = lote_do_corpus(batch=2, num_tokens=64)
    prompt = texto_para_ids(PROMPT)

    linhas: list[dict] = []
    for taxa in TAXAS:
        modelo = novo_modelo(drop_rate=taxa)
        referencia = gerar(modelo.eval(), prompt, NOVOS_TOKENS, 256)
        variacao_eval = _variacao_relativa(modelo.eval(), entradas)
        tempo_eval = cronometrar(lambda: modelo(entradas))

        modelo.train()
        torch.manual_seed(0)
        variacao_train = _variacao_relativa(modelo, entradas)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # o aviso de modo de treino é o objeto do teste
            geracoes = [gerar(modelo, prompt, NOVOS_TOKENS, 256) for _ in range(5)]
        diferentes = sum(not torch.equal(g, referencia) for g in geracoes)
        tempo_train = cronometrar(lambda: modelo(entradas))

        linhas.append(
            {
                "drop_rate": taxa,
                "variacao_logits_eval": variacao_eval,
                "variacao_logits_train": variacao_train,
                "geracoes_diferentes_em_train": f"{diferentes}/5",
                "tempo_eval_ms": tempo_eval,
                "tempo_train_ms": tempo_train,
            }
        )

    imprimir_tabela(linhas)
    salvar_tabela(linhas, "exp05-dropout")
    _graficos(linhas)
    return linhas


def _graficos(linhas: list[dict]) -> None:
    taxa = [l["drop_rate"] for l in linhas]
    fig, eixos = plt.subplots(1, 2, figsize=(10.5, 3.8))

    eixos[0].plot(taxa, [100 * l["variacao_logits_train"] for l in linhas],
                  color=CATEGORICA[1], marker=MARCADORES[1], label="train()")
    eixos[0].plot(taxa, [100 * l["variacao_logits_eval"] for l in linhas],
                  color=CATEGORICA[0], marker=MARCADORES[0], label="eval()")
    eixos[0].legend(fontsize=7)
    estilizar_linhas(eixos[0], f"Variação dos logits entre {PASSAGENS} passagens\nda mesma entrada",
                     "drop_rate", "% da escala dos logits")

    eixos[1].plot(taxa, [l["tempo_train_ms"] for l in linhas],
                  color=CATEGORICA[1], marker=MARCADORES[1], label="train()")
    eixos[1].plot(taxa, [l["tempo_eval_ms"] for l in linhas],
                  color=CATEGORICA[0], marker=MARCADORES[0], label="eval()")
    eixos[1].set_ylim(bottom=0)
    eixos[1].legend(fontsize=7)
    estilizar_linhas(eixos[1], "Tempo da passagem direta\n(lote 2 × 64 tokens)", "drop_rate", "ms")

    fig.suptitle("Experimento 05 — dropout em modo de treino e de inferência", y=1.04)
    fig.tight_layout()
    salvar_figura(fig, "exp05-dropout")


if __name__ == "__main__":
    executar()
