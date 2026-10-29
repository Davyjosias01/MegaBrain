"""Experimento 06 — tamanho do contexto.

**O que varia:** (a) o número de tokens da sequência, de 16 a 1.024;
(b) a posição na geração, de 1 a 300 tokens de contexto.
**O que fica fixo:** configuração-base (``emb_dim = 256``, 4 cabeças, 4 blocos),
semente 123, ``eval()``. Em (a) ``context_length = 1024``; em (b),
``context_length = 128``.

**Pergunta:** como o comprimento da sequência pesa no custo do modelo e da
geração?

(a) Atenção compara cada token com todos os anteriores: as matrizes de pesos
    têm ``n_heads · L²`` elementos por bloco, enquanto as projeções lineares e
    a feed forward crescem com ``L``. O experimento mede onde a curva de tempo
    passa a ser dominada pelo termo quadrático.

(b) A geração implementada não guarda nada entre passos: a cada token novo o
    modelo reprocessa o contexto inteiro. O custo de cada passo deve crescer
    com o tamanho do contexto até atingir ``context_length``; a partir daí o
    contexto é truncado aos últimos 128 tokens e o custo se estabiliza.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import torch

from comum import (
    CATEGORICA,
    MARCADORES,
    TINTA_SECUNDARIA,
    cabecalho,
    cronometrar,
    estilizar_linhas,
    imprimir_tabela,
    novo_modelo,
    salvar_figura,
    salvar_tabela,
)

from geracao import selecionar_proximo_token, texto_para_ids
from gpt import contar_parametros

COMPRIMENTOS = [16, 32, 64, 128, 256, 512, 1024]
CONTEXTO_GERACAO = 128
PASSOS_GERACAO = 300


def executar() -> list[dict]:
    cabecalho("Experimento 06 — tamanho do contexto")

    # (a) custo da passagem direta em função do comprimento da sequência
    modelo = novo_modelo(context_length=1024)
    n_heads, n_layers = modelo.cfg["n_heads"], modelo.cfg["n_layers"]
    linhas: list[dict] = []
    for L in COMPRIMENTOS:
        torch.manual_seed(0)
        ids = torch.randint(0, modelo.cfg["vocab_size"], (1, L))
        with torch.no_grad():
            x = modelo.embedding(ids)
        tempo = cronometrar(lambda: modelo.trf_blocks(x), repeticoes=5)
        linhas.append(
            {
                "num_tokens": L,
                "elementos_atencao_total": n_layers * n_heads * L * L,
                "memoria_atencao_mb": n_layers * n_heads * L * L * 4 / 1024**2,
                "tempo_blocos_ms": tempo,
                "tempo_por_token_us": 1000 * tempo / L,
            }
        )
    imprimir_tabela(linhas)
    salvar_tabela(linhas, "exp06-tamanho-do-contexto")

    parametros = {c: contar_parametros(novo_modelo(context_length=c)) for c in (128, 1024)}
    print(f"\n  parâmetros com context_length 128: {parametros[128]:,}; "
          f"com 1024: {parametros[1024]:,} "
          f"(+{parametros[1024] - parametros[128]:,}, só na tabela posicional)")

    # (b) custo de cada passo da geração sem cache
    modelo = novo_modelo(context_length=CONTEXTO_GERACAO)
    ids = texto_para_ids("Every")
    passos: list[dict] = []
    with torch.no_grad():
        for passo in range(PASSOS_GERACAO):
            contexto = ids[:, -CONTEXTO_GERACAO:]
            inicio = time.perf_counter()
            logits = modelo(contexto)[:, -1, :]
            duracao = (time.perf_counter() - inicio) * 1000
            ids = torch.cat((ids, selecionar_proximo_token(logits)), dim=1)
            passos.append({"passo": passo + 1, "tokens_no_contexto": contexto.shape[1], "tempo_ms": duracao})
    salvar_tabela(passos, "exp06-custo-por-passo-de-geracao")
    antes = sum(p["tempo_ms"] for p in passos[100:120]) / 20
    depois = sum(p["tempo_ms"] for p in passos[-20:]) / 20
    inicio = sum(p["tempo_ms"] for p in passos[2:22]) / 20
    print(f"  passo médio com ~10 tokens: {inicio:.2f} ms; com ~110: {antes:.2f} ms; "
          f"depois da janela de {CONTEXTO_GERACAO}: {depois:.2f} ms")

    _graficos(linhas, passos)
    return linhas


def _graficos(linhas: list[dict], passos: list[dict]) -> None:
    fig, eixos = plt.subplots(1, 3, figsize=(12.5, 3.8))
    L = [l["num_tokens"] for l in linhas]

    eixos[0].plot(L, [l["tempo_blocos_ms"] for l in linhas], color=CATEGORICA[0], marker=MARCADORES[0])
    eixos[0].set_xscale("log", base=2)
    eixos[0].set_yscale("log")
    estilizar_linhas(eixos[0], "Tempo dos 4 blocos\n(lote 1)", "tokens na sequência", "ms")

    eixos[1].plot(L, [l["memoria_atencao_mb"] for l in linhas], color=CATEGORICA[1], marker=MARCADORES[1])
    eixos[1].set_xscale("log", base=2)
    eixos[1].set_yscale("log")
    estilizar_linhas(eixos[1], "Memória dos pesos de atenção\n(4 blocos × 4 cabeças × L²)",
                     "tokens na sequência", "MB")

    eixos[2].plot([p["passo"] for p in passos], [p["tempo_ms"] for p in passos],
                  color=CATEGORICA[2], linewidth=1.0)
    eixos[2].axvline(CONTEXTO_GERACAO, color=TINTA_SECUNDARIA, linestyle="--", linewidth=1)
    eixos[2].text(CONTEXTO_GERACAO + 4, eixos[2].get_ylim()[1] * 0.9,
                  f"context_length = {CONTEXTO_GERACAO}", fontsize=7, color=TINTA_SECUNDARIA)
    estilizar_linhas(eixos[2], "Custo de cada passo da geração\n(sem cache)", "token gerado", "ms")

    fig.suptitle("Experimento 06 — efeito do tamanho do contexto", y=1.04)
    fig.tight_layout()
    salvar_figura(fig, "exp06-tamanho-do-contexto")


if __name__ == "__main__":
    executar()
