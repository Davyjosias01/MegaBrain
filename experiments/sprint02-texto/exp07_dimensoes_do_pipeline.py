"""Experimento 07 — dimensões das estruturas produzidas pelo pipeline.

**O que varia:** a configuração — a do exemplo do livro, a usada nos
experimentos da Sprint 3 e a do GPT-2 small.
**O que fica fixo:** o corpus ``the-verdict.txt``, ``stride = max_length``,
semente 123.

**Pergunta:** qual a forma, o tipo e o tamanho de cada estrutura entre o texto
bruto e a entrada do modelo?

O pipeline é executado de verdade, etapa por etapa, e cada estrutura
intermediária é registrada — nada é calculado só pela fórmula. As etapas são as
da Sprint 2:

    texto -> tokens -> token IDs -> amostras -> lote de IDs (b, L)
          -> embedding de token (b, L, d) + embedding posicional (L, d)
          -> entrada do modelo (b, L, d)

Duas transições merecem atenção: a passagem de uma lista Python para tensores
``int64`` (8 bytes por ID) e a passagem de IDs para vetores ``float32``, que
multiplica a memória por ``d / 2`` — de longe a maior expansão do pipeline.
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from comum import (
    CATEGORICA,
    cabecalho,
    imprimir_tabela,
    salvar_figura,
    salvar_tabela,
    semear,
)

from dados import carregar_texto, criar_dataloader
from embeddings import EmbeddingDeEntrada
from tokenizacao import TokenizadorBPE

CONFIGURACOES = [
    # nome, batch, max_length, d_emb
    ("livro (seção 2.8)", 8, 4, 256),
    ("experimentos da Sprint 3", 8, 128, 64),
    ("GPT-2 small", 2, 1024, 768),
]


def _registro(config: str, etapa: str, forma: str, tipo: str, elementos: int, num_bytes: int) -> dict:
    return {
        "configuracao": config,
        "etapa": etapa,
        "forma": forma,
        "tipo": tipo,
        "elementos": elementos,
        "bytes": num_bytes,
    }


def _tensor(config: str, etapa: str, t: torch.Tensor) -> dict:
    return _registro(
        config, etapa, "(" + ", ".join(map(str, t.shape)) + ")",
        str(t.dtype).replace("torch.", ""), t.numel(), t.numel() * t.element_size(),
    )


def executar() -> list[dict]:
    cabecalho("Experimento 07 — dimensões das estruturas do pipeline")
    texto = carregar_texto()
    bpe = TokenizadorBPE()
    ids = bpe.encode(texto)
    pedacos = bpe.ids_para_tokens(ids)

    linhas: list[dict] = []
    for nome, lote, contexto, d in CONFIGURACOES:
        linhas.append(_registro(nome, "1. texto bruto", f"{len(texto)} caracteres", "str",
                                len(texto), len(texto.encode("utf-8"))))
        linhas.append(_registro(nome, "2. tokens", f"lista de {len(pedacos)}", "list[str]",
                                len(pedacos), sum(len(p.encode("utf-8")) for p in pedacos)))
        linhas.append(_registro(nome, "3. token IDs", f"lista de {len(ids)}", "list[int]",
                                len(ids), sys.getsizeof(ids) + sum(sys.getsizeof(i) for i in ids)))

        carregador = criar_dataloader(
            texto, batch_size=lote, max_length=contexto, stride=contexto, tokenizador=bpe
        )
        amostras = len(carregador.dataset)
        # O dataset guarda entrada e alvo de cada amostra: 2 tensores de max_length.
        linhas.append(_registro(nome, "4. amostras (entradas + alvos)",
                                f"{amostras} × 2 × ({contexto},)", "int64",
                                2 * amostras * contexto, 2 * amostras * contexto * 8))

        entradas, alvos = next(iter(carregador))
        linhas.append(_tensor(nome, "5. lote de entrada", entradas))
        linhas.append(_tensor(nome, "6. lote de alvos", alvos))

        semear()
        camada = EmbeddingDeEntrada(bpe.tamanho_vocabulario, d, contexto)
        with torch.no_grad():
            emb_token = camada.token_emb(entradas)
            emb_pos = camada.pos_emb(torch.arange(contexto))
            entrada_modelo = camada(entradas)

        linhas.append(_tensor(nome, "7. tabela de tokens (pesos)", camada.token_emb.weight.detach()))
        linhas.append(_tensor(nome, "8. tabela posicional (pesos)", camada.pos_emb.weight.detach()))
        linhas.append(_tensor(nome, "9. embeddings de token", emb_token))
        linhas.append(_tensor(nome, "10. embeddings posicionais", emb_pos))
        linhas.append(_tensor(nome, "11. entrada do modelo", entrada_modelo))
        assert torch.allclose(entrada_modelo, emb_token + emb_pos)

    for nome, *_ in CONFIGURACOES:
        print(f"\n  [{nome}]")
        imprimir_tabela(
            [{k: v for k, v in l.items() if k != "configuracao"} for l in linhas
             if l["configuracao"] == nome]
        )
    salvar_tabela(linhas, "exp07-dimensoes-do-pipeline")
    _grafico(linhas)
    return linhas


def _grafico(linhas: list[dict]) -> None:
    # Só as estruturas que dependem da configuração (do lote em diante).
    etapas = [e for e in dict.fromkeys(l["etapa"] for l in linhas)
              if int(e.split(".")[0]) >= 5]
    posicoes = np.arange(len(etapas))
    largura = 0.26

    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, (nome, *_) in enumerate(CONFIGURACOES):
        valores = [
            next(l["bytes"] for l in linhas if l["configuracao"] == nome and l["etapa"] == e)
            for e in etapas
        ]
        ax.bar(posicoes + (i - 1) * largura, valores, width=largura,
               color=CATEGORICA[i], label=nome)

    ax.set_yscale("log")
    ax.set_xticks(posicoes, [e.split(". ", 1)[1] for e in etapas], rotation=20, ha="right")
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("bytes (escala log)")
    ax.legend(fontsize=8)
    ax.set_title("Experimento 07 — memória de cada estrutura do pipeline", pad=8)
    fig.tight_layout()
    salvar_figura(fig, "exp07-dimensoes-do-pipeline")


if __name__ == "__main__":
    executar()
