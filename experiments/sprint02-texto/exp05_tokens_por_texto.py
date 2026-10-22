"""Experimento 05 — quantidade de tokens produzidos para diferentes textos.

**O que varia:** a natureza do texto — o corpus em inglês, um parágrafo em
português, código Python, números e palavras raras.
**O que fica fixo:** os dois tokenizadores da Sprint 2, com o vocabulário
simples construído a partir de ``the-verdict.txt``.

**Pergunta:** quantos tokens um texto custa, e do que esse custo depende?

O número de tokens determina quanto texto cabe no contexto do modelo e quanto
processamento cada documento consome. Duas grandezas resumem o comportamento:

- **caracteres por token** — quanto texto cada token carrega em média;
- **tokens por palavra** — quantos pedaços o tokenizador usa por palavra.

O BPE do GPT-2 foi treinado majoritariamente em inglês. A expectativa é que ele
seja mais eficiente nesse idioma e fragmente mais o português, os números e as
palavras raras. O tokenizador simples tem o comportamento oposto: um token por
palavra sempre, mas com uma fração crescente de ``<|unk|>`` à medida que o texto
se afasta do corpus em que o vocabulário foi construído.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from comum import (
    CATEGORICA,
    TEXTO_CODIGO,
    TEXTO_NUMEROS,
    TEXTO_PORTUGUES,
    TEXTO_RARO,
    cabecalho,
    imprimir_tabela,
    salvar_figura,
    salvar_tabela,
)

from dados import carregar_texto
from tokenizacao import DESCONHECIDO, TokenizadorBPE, TokenizadorSimples, dividir_em_tokens

CONTEXTO_GPT2 = 1024


def _textos() -> list[tuple[str, str]]:
    return [
        ("inglês (corpus)", carregar_texto()),
        ("português", TEXTO_PORTUGUES),
        ("código Python", TEXTO_CODIGO),
        ("números", TEXTO_NUMEROS),
        ("palavras raras", TEXTO_RARO),
    ]


def executar() -> list[dict]:
    cabecalho("Experimento 05 — quantidade de tokens para diferentes textos")
    simples = TokenizadorSimples.a_partir_do_texto(carregar_texto())
    bpe = TokenizadorBPE()

    linhas: list[dict] = []
    for nome, texto in _textos():
        palavras = len(texto.split())
        tokens_s = dividir_em_tokens(texto)
        desconhecidos = sum(t not in simples.str_para_id for t in tokens_s)
        n_bpe = len(bpe.encode(texto))
        caracteres_por_token = len(texto) / n_bpe
        linhas.append(
            {
                "texto": nome,
                "caracteres": len(texto),
                "palavras": palavras,
                "tokens_simples": len(tokens_s),
                "unk_simples_pct": 100 * desconhecidos / len(tokens_s),
                "tokens_bpe": n_bpe,
                "caracteres_por_token_bpe": caracteres_por_token,
                "tokens_bpe_por_palavra": n_bpe / palavras,
                "caracteres_em_1024_tokens": round(caracteres_por_token * CONTEXTO_GPT2),
            }
        )

    imprimir_tabela(linhas)
    salvar_tabela(linhas, "exp05-tokens-por-texto")
    _graficos(linhas)
    return linhas


def _graficos(linhas: list[dict]) -> None:
    nomes = [l["texto"] for l in linhas]
    posicoes = np.arange(len(nomes))
    fig, eixos = plt.subplots(1, 3, figsize=(12.5, 3.9))

    eixos[0].barh(posicoes, [l["caracteres_por_token_bpe"] for l in linhas], color=CATEGORICA[0])
    eixos[0].set_yticks(posicoes, nomes)
    eixos[0].invert_yaxis()
    eixos[0].grid(axis="y", visible=False)
    eixos[0].set_title("Caracteres por token (BPE)", pad=8)
    eixos[0].set_xlabel("caracteres / token")

    largura = 0.38
    eixos[1].barh(
        posicoes - largura / 2, [l["tokens_simples"] / l["palavras"] for l in linhas],
        height=largura, color=CATEGORICA[0], label="simples",
    )
    eixos[1].barh(
        posicoes + largura / 2, [l["tokens_bpe_por_palavra"] for l in linhas],
        height=largura, color=CATEGORICA[2], label="BPE",
    )
    eixos[1].set_yticks(posicoes, nomes)
    eixos[1].invert_yaxis()
    eixos[1].grid(axis="y", visible=False)
    eixos[1].legend(fontsize=7)
    eixos[1].set_title("Tokens por palavra", pad=8)
    eixos[1].set_xlabel("tokens / palavra")

    eixos[2].barh(posicoes, [l["unk_simples_pct"] for l in linhas], color=CATEGORICA[1])
    eixos[2].set_yticks(posicoes, nomes)
    eixos[2].invert_yaxis()
    eixos[2].set_xlim(0, 100)
    eixos[2].grid(axis="y", visible=False)
    eixos[2].set_title(f"Tokens que viram {DESCONHECIDO}\n(tokenizador simples)", pad=8)
    eixos[2].set_xlabel("% dos tokens")

    fig.suptitle("Experimento 05 — custo em tokens de diferentes textos", y=1.04)
    fig.tight_layout()
    salvar_figura(fig, "exp05-tokens-por-texto")


if __name__ == "__main__":
    executar()
