"""Experimento 04 — diferentes sequências de entrada.

**O que varia:** a frase de entrada — maiúsculas, contração, palavra fora do
corpus, português com acentos, números e espaços.
**O que fica fixo:** os dois tokenizadores da Sprint 2: ``TokenizadorSimples``
com o vocabulário de ``the-verdict.txt`` (1.132 tokens) e ``TokenizadorBPE``
(GPT-2, 50.257 tokens).

**Pergunta:** como a representação textual é modificada pela tokenização, e o
que se perde no caminho?

Para cada frase o experimento registra os pedaços produzidos, os token IDs,
quantos tokens caíram em ``<|unk|>`` e se ``decode(encode(texto))`` reconstrói
o texto original. A figura desenha cada frase como uma fita de tokens, para que
as diferenças de fatiamento fiquem visíveis.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from comum import (
    CATEGORICA,
    GRADE,
    TINTA,
    TINTA_SECUNDARIA,
    cabecalho,
    imprimir_tabela,
    salvar_figura,
    salvar_tabela,
)

from dados import carregar_texto
from tokenizacao import DESCONHECIDO, TokenizadorBPE, TokenizadorSimples, dividir_em_tokens

FRASES = [
    ("exemplo do livro", "Hello, do you like tea?"),
    ("maiúsculas", "The cat. the cat. THE CAT."),
    ("contração", "It's the last he painted, you know."),
    ("fora do corpus", "In the sunlit terraces of the palace."),
    ("português", "A atenção é o coração do Transformer."),
    ("números", "Em 2026 o GPT-2 tinha 50257 tokens."),
    ("espaços", "uma linha\n    outra  linha"),
]


def _pedacos_simples(tokenizador: TokenizadorSimples, texto: str) -> list[str]:
    """Pedaços do tokenizador simples, já com ``<|unk|>`` onde ele se aplica."""
    return [t if t in tokenizador.str_para_id else DESCONHECIDO for t in dividir_em_tokens(texto)]


def executar() -> list[dict]:
    cabecalho("Experimento 04 — diferentes sequências de entrada")
    simples = TokenizadorSimples.a_partir_do_texto(carregar_texto())
    bpe = TokenizadorBPE()
    print(
        f"  vocabulário simples={simples.tamanho_vocabulario}  "
        f"vocabulário BPE={bpe.tamanho_vocabulario}"
    )

    linhas: list[dict] = []
    fitas: list[tuple[str, list[str], list[str]]] = []
    for nome, texto in FRASES:
        ids_s = simples.encode(texto)
        ids_b = bpe.encode(texto)
        pedacos_s = _pedacos_simples(simples, texto)
        pedacos_b = bpe.ids_para_tokens(ids_b)
        fitas.append((nome, pedacos_s, pedacos_b))

        linhas.append(
            {
                "sequencia": nome,
                "caracteres": len(texto),
                "tokens_simples": len(ids_s),
                "desconhecidos_simples": pedacos_s.count(DESCONHECIDO),
                "reconstroi_simples": simples.decode(ids_s) == texto,
                "tokens_bpe": len(ids_b),
                "reconstroi_bpe": bpe.decode(ids_b) == texto,
                "pedacos_simples": " | ".join(pedacos_s),
                "pedacos_bpe": " | ".join(p.replace("\n", "\\n") for p in pedacos_b),
                "ids_bpe": " ".join(map(str, ids_b)),
            }
        )

    # A tabela impressa omite as colunas longas; o CSV guarda tudo.
    curtas = [k for k in linhas[0] if not k.startswith(("pedacos", "ids"))]
    imprimir_tabela([{k: l[k] for k in curtas} for l in linhas])
    for l in linhas:
        print(f"\n  [{l['sequencia']}]\n    simples: {l['pedacos_simples']}\n    BPE    : {l['pedacos_bpe']}")
    salvar_tabela(linhas, "exp04-sequencias-de-entrada")

    # Maiúsculas e minúsculas: IDs diferentes para a "mesma" palavra.
    variantes = ["the", " the", "The", " The", "THE"]
    print("\n  'the' no BPE, conforme a grafia:")
    for v in variantes:
        print(f"    {v!r:8} -> {bpe.encode(v)}")

    _grafico(fitas)
    return linhas


def _rotulo(pedaco: str) -> str:
    """Torna espaços e quebras de linha visíveis na fita de tokens."""
    return pedaco.replace("\n", "\\n").replace(" ", "·")


def _fita(ax: plt.Axes, y: float, pedacos: list[str], cor: str) -> float:
    x = 0.0
    for i, pedaco in enumerate(pedacos):
        rotulo = _rotulo(pedaco)
        largura = max(len(rotulo), 1) * 0.62 + 0.5
        desconhecido = pedaco == DESCONHECIDO
        ax.add_patch(
            FancyBboxPatch(
                (x, y - 0.32), largura, 0.64,
                boxstyle="round,pad=0,rounding_size=0.12",
                facecolor=CATEGORICA[1] if desconhecido else cor,
                alpha=1.0 if desconhecido else (0.22 if i % 2 else 0.38),
                edgecolor="none",
            )
        )
        ax.text(x + largura / 2, y, rotulo, ha="center", va="center", fontsize=7, color=TINTA)
        x += largura + 0.15
    return x


def _grafico(fitas: list[tuple[str, list[str], list[str]]]) -> None:
    altura = 1.0 * len(fitas) + 0.6
    fig, ax = plt.subplots(figsize=(12, altura))
    largura_max = 0.0
    for i, (nome, pedacos_s, pedacos_b) in enumerate(fitas):
        base = -i * 2.4
        ax.text(0, base + 0.62, nome, ha="left", va="center", fontsize=8, color=TINTA, weight="bold")
        largura_max = max(largura_max, _fita(ax, base, pedacos_s, CATEGORICA[0]))
        largura_max = max(largura_max, _fita(ax, base - 0.8, pedacos_b, CATEGORICA[2]))
        ax.text(-0.4, base, f"simples ({len(pedacos_s)})", ha="right", va="center",
                fontsize=7, color=TINTA_SECUNDARIA)
        ax.text(-0.4, base - 0.8, f"BPE ({len(pedacos_b)})", ha="right", va="center",
                fontsize=7, color=TINTA_SECUNDARIA)
        if i < len(fitas) - 1:
            ax.axhline(base - 1.45, color=GRADE, linewidth=0.6)

    ax.set_xlim(-7, largura_max + 0.5)
    ax.set_ylim(-2.4 * (len(fitas) - 1) - 1.3, 1.1)
    ax.axis("off")
    ax.set_title(
        "Experimento 04 — como cada tokenizador fatia a mesma frase  "
        "(azul: simples · verde: BPE · laranja: <|unk|> · '·' = espaço)",
        color=TINTA, pad=8,
    )
    salvar_figura(fig, "exp04-sequencias-de-entrada")


if __name__ == "__main__":
    executar()
