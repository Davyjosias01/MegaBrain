"""Utilidades compartilhadas pelos experimentos da Sprint 2.

Centraliza o que todos os experimentos precisam: acesso aos módulos de
``src/``, semente fixa, medição de tempo e de memória, os textos de teste e um
estilo único de gráfico — o mesmo dos experimentos da Sprint 3, para que as
figuras do projeto formem um conjunto coerente.

Todas as figuras e tabelas são gravadas em ``results-by-sprints/sprint02/``.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

# --- caminhos --------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[2]
SRC = RAIZ / "src"
RESULTADOS = RAIZ / "results-by-sprints" / "sprint02"
RESULTADOS.mkdir(parents=True, exist_ok=True)

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# --- reprodutibilidade -----------------------------------------------------
SEMENTE = 123


def semear(semente: int = SEMENTE) -> None:
    """Fixa a semente global antes de cada configuração medida."""
    torch.manual_seed(semente)


# --- textos de teste ---------------------------------------------------------
# Textos curtos, de naturezas diferentes, escritos para os experimentos 04 e 05.
# Ficam fixos no código para que os resultados não mudem se outros arquivos do
# repositório forem editados.
TEXTO_PORTUGUES = (
    "Este projeto acompanha a construção incremental de um modelo de linguagem a "
    "partir do zero. Em vez de consumir um modelo já treinado, cada componente é "
    "implementado, testado e analisado separadamente: a tokenização transforma o "
    "texto em números, os embeddings transformam esses números em vetores e o "
    "mecanismo de atenção relaciona os tokens entre si. A tradução automática de "
    "frases, a geração de código e a compreensão de perguntas dependem dessa mesma "
    "sequência de etapas."
)

TEXTO_CODIGO = '''def scaled_dot_product_attention(queries, keys, values, mask=None):
    d_k = keys.shape[-1]
    attn_scores = queries @ keys.transpose(-2, -1)
    attn_scores = attn_scores / d_k**0.5
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
    attn_weights = torch.softmax(attn_scores, dim=-1)
    return attn_weights @ values, attn_weights
'''

TEXTO_NUMEROS = (
    "3.14159 2.71828 1.41421 1.73205 0.57721 1024 2048 4096 50257 768 12288 "
    "21/10/2026 23:59 R$ 1.299,90 +55 49 3551-2000 192.168.0.1"
)

TEXTO_RARO = (
    "Akwirw ier someunknownPlace supercalifragilisticexpialidocious "
    "pneumoultramicroscopicossilicovulcanoconiótico Xylophagous quokka zugzwang"
)

# --- paleta ----------------------------------------------------------------
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
GRADE = "#e3e2de"
CATEGORICA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARCADORES = ["o", "s", "^", "D"]

plt.rcParams.update(
    {
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "text.color": TINTA,
        "axes.labelcolor": TINTA_SECUNDARIA,
        "xtick.color": TINTA_SECUNDARIA,
        "ytick.color": TINTA_SECUNDARIA,
        "axes.edgecolor": GRADE,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRADE,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "figure.dpi": 160,
    }
)


# --- saída -----------------------------------------------------------------
def cabecalho(texto: str) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("\n" + "=" * 74)
    print(texto)
    print("=" * 74)


def salvar_figura(fig: plt.Figure, nome: str) -> Path:
    destino = RESULTADOS / f"{nome}.png"
    fig.savefig(destino, bbox_inches="tight")
    plt.close(fig)
    print(f"  figura -> results-by-sprints/sprint02/{destino.name}")
    return destino


def salvar_tabela(linhas: list[dict], nome: str) -> Path:
    destino = RESULTADOS / f"{nome}.csv"
    if not linhas:
        raise ValueError("nenhuma linha para gravar")
    with destino.open("w", encoding="utf-8", newline="") as arq:
        escritor = csv.DictWriter(arq, fieldnames=list(linhas[0].keys()))
        escritor.writeheader()
        escritor.writerows(linhas)
    print(f"  tabela -> results-by-sprints/sprint02/{destino.name}")
    return destino


def imprimir_tabela(linhas: list[dict]) -> None:
    colunas = list(linhas[0].keys())
    larguras = [max(len(c), max(len(_fmt(l[c])) for l in linhas)) for c in colunas]
    print("  " + "  ".join(c.ljust(w) for c, w in zip(colunas, larguras)))
    print("  " + "  ".join("-" * w for w in larguras))
    for linha in linhas:
        print("  " + "  ".join(_fmt(linha[c]).ljust(w) for c, w in zip(colunas, larguras)))


def _fmt(valor: object) -> str:
    if isinstance(valor, bool):
        return "sim" if valor else "não"
    if isinstance(valor, float):
        return f"{valor:.4f}"
    if isinstance(valor, int):
        return f"{valor:,}"
    return str(valor)


# --- medidas ---------------------------------------------------------------
def cronometrar(funcao, repeticoes: int = 10, aquecimento: int = 2) -> float:
    """Tempo médio de uma chamada, em milissegundos."""
    with torch.no_grad():
        for _ in range(aquecimento):
            funcao()
        inicio = time.perf_counter()
        for _ in range(repeticoes):
            funcao()
        fim = time.perf_counter()
    return (fim - inicio) / repeticoes * 1000


def bytes_de(forma: tuple[int, ...], dtype: torch.dtype) -> int:
    """Memória ocupada por um tensor denso de ``forma`` e ``dtype``."""
    n = 1
    for d in forma:
        n *= d
    return n * torch.empty((), dtype=dtype).element_size()


def legivel(num_bytes: float) -> str:
    """``1536`` -> ``'1.5 KiB'``."""
    for unidade in ("B", "KiB", "MiB", "GiB"):
        if num_bytes < 1024 or unidade == "GiB":
            return f"{num_bytes:.0f} {unidade}" if unidade == "B" else f"{num_bytes:.1f} {unidade}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} GiB"


def similaridade_cosseno(vetores: torch.Tensor) -> torch.Tensor:
    """Matriz ``(n, n)`` de similaridade de cosseno entre as linhas de ``vetores``."""
    normalizados = torch.nn.functional.normalize(vetores.detach(), dim=-1)
    return normalizados @ normalizados.T


# --- gráficos ---------------------------------------------------------------
def estilizar_linhas(ax: plt.Axes, titulo: str, x: str, y: str) -> None:
    ax.set_title(titulo, color=TINTA, pad=8)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(axis="x", visible=False)
