"""Utilidades compartilhadas pelos experimentos da Sprint 3.

Centraliza o que todos os experimentos precisam: acesso aos módulos de
``src/``, semente fixa, medição de tempo, métricas sobre a matriz de atenção e
um estilo único de gráfico. Assim cada script de experimento contém apenas o que
está de fato sob teste.

Todas as figuras e tabelas são gravadas em ``results-by-sprints/sprint03/``.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# --- caminhos --------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[2]
SRC = RAIZ / "src"
RESULTADOS = RAIZ / "results-by-sprints" / "sprint03"
RESULTADOS.mkdir(parents=True, exist_ok=True)

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# --- reprodutibilidade -----------------------------------------------------
SEMENTE = 123


def semear(semente: int = SEMENTE) -> None:
    """Fixa a semente global antes de cada configuração medida."""
    torch.manual_seed(semente)


# Exemplo canônico do livro (seção 3.3.1, p. 71): "Your journey starts with one step".
ENTRADAS_LIVRO = torch.tensor(
    [
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55],
    ]
)
TOKENS_LIVRO = ["Your", "journey", "starts", "with", "one", "step"]

# --- paleta ----------------------------------------------------------------
# Escala sequencial de um único tom (azul, claro -> escuro) para magnitude, e
# uma paleta categórica de ordem fixa para identidade de séries.
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
GRADE = "#e3e2de"
NEUTRO = "#d9d8d3"  # células sem dado (posições mascaradas)

CATEGORICA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARCADORES = ["o", "s", "^", "D"]

SEQUENCIAL = LinearSegmentedColormap.from_list(
    "azul_sequencial",
    ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

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
    print(f"  figura -> results-by-sprints/sprint03/{destino.name}")
    return destino


def salvar_tabela(linhas: list[dict], nome: str) -> Path:
    destino = RESULTADOS / f"{nome}.csv"
    if not linhas:
        raise ValueError("nenhuma linha para gravar")
    with destino.open("w", encoding="utf-8", newline="") as arq:
        escritor = csv.DictWriter(arq, fieldnames=list(linhas[0].keys()))
        escritor.writeheader()
        escritor.writerows(linhas)
    print(f"  tabela -> results-by-sprints/sprint03/{destino.name}")
    return destino


def imprimir_tabela(linhas: list[dict]) -> None:
    colunas = list(linhas[0].keys())
    larguras = [
        max(len(c), max(len(_fmt(l[c])) for l in linhas)) for c in colunas
    ]
    print("  " + "  ".join(c.ljust(w) for c, w in zip(colunas, larguras)))
    print("  " + "  ".join("-" * w for w in larguras))
    for linha in linhas:
        print("  " + "  ".join(_fmt(linha[c]).ljust(w) for c, w in zip(colunas, larguras)))


def _fmt(valor: object) -> str:
    if isinstance(valor, float):
        return f"{valor:.4f}"
    if isinstance(valor, int):
        return f"{valor:,}"
    return str(valor)


# --- métricas --------------------------------------------------------------
def entropia_normalizada(pesos: torch.Tensor, causal: bool = True) -> float:
    """Entropia média das linhas da matriz de atenção, normalizada para [0, 1].

    A entropia de Shannon ``H = -Σ p·log(p)`` mede o quanto a atenção está
    espalhada: ``H`` é máxima quando todos os pesos válidos são iguais e nula
    quando toda a massa está em um único token. Dividindo por ``log(k)``, em que
    ``k`` é o número de posições visíveis naquela linha, o valor fica em ``[0, 1]``
    e passa a ser comparável entre linhas e entre configurações de tamanhos
    diferentes:

    - perto de **1** — atenção quase uniforme, o token olha todo mundo por igual;
    - perto de **0** — atenção concentrada, o token olha essencialmente um só.

    Em atenção causal a linha 0 é ignorada: ela tem uma única posição visível e
    seu peso é sempre 1, o que não carrega informação.
    """
    pesos = pesos.detach()
    p = pesos.reshape(-1, pesos.shape[-2], pesos.shape[-1])
    n = p.shape[-1]
    h = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)

    if causal:
        k = torch.arange(1, n + 1, dtype=h.dtype, device=h.device)
        h_max = torch.log(k)
        valido = k > 1
    else:
        h_max = torch.full((n,), float(np.log(n)), dtype=h.dtype, device=h.device)
        valido = torch.ones(n, dtype=torch.bool, device=h.device)

    return float((h[:, valido] / h_max[valido]).mean())


def massa_no_primeiro_token(pesos: torch.Tensor) -> float:
    """Fração média da atenção que cada token dirige ao primeiro token da sequência."""
    p = pesos.detach().reshape(-1, pesos.shape[-2], pesos.shape[-1])
    return float(p[:, 1:, 0].mean())


def dissimilaridade_entre_cabecas(pesos: torch.Tensor) -> float:
    """Distância média entre as matrizes de atenção de cada par de cabeças.

    ``pesos`` tem forma ``(b, num_heads, num_tokens, num_tokens)``. Devolve a
    média das distâncias L1 normalizadas entre pares distintos de cabeças. Zero
    significa cabeças idênticas (redundantes); quanto maior, mais cada cabeça
    distribui a atenção de um jeito próprio.
    """
    p = pesos.detach().mean(dim=0)  # média sobre o lote -> (num_heads, n, n)
    h = p.shape[0]
    if h < 2:
        return 0.0
    distancias = [
        float((p[i] - p[j]).abs().sum() / p.shape[-2])
        for i in range(h)
        for j in range(i + 1, h)
    ]
    return sum(distancias) / len(distancias)


def cronometrar(funcao, repeticoes: int = 30, aquecimento: int = 5) -> float:
    """Tempo médio de uma passagem direta, em milissegundos."""
    with torch.no_grad():
        for _ in range(aquecimento):
            funcao()
        inicio = time.perf_counter()
        for _ in range(repeticoes):
            funcao()
        fim = time.perf_counter()
    return (fim - inicio) / repeticoes * 1000


def contar_parametros(modulo: torch.nn.Module) -> int:
    return sum(p.numel() for p in modulo.parameters())


# --- gráficos ---------------------------------------------------------------
def desenhar_matriz_atencao(
    ax: plt.Axes,
    matriz: torch.Tensor | np.ndarray,
    titulo: str,
    rotulos: list[str] | None = None,
    marcar_mascarados: bool = False,
    anotar: bool = False,
    vmax: float | None = None,
) -> None:
    """Desenha uma matriz de atenção como mapa de calor sequencial.

    As linhas são as queries (o token que olha) e as colunas são as keys (o token
    olhado). Magnitude pede escala sequencial de um único tom: claro = pouca
    atenção, escuro = muita.

    Com ``marcar_mascarados``, as posições exatamente nulas acima da diagonal são
    pintadas de cinza neutro, para distinguir "escondido pela máscara causal" de
    "visível, mas com peso quase zero".
    """
    m = matriz.detach().cpu().numpy() if isinstance(matriz, torch.Tensor) else np.asarray(matriz)
    n = m.shape[0]

    if marcar_mascarados:
        mascarado = np.triu(np.ones_like(m, dtype=bool), k=1) & (m == 0)
        ax.imshow(
            np.where(mascarado, 1.0, np.nan),
            cmap=LinearSegmentedColormap.from_list("neutro", [NEUTRO, NEUTRO]),
            vmin=0,
            vmax=1,
        )
        m = np.where(mascarado, np.nan, m)

    limite = vmax if vmax else float(np.nanmax(m))
    im = ax.imshow(m, cmap=SEQUENCIAL, vmin=0.0, vmax=limite)

    ax.set_title(titulo, color=TINTA, pad=8)
    ax.set_xlabel("key — token olhado")
    ax.set_ylabel("query — token que olha")
    ax.grid(False)
    ax.tick_params(length=0)

    if rotulos is not None and n <= 24:
        ax.set_xticks(range(n), rotulos, rotation=90, fontsize=7)
        ax.set_yticks(range(n), rotulos, fontsize=7)
    else:
        passo = max(1, n // 8)
        ax.set_xticks(range(0, n, passo))
        ax.set_yticks(range(0, n, passo))

    # Separação de 1px entre células, como exige a especificação de marcas.
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=SUPERFICIE, linewidth=1.0)
    ax.tick_params(which="minor", length=0)

    if anotar and n <= 8:
        for i in range(n):
            for j in range(n):
                if np.isnan(m[i, j]):
                    continue
                ax.text(
                    j,
                    i,
                    f"{m[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    # O limiar acompanha a escala de cor efetiva, para o texto
                    # nunca sair branco sobre uma célula clara.
                    color="#ffffff" if m[i, j] > 0.55 * limite else TINTA_SECUNDARIA,
                )

    barra = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    barra.set_label("peso de atenção", color=TINTA_SECUNDARIA, fontsize=8)
    barra.outline.set_edgecolor(GRADE)
    barra.ax.tick_params(labelsize=7)


def estilizar_linhas(ax: plt.Axes, titulo: str, x: str, y: str) -> None:
    ax.set_title(titulo, color=TINTA, pad=8)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(axis="x", visible=False)
