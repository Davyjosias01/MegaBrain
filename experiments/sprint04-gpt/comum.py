"""Utilidades compartilhadas pelos experimentos da Sprint 4.

Centraliza o que todos os experimentos precisam: acesso aos módulos de
``src/``, semente fixa, a configuração-base dos experimentos, um lote real do
corpus, medidas de tempo e de comportamento do modelo e o estilo de gráfico
das sprints anteriores.

Todas as figuras e tabelas são gravadas em ``results-by-sprints/sprint04/``.
"""

from __future__ import annotations

import csv
import math
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
RESULTADOS = RAIZ / "results-by-sprints" / "sprint04"
RESULTADOS.mkdir(parents=True, exist_ok=True)

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dados import carregar_texto, criar_dataloader  # noqa: E402
from geracao import gerar  # noqa: E402
from gpt import CONFIG_GPT2_124M, GPTModel  # noqa: E402

# --- reprodutibilidade -----------------------------------------------------
SEMENTE = 123


def semear(semente: int = SEMENTE) -> None:
    """Fixa a semente global antes de cada configuração medida."""
    torch.manual_seed(semente)


# --- configuração-base -----------------------------------------------------
# Um GPT reduzido, para que as varreduras caibam em minutos de CPU. Mantém o
# vocabulário do GPT-2 e muda só a escala: d = 256, 4 cabeças, 4 blocos.
CONFIG_BASE = dict(
    CONFIG_GPT2_124M,
    context_length=256,
    emb_dim=256,
    n_heads=4,
    n_layers=4,
)
BATCH = 8
NUM_TOKENS = 128
PROMPT = "Every effort moves you"


def lote_do_corpus(batch: int = BATCH, num_tokens: int = NUM_TOKENS) -> tuple[torch.Tensor, torch.Tensor]:
    """Primeiro lote ``(entradas, alvos)`` do corpus, com janelas encostadas."""
    carregador = criar_dataloader(
        carregar_texto(), batch_size=batch, max_length=num_tokens, stride=num_tokens
    )
    return next(iter(carregador))


def novo_modelo(semente: int = SEMENTE, **alteracoes) -> GPTModel:
    """``GPTModel`` com a configuração-base alterada, em modo ``eval()``."""
    semear(semente)
    return GPTModel(dict(CONFIG_BASE, **alteracoes)).eval()


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
    print(f"  figura -> results-by-sprints/sprint04/{destino.name}")
    return destino


def salvar_tabela(linhas: list[dict], nome: str) -> Path:
    destino = RESULTADOS / f"{nome}.csv"
    if not linhas:
        raise ValueError("nenhuma linha para gravar")
    with destino.open("w", encoding="utf-8", newline="") as arq:
        escritor = csv.DictWriter(arq, fieldnames=list(linhas[0].keys()))
        escritor.writeheader()
        escritor.writerows(linhas)
    print(f"  tabela -> results-by-sprints/sprint04/{destino.name}")
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
        # Gradientes e probabilidades minúsculos não podem virar "0.0000".
        return f"{valor:.2e}" if 0 < abs(valor) < 1e-3 else f"{valor:.4f}"
    if isinstance(valor, int):
        return f"{valor:,}"
    return str(valor)


def estilizar_linhas(ax: plt.Axes, titulo: str, x: str, y: str) -> None:
    ax.set_title(titulo, color=TINTA, pad=8)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(axis="x", visible=False)


# --- medidas ---------------------------------------------------------------
def cronometrar(funcao, repeticoes: int = 5, aquecimento: int = 1) -> float:
    """Tempo médio de uma chamada, em milissegundos."""
    with torch.no_grad():
        for _ in range(aquecimento):
            funcao()
        inicio = time.perf_counter()
        for _ in range(repeticoes):
            funcao()
        fim = time.perf_counter()
    return (fim - inicio) / repeticoes * 1000


def ms_por_token_gerado(modelo: GPTModel, ids: torch.Tensor, novos: int = 20) -> float:
    """Tempo médio, em ms, para gerar cada token (greedy, sem cache)."""
    tempo = cronometrar(
        lambda: gerar(modelo, ids, novos, modelo.cfg["context_length"]),
        repeticoes=2,
        aquecimento=1,
    )
    return tempo / novos


def entropia_proximo_token(logits: torch.Tensor) -> float:
    """Entropia média da distribuição do próximo token, normalizada por ``log V``.

    ``logits`` tem forma ``(..., V)``. Vale 1 para uma distribuição uniforme sobre
    todo o vocabulário e 0 quando toda a probabilidade está em um único token.
    """
    log_p = torch.log_softmax(logits.detach().float(), dim=-1)
    h = -(log_p.exp() * log_p).sum(dim=-1)
    return float(h.mean() / math.log(logits.shape[-1]))


def prob_maxima(logits: torch.Tensor) -> float:
    """Probabilidade média do token mais provável."""
    return float(torch.softmax(logits.detach().float(), dim=-1).max(dim=-1).values.mean())
