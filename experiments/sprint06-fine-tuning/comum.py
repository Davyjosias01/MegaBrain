"""Utilidades compartilhadas pelos experimentos da Sprint 6.

Centraliza o acesso aos módulos de ``src/``, a semente, o dispositivo, as
métricas sobre o texto gerado e a gravação de tabelas e figuras.

Todas as figuras e tabelas são gravadas em ``results-by-sprints/sprint06/``.
Os pesos pré-treinados e os checkpoints do fine-tuning ficam em ``modelos/``,
que não é versionada.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

# --- caminhos --------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[2]
SRC = RAIZ / "src"
RESULTADOS = RAIZ / "results-by-sprints" / "sprint06"
RESULTADOS.mkdir(parents=True, exist_ok=True)
CHECKPOINT_SPRINT5 = RAIZ / "results-by-sprints" / "sprint05" / "base-modelo.pt"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tokenizacao import TokenizadorBPE  # noqa: E402

# --- reprodutibilidade e dispositivo ----------------------------------------
SEMENTE = 123
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
CONTEXTO = "Every effort moves you"
TOKENIZADOR = TokenizadorBPE()


def semear(semente: int = SEMENTE) -> None:
    """Fixa a semente global antes de cada configuração medida."""
    torch.manual_seed(semente)


def cabecalho(titulo: str) -> None:
    print("\n" + "=" * 74)
    print(titulo)
    print("=" * 74)


# --- métricas sobre o texto gerado ------------------------------------------
def distinct_n(ids: list[int], n: int) -> float:
    """Fração de n-gramas distintos entre os gerados: 1 = nenhuma repetição."""
    ngramas = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(ngramas)) / len(ngramas) if ngramas else 0.0


def repeticao_n(ids: list[int], n: int = 3) -> float:
    """Fração dos n-gramas que já tinham aparecido antes no mesmo texto."""
    vistos, repetidos, total = set(), 0, 0
    for i in range(len(ids) - n + 1):
        g = tuple(ids[i:i + n])
        repetidos += g in vistos
        vistos.add(g)
        total += 1
    return repetidos / total if total else 0.0


# --- gravação ---------------------------------------------------------------
def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def salvar_tabela(linhas: list[dict], nome: str) -> Path:
    destino = RESULTADOS / f"{nome}.csv"
    with open(destino, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        for linha in linhas:
            w.writerow({k: _fmt(v) for k, v in linha.items()})
    print(f"  tabela -> results-by-sprints/sprint06/{destino.name}")
    return destino


def imprimir_tabela(linhas: list[dict]) -> None:
    colunas = list(linhas[0])
    larguras = [max(len(c), max(len(_fmt(l[c])) for l in linhas)) for c in colunas]
    print("  " + "  ".join(c.ljust(w) for c, w in zip(colunas, larguras)))
    print("  " + "  ".join("-" * w for w in larguras))
    for l in linhas:
        print("  " + "  ".join(_fmt(l[c]).ljust(w) for c, w in zip(colunas, larguras)))


def salvar_figura(fig, nome: str) -> Path:
    destino = RESULTADOS / f"{nome}.png"
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    print(f"  figura -> results-by-sprints/sprint06/{destino.name}")
    return destino


__all__ = [
    "RAIZ", "RESULTADOS", "CHECKPOINT_SPRINT5", "SEMENTE", "DISPOSITIVO", "CONTEXTO", "TOKENIZADOR",
    "semear", "cabecalho", "distinct_n", "repeticao_n",
    "salvar_tabela", "imprimir_tabela", "salvar_figura", "plt",
]
