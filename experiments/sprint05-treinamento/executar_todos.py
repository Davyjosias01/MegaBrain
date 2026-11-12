"""Executa os experimentos da Sprint 5 em sequência.

Grava todas as figuras e tabelas em ``results-by-sprints/sprint05/`` e o registro
completo do console em ``results-by-sprints/sprint05/execucao.log``.

E5 e E6 precisam do checkpoint do treino base; se ele não existir,
``treinar_base.py`` é executado antes.

Uso, a partir deste diretório::

    uv run python executar_todos.py
"""

from __future__ import annotations

import importlib
import platform
import sys
import time
from datetime import datetime

import torch

from comum import CHECKPOINT_BASE, DISPOSITIVO, RESULTADOS, SEMENTE, cabecalho

EXPERIMENTOS = [
    "exp01_numero_de_epocas",
    "exp02_taxa_de_aprendizado",
    "exp03_configuracao_de_treino",
    "exp04_tamanho_do_modelo",
    "exp05_estrategias_de_geracao",
    "exp06_geracao_vs_treinamento",
]


class _Espelho:
    """Escreve simultaneamente no console e no arquivo de log."""

    def __init__(self, *destinos):
        self.destinos = destinos

    def write(self, texto):
        for d in self.destinos:
            d.write(texto)

    def flush(self):
        for d in self.destinos:
            d.flush()


def _ambiente() -> None:
    cabecalho("Sprint 5 — execução dos experimentos de treinamento")
    print(f"  data           : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  python         : {platform.python_version()}")
    print(f"  torch          : {torch.__version__}")
    gpu = f" ({torch.cuda.get_device_name(0)})" if DISPOSITIVO == "cuda" else ""
    print(f"  dispositivo    : {DISPOSITIVO}{gpu}")
    print(f"  semente        : {SEMENTE}")
    print(f"  processador    : {platform.processor()}")
    print(f"  sistema        : {platform.platform()}")


def main(selecao: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    nomes = [n for n in EXPERIMENTOS if not selecao or any(s in n for s in selecao)]
    with open(RESULTADOS / "execucao.log", "w", encoding="utf-8") as log:
        antigo = sys.stdout
        sys.stdout = _Espelho(antigo, log)
        try:
            _ambiente()
            if any(n.startswith(("exp05", "exp06")) for n in nomes) and not CHECKPOINT_BASE.exists():
                print("\n  checkpoint base ausente: executando treinar_base.py")
                importlib.import_module("treinar_base").main()
            inicio = time.perf_counter()
            for nome in nomes:
                importlib.import_module(nome).executar()
            print(f"\n  tempo total: {time.perf_counter() - inicio:.1f} s")
            print("  artefatos em results-by-sprints/sprint05/")
        finally:
            sys.stdout = antigo


if __name__ == "__main__":
    main(sys.argv[1:])
