"""Executa os experimentos da Sprint 6 em sequência.

Grava as figuras e tabelas em ``results-by-sprints/sprint06/`` e o registro
do console em ``results-by-sprints/sprint06/execucao.log``.

Os experimentos 02 a 04 partem do treino base do classificador, e os 05 e 06,
do treino de instruções. Se o checkpoint correspondente não existir em
``modelos/``, o treino é executado antes.

Em CPU a sequência completa leva várias horas (o treino base do classificador
leva ~40 min, e os experimentos 03 e 04 refazem esse treino com outras
configurações). Para refazer só uma parte, passe filtros::

    uv run python executar_todos.py              # tudo
    uv run python executar_todos.py exp01        # só o experimento 01
    uv run python executar_todos.py exp05 exp06  # só os de instruções
"""

from __future__ import annotations

import importlib
import platform
import sys
import time
from datetime import datetime

import torch

from comum import (
    CHECKPOINT_CLASSIFICADOR,
    CHECKPOINT_INSTRUCOES,
    DISPOSITIVO,
    RESULTADOS,
    SEMENTE,
    cabecalho,
)

EXPERIMENTOS = [
    "exp01_aleatorio_vs_pretreinado",
    "exp02_antes_e_depois",
    "exp03_camadas_treinaveis",
    "exp04_token_de_classificacao",
    "exp05_respostas_de_teste",
    "exp06_avaliacao_das_respostas",
]

# Treino exigido por cada experimento: (prefixos, checkpoint, módulo do treino).
PRE_REQUISITOS = [
    (("exp02", "exp03", "exp04"), CHECKPOINT_CLASSIFICADOR, "treinar_classificador"),
    (("exp05",), CHECKPOINT_INSTRUCOES, "treinar_instrucoes"),
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
    cabecalho("Sprint 6 — execução dos experimentos de fine-tuning")
    print(f"  data           : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  python         : {platform.python_version()}")
    print(f"  torch          : {torch.__version__}")
    gpu = f" ({torch.cuda.get_device_name(0)})" if DISPOSITIVO == "cuda" else ""
    print(f"  dispositivo    : {DISPOSITIVO}{gpu}")
    print(f"  semente        : {SEMENTE}")
    print(f"  processador    : {platform.processor()}")
    print(f"  sistema        : {platform.platform()}")


def selecionar(selecao: list[str] | None) -> list[str]:
    """Experimentos cujo nome contém algum dos filtros (todos, sem filtros)."""
    return [n for n in EXPERIMENTOS if not selecao or any(s in n for s in selecao)]


def treinos_necessarios(nomes: list[str]) -> list[str]:
    """Treinos que precisam rodar antes, por falta do checkpoint."""
    return [
        modulo for prefixos, checkpoint, modulo in PRE_REQUISITOS
        if any(n.startswith(prefixos) for n in nomes) and not checkpoint.exists()
    ]


def main(selecao: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    nomes = selecionar(selecao)
    with open(RESULTADOS / "execucao.log", "w", encoding="utf-8") as log:
        antigo = sys.stdout
        sys.stdout = _Espelho(antigo, log)
        try:
            _ambiente()
            inicio = time.perf_counter()
            for modulo in treinos_necessarios(nomes):
                print(f"\n  checkpoint ausente: executando {modulo}.py")
                importlib.import_module(modulo).treinar()
            for nome in nomes:
                importlib.import_module(nome).executar()
            print(f"\n  tempo total: {(time.perf_counter() - inicio) / 60:.1f} min")
            print("  artefatos em results-by-sprints/sprint06/")
        finally:
            sys.stdout = antigo


if __name__ == "__main__":
    main(sys.argv[1:])
