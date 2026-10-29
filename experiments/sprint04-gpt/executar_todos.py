"""Executa os oito experimentos da Sprint 4 em sequência.

Grava todas as figuras e tabelas em ``results-by-sprints/sprint04/`` e o registro
completo do console em ``results-by-sprints/sprint04/execucao.log``.

Uso, a partir deste diretório::

    python executar_todos.py
"""

from __future__ import annotations

import platform
import sys
from datetime import datetime

import torch

from comum import CONFIG_BASE, RESULTADOS, SEMENTE, cabecalho

import exp01_numero_de_blocos
import exp02_numero_de_heads
import exp03_dimensao_embedding
import exp04_dimensao_feedforward
import exp05_dropout
import exp06_tamanho_do_contexto
import exp07_residuais_e_layernorm
import exp08_estrategias_de_geracao

EXPERIMENTOS = [
    exp01_numero_de_blocos,
    exp02_numero_de_heads,
    exp03_dimensao_embedding,
    exp04_dimensao_feedforward,
    exp05_dropout,
    exp06_tamanho_do_contexto,
    exp07_residuais_e_layernorm,
    exp08_estrategias_de_geracao,
]


class _Espelho:
    """Escreve simultaneamente no console e no arquivo de log."""

    def __init__(self, *destinos) -> None:
        self.destinos = destinos

    def write(self, texto: str) -> int:
        for d in self.destinos:
            d.write(texto)
        return len(texto)

    def flush(self) -> None:
        for d in self.destinos:
            d.flush()

    def reconfigure(self, **kwargs) -> None:  # tolerado por comum.cabecalho
        pass


def _ambiente() -> None:
    cabecalho("Sprint 4 — execução dos experimentos da arquitetura GPT")
    print(f"  data           : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  python         : {platform.python_version()}")
    print(f"  torch          : {torch.__version__}")
    print(f"  dispositivo    : cpu (todos os experimentos)")
    print(f"  threads torch  : {torch.get_num_threads()}")
    print(f"  semente        : {SEMENTE}")
    print(f"  processador    : {platform.processor() or platform.machine()}")
    print(f"  sistema        : {platform.platform()}")
    print(f"  config-base    : {CONFIG_BASE}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    log = (RESULTADOS / "execucao.log").open("w", encoding="utf-8")
    original = sys.stdout
    sys.stdout = _Espelho(original, log)
    try:
        _ambiente()
        for modulo in EXPERIMENTOS:
            modulo.executar()
        cabecalho("Concluído")
        print("  artefatos em results-by-sprints/sprint04/")
    finally:
        sys.stdout = original
        log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
