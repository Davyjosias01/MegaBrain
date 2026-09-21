"""Executa os oito experimentos da Sprint 3 em sequência.

Grava todas as figuras e tabelas em ``results-by-sprints/sprint03/`` e o registro
completo do console em ``results-by-sprints/sprint03/execucao.log``.

Uso, a partir deste diretório::

    python executar_todos.py
"""

from __future__ import annotations

import platform
import sys
from datetime import datetime

import torch

from comum import RESULTADOS, SEMENTE, cabecalho

import exp01_dimensao_embedding
import exp02_numero_de_heads
import exp03_dimensao_de_head
import exp04_self_vs_multihead
import exp05_visualizacao_matriz
import exp06_com_e_sem_escala
import exp07_mascara_causal
import exp08_sequencias_de_entrada

EXPERIMENTOS = [
    exp01_dimensao_embedding,
    exp02_numero_de_heads,
    exp03_dimensao_de_head,
    exp04_self_vs_multihead,
    exp05_visualizacao_matriz,
    exp06_com_e_sem_escala,
    exp07_mascara_causal,
    exp08_sequencias_de_entrada,
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
    cabecalho("Sprint 3 — execução dos experimentos de atenção")
    print(f"  data           : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  python         : {platform.python_version()}")
    print(f"  torch          : {torch.__version__}")
    print(f"  dispositivo    : cpu (todos os experimentos)")
    print(f"  cuda disponível: {torch.cuda.is_available()}")
    print(f"  threads torch  : {torch.get_num_threads()}")
    print(f"  semente        : {SEMENTE}")
    print(f"  processador    : {platform.processor() or platform.machine()}")
    print(f"  sistema        : {platform.platform()}")


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
        print(f"  artefatos em results-by-sprints/sprint03/")
    finally:
        sys.stdout = original
        log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
