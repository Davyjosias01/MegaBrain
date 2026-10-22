"""Executa os oito experimentos da Sprint 2 em sequência.

Grava todas as figuras e tabelas em ``results-by-sprints/sprint02/`` e o registro
completo do console em ``results-by-sprints/sprint02/execucao.log``.

Uso, a partir deste diretório::

    python executar_todos.py
"""

from __future__ import annotations

import platform
import sys
from datetime import datetime

import tiktoken
import torch

from comum import RESULTADOS, SEMENTE, cabecalho

import exp01_tamanho_de_contexto
import exp02_tamanho_de_lote
import exp03_dimensao_embedding
import exp04_sequencias_de_entrada
import exp05_tokens_por_texto
import exp06_contexto_e_amostras
import exp07_dimensoes_do_pipeline
import exp08_embeddings_posicionais

EXPERIMENTOS = [
    exp01_tamanho_de_contexto,
    exp02_tamanho_de_lote,
    exp03_dimensao_embedding,
    exp04_sequencias_de_entrada,
    exp05_tokens_por_texto,
    exp06_contexto_e_amostras,
    exp07_dimensoes_do_pipeline,
    exp08_embeddings_posicionais,
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
    cabecalho("Sprint 2 — execução dos experimentos de processamento de texto")
    print(f"  data           : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  python         : {platform.python_version()}")
    print(f"  torch          : {torch.__version__}")
    print(f"  tiktoken       : {tiktoken.__version__}")
    print(f"  dispositivo    : cpu (todos os experimentos)")
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
        print("  artefatos em results-by-sprints/sprint02/")
    finally:
        sys.stdout = original
        log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
