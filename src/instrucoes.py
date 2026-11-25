"""Fine-tuning do GPT para seguir instruções — Sprint 6 (Capítulo 7).

Na classificação (``classificacao.py``) o modelo foi adaptado para devolver
uma classe. Aqui ele continua fazendo o que sempre fez — prever o próximo
token — mas é ajustado em textos com um formato fixo, em que uma **instrução**
é seguida da **resposta** esperada. Depois do ajuste, quando recebe só a
instrução nesse formato, a continuação mais provável é uma resposta.

Nada muda na arquitetura nem na função de perda: é o mesmo treinamento da
Sprint 5, com outros dados. O trabalho deste módulo está na preparação desses
dados.

Etapas, na ordem das seções deste arquivo:

1. **dados** — carregar a base, formatar cada exemplo como um prompt e
   dividir em treino, validação e teste;
2. **lotes** — tokenizar e agrupar exemplos de tamanhos diferentes.

A base é a do capítulo: ``instruction-data.json``, com 1.100 exemplos em
inglês, do repositório do livro (github.com/rasbt/LLMs-from-scratch, licença
Apache 2.0).
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

__all__ = [
    "PASTA_INSTRUCOES",
    "MARCADOR_RESPOSTA",
    "baixar_base_instrucoes",
    "carregar_instrucoes",
    "remover_instrucoes_repetidas",
    "formatar_entrada",
    "formatar_exemplo",
    "dividir_instrucoes",
]

RAIZ = Path(__file__).resolve().parent.parent
PASTA_INSTRUCOES = RAIZ / "sprints" / "dados" / "instrucoes"

URL_INSTRUCOES = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/"
    "ch07/01_main-chapter-code/instruction-data.json"
)

# Textos fixos do formato Alpaca. Ficam em inglês porque a base e o modelo
# pré-treinado são em inglês.
_ABERTURA = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request."
)
_MARCADOR_INSTRUCAO = "\n\n### Instruction:\n"
_MARCADOR_ENTRADA = "\n\n### Input:\n"
MARCADOR_RESPOSTA = "\n\n### Response:\n"


# ---------------------------------------------------------------------------
# 1. Dados: carga, formato do prompt e divisão
# ---------------------------------------------------------------------------
def baixar_base_instrucoes(pasta: Path | str = PASTA_INSTRUCOES) -> Path:
    """Baixa ``instruction-data.json`` uma única vez e devolve o caminho local."""
    destino = Path(pasta) / "instruction-data.json"
    if destino.exists():
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    requisicao = urllib.request.Request(URL_INSTRUCOES, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(requisicao) as resposta:
        destino.write_bytes(resposta.read())
    return destino


def carregar_instrucoes(caminho: Path | str | None = None) -> list[dict]:
    """Lê a base: uma lista de exemplos ``{"instruction", "input", "output"}``.

    - ``instruction`` — a tarefa, em linguagem natural;
    - ``input`` — o dado sobre o qual a tarefa age; vazio em cerca de dois
      terços dos exemplos, quando a instrução já contém tudo;
    - ``output`` — a resposta esperada.
    """
    caminho = Path(caminho) if caminho is not None else baixar_base_instrucoes()
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    for exemplo in dados:
        if set(exemplo) != {"instruction", "input", "output"}:
            raise ValueError(f"exemplo com campos inesperados: {sorted(exemplo)}")
    return dados


def remover_instrucoes_repetidas(dados: list[dict]) -> list[dict]:
    """Mantém a primeira ocorrência de cada par (instrução, entrada).

    A base tem 25 pares repetidos, com respostas redigidas de forma diferente.
    Sem esta etapa, a divisão do capítulo deixa 2 perguntas do teste e 4 da
    validação também no treino — o modelo seria avaliado em perguntas cuja
    resposta ele já viu.
    """
    vistos, unicos = set(), []
    for exemplo in dados:
        chave = (exemplo["instruction"], exemplo["input"])
        if chave not in vistos:
            vistos.add(chave)
            unicos.append(exemplo)
    return unicos


def formatar_entrada(exemplo: dict) -> str:
    """Monta o prompt no formato Alpaca, **sem** a resposta.

    ::

        Below is an instruction that describes a task. Write a response that
        appropriately completes the request.

        ### Instruction:
        <instrução>

        ### Input:
        <entrada>            <- bloco omitido quando a entrada é vazia

    Os marcadores ``### ...`` não são tokens especiais: são texto comum, que o
    tokenizador divide em tokens normais. O modelo aprende, pelo ajuste, que
    depois de ``### Response:`` vem a resposta — é isso que dá ao formato o
    seu papel.
    """
    texto = _ABERTURA + _MARCADOR_INSTRUCAO + exemplo["instruction"]
    if exemplo.get("input"):
        texto += _MARCADOR_ENTRADA + exemplo["input"]
    return texto


def formatar_exemplo(exemplo: dict) -> str:
    """Prompt completo de treino: entrada formatada + marcador + resposta esperada."""
    return formatar_entrada(exemplo) + MARCADOR_RESPOSTA + exemplo["output"]


def dividir_instrucoes(
    dados: list[dict], proporcao_treino: float = 0.85, proporcao_teste: float = 0.10
) -> tuple[list[dict], list[dict], list[dict]]:
    """Divide em treino, validação e teste (padrão 85% / 5% / 10%), na ordem da base.

    O teste é maior que a validação porque aqui a avaliação final é feita
    sobre respostas **geradas**, comparadas uma a uma com as esperadas; a
    validação só acompanha a perda durante o treino. O que sobra depois de
    treino e teste vai para a validação.

    Returns:
        ``(treino, validacao, teste)``.
    """
    if not 0 < proporcao_treino < 1 or not 0 <= proporcao_teste < 1 - proporcao_treino:
        raise ValueError("proporções inválidas: treino e teste devem somar menos de 1.")
    fim_treino = int(len(dados) * proporcao_treino)
    fim_teste = fim_treino + int(len(dados) * proporcao_teste)
    return dados[:fim_treino], dados[fim_teste:], dados[fim_treino:fim_teste]
