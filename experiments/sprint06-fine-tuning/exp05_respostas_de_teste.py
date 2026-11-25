"""Experimento 05 — respostas às instruções de teste, antes e depois do fine-tuning.

**O que varia:** os pesos. O mesmo ``GPTModel`` responde às instruções do
conjunto de **teste** com os pesos pré-treinados do GPT-2 124M (antes) e com
os pesos ajustados por ``treinar_instrucoes.py`` (depois).
**O que fica fixo:** arquitetura, formato do prompt, estratégia de geração
(*greedy*) e limite de tokens gerados.

**Pergunta:** o que o fine-tuning de instruções muda no comportamento do
modelo diante de um pedido?

O modelo pré-treinado já sabe inglês, mas foi treinado para *continuar* texto,
não para *atender* a um pedido. Diante do prompt, a continuação mais provável
costuma ser repetir a instrução, emendar outra parecida ou seguir escrevendo
sem relação com a tarefa — e ele não para, porque nunca viu que depois da
resposta vem ``<|endoftext|>``. Depois do ajuste, o esperado é uma resposta
curta, no formato das respostas da base, terminada pelo próprio modelo.

Este script só **gera** as respostas e as grava em
``results-by-sprints/sprint06/exp05-respostas-de-teste.json``, uma entrada por
exemplo de teste. A medição fica no experimento 06, que pode ser refeito sem
gerar tudo de novo.
"""

from __future__ import annotations

import gc
import json
import time

from comum import (
    CHECKPOINT_INSTRUCOES,
    DISPOSITIVO,
    RESULTADOS,
    TOKENIZADOR,
    TREINO_INSTRUCOES,
    cabecalho,
)
from checkpoint import carregar_checkpoint
from instrucoes import preparar_instrucoes, responder
from pesos_pretreinados import carregar_gpt2_pretreinado

MAX_NOVOS_TOKENS = 48
ARQUIVO = RESULTADOS / "exp05-respostas-de-teste.json"


def _ajustado():
    modelo, dados = carregar_checkpoint(CHECKPOINT_INSTRUCOES, DISPOSITIVO)
    del dados  # o dicionário guarda uma segunda cópia dos pesos
    return modelo


def executar() -> list[dict]:
    cabecalho("Experimento 05 — respostas às instruções de teste (antes × depois)")
    if not CHECKPOINT_INSTRUCOES.exists():
        raise FileNotFoundError(
            "checkpoint de instruções ausente: execute treinar_instrucoes.py antes."
        )
    _, _, teste = preparar_instrucoes()
    registros = [dict(exemplo) for exemplo in teste]

    # Um modelo de cada vez na memória.
    for campo, criar in [
        ("resposta_antes", lambda: carregar_gpt2_pretreinado(TREINO_INSTRUCOES["tamanho"], dispositivo=DISPOSITIVO)),
        ("resposta_depois", _ajustado),
    ]:
        modelo = criar()
        inicio = time.perf_counter()
        for registro in registros:
            registro[campo] = responder(
                registro, modelo, TOKENIZADOR, dispositivo=DISPOSITIVO,
                max_novos_tokens=MAX_NOVOS_TOKENS)
        tokens = sum(len(TOKENIZADOR.encode(r[campo])) for r in registros) / len(registros)
        print(f"  {campo}: {len(registros)} respostas em {(time.perf_counter() - inicio) / 60:.1f} min, "
              f"{tokens:.1f} tokens em média (limite {MAX_NOVOS_TOKENS})")
        del modelo
        gc.collect()

    ARQUIVO.write_text(json.dumps(registros, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  respostas -> results-by-sprints/sprint06/{ARQUIVO.name}")
    for registro in registros[:3]:
        print(f"\n  instrução: {registro['instruction']} {registro['input']}".rstrip())
        print(f"    esperada: {registro['output']!r}")
        print(f"    antes   : {registro['resposta_antes'][:110]!r}")
        print(f"    depois  : {registro['resposta_depois'][:110]!r}")
    return registros


if __name__ == "__main__":
    executar()
