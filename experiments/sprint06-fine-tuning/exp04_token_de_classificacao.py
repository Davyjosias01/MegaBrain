"""Experimento 04 — qual token representa a mensagem.

**O que varia:** a posição da sequência cujos logits são usados como saída do
classificador: o **último** token (``posicao = -1``, configuração base) ou o
**primeiro** (``posicao = 0``).
**O que fica fixo:** todo o resto da configuração base — pesos pré-treinados,
último bloco treinável, dados, semente, épocas e otimizador.

**Pergunta:** faz diferença de qual token se lê a classe?

O modelo produz um vetor de saída para cada token, e qualquer um deles poderia
alimentar a cabeça de classificação. A diferença está na **máscara causal** da
Sprint 3: a atenção de uma posição só alcança ela mesma e as anteriores.

- A última posição enxerga a mensagem inteira.
- A primeira posição enxerga só o primeiro token. O vetor dela é uma função
  apenas da primeira palavra da mensagem; todo o resto é invisível para ela,
  por mais que o treino ajuste os pesos.

O classificador pelo primeiro token deve então ficar bem abaixo do outro, mas
acima de 50%: a primeira palavra sozinha já carrega algum sinal ("URGENT",
"FREE", "Congratulations" contra "Ok", "Hey", "I").

Em modelos com atenção bidirecional, como o BERT, a situação é a oposta: todo
token enxerga a sequência inteira e a convenção é classificar por um token
especial colocado no início.

A configuração do último token reaproveita o resultado de
``treinar_classificador.py`` quando ele existe.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from comum import (
    TOKENIZADOR,
    cabecalho,
    imprimir_tabela,
    plt,
    salvar_figura,
    salvar_tabela,
    treinar_configuracao,
)
from classificacao import PASTA_SPAM
from exp03_camadas_treinaveis import resultado_base


def teto_do_primeiro_token() -> float:
    """Acurácia de teste do melhor classificador possível que só vê o primeiro token.

    Para cada primeiro token visto no treino, responde a classe mais frequente
    entre as mensagens de treino que começam com ele; para um token nunca
    visto, a classe mais frequente do treino. É uma referência independente
    do modelo: mostra quanto da tarefa se resolve só com a primeira palavra.
    """
    treino = pd.read_csv(PASTA_SPAM / "treino.csv")
    teste = pd.read_csv(PASTA_SPAM / "teste.csv")
    primeiro = lambda texto: TOKENIZADOR.encode(texto)[0]  # noqa: E731
    votos: dict[int, Counter] = {}
    for texto, rotulo in zip(treino.texto, treino.rotulo):
        votos.setdefault(primeiro(texto), Counter())[rotulo] += 1
    padrao = Counter(treino.rotulo).most_common(1)[0][0]
    acertos = sum(
        (votos[primeiro(t)].most_common(1)[0][0] if primeiro(t) in votos else padrao) == r
        for t, r in zip(teste.texto, teste.rotulo)
    )
    return acertos / len(teste)


def executar() -> list[dict]:
    cabecalho("Experimento 04 — token usado na classificação")
    linhas = []
    for rotulo, posicao in [("último token", -1), ("primeiro token", 0)]:
        r = resultado_base() if posicao == -1 else None
        origem = "treinar_classificador.py" if r else "este experimento"
        if r is None:
            r = treinar_configuracao(posicao=posicao)
            r.pop("modelo"), r.pop("otimizador"), r.pop("loaders")
        linhas.append({
            "posicao": rotulo,
            "tokens_visiveis": "mensagem inteira" if posicao == -1 else "só o primeiro",
            "acuracia_validacao": r["acuracia_validacao"],
            "acuracia_teste": r["acuracia_teste"],
            "tempo_min": r["tempo_s"] / 60,
            "origem": origem,
        })
    teto = teto_do_primeiro_token()
    imprimir_tabela(linhas)
    print(f"  referência: melhor regra possível usando só o primeiro token = {teto:.2%} no teste")
    linhas.append({
        "posicao": "regra de contagem pelo primeiro token (sem modelo)",
        "tokens_visiveis": "só o primeiro",
        "acuracia_validacao": "-", "acuracia_teste": teto, "tempo_min": 0.0, "origem": "contagem no treino",
    })
    salvar_tabela(linhas, "exp04-token-de-classificacao")
    _grafico(linhas)
    return linhas


def _grafico(linhas: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    rotulos = ["último token\n(GPT ajustado)", "primeiro token\n(GPT ajustado)", "primeiro token\n(regra de contagem)"]
    valores = [l["acuracia_teste"] for l in linhas]
    barras = ax.bar(rotulos, valores, color=["#1f77b4", "#ff7f0e", "#888888"])
    for barra, v in zip(barras, valores):
        ax.annotate(f"{v:.1%}", (barra.get_x() + barra.get_width() / 2, v), ha="center", va="bottom")
    ax.axhline(0.5, color="lightgray", linewidth=1)
    ax.set_ylim(0.45, 1.03)
    ax.set_ylabel("acurácia no teste")
    ax.set_title("De qual posição se lê a classe")
    ax.grid(alpha=0.3, axis="y")
    salvar_figura(fig, "exp04-token-de-classificacao")


if __name__ == "__main__":
    executar()
