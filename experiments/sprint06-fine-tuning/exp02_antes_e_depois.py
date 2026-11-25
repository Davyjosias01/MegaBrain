"""Experimento 02 — o classificador antes e depois do fine-tuning.

**O que varia:** o treino. O mesmo GPT-2 124M com cabeça de duas classes é
avaliado logo depois de receber a cabeça nova (pesos pré-treinados no corpo,
cabeça aleatória) e depois das 5 épocas de ajuste (``treinar_classificador.py``).
**O que fica fixo:** arquitetura, pesos pré-treinados de partida, o conjunto de
**teste** — 262 mensagens que não participaram do treino nem da escolha de
nenhuma configuração — e o preparo das entradas.

**Pergunta:** o que o fine-tuning muda na capacidade de executar a tarefa?

Antes do ajuste o corpo do modelo já "entende" inglês, mas a cabeça que lê
esse entendimento é uma projeção aleatória: a saída não tem relação com spam e
a acurácia fica perto dos 50% de um palpite. O ajuste não ensina a língua ao
modelo — ensina a cabeça e o último bloco a extrair, do que o modelo já
representa, a informação que separa as duas classes.

Além da acurácia, o experimento mede precisão, revocação e F1 (com *spam* como
classe positiva) e lista as mensagens de teste classificadas errado, que são o
material para a análise qualitativa.
"""

from __future__ import annotations

import gc

import pandas as pd

from comum import (
    CHECKPOINT_CLASSIFICADOR,
    DISPOSITIVO,
    RESULTADOS,
    TOKENIZADOR,
    cabecalho,
    imprimir_tabela,
    novo_classificador,
    plt,
    salvar_figura,
    salvar_tabela,
)
from classificacao import (
    NOMES_DAS_CLASSES,
    PASTA_SPAM,
    carregar_classificador,
    classificar_texto,
    criar_loaders_classificacao,
    metricas_de_classificacao,
    prever_loader,
)

# Mensagens escritas para o experimento: não estão na base.
MENSAGENS_NOVAS = [
    "You are a winner you have been specially selected to receive $1000 cash or a $2000 award.",
    "Hey, just wanted to check if we're still on for dinner tonight? Let me know!",
    "URGENT! Your account has been suspended. Reply with your PIN to restore access now.",
    "Can you send me the notes from today's class when you get home?",
]


def executar() -> list[dict]:
    cabecalho("Experimento 02 — antes × depois do fine-tuning (conjunto de teste)")
    if not CHECKPOINT_CLASSIFICADOR.exists():
        raise FileNotFoundError(
            "checkpoint do classificador ausente: execute treinar_classificador.py antes."
        )
    _, _, loader_teste = criar_loaders_classificacao(tokenizador=TOKENIZADOR)
    max_length = loader_teste.dataset.max_length
    teste = pd.read_csv(PASTA_SPAM / "teste.csv")

    # Um modelo de cada vez na memória: são 124 milhões de parâmetros cada.
    linhas, matrizes, novas = [], {}, {}
    previstos = alvos = probs_erros = None
    for rotulo, criar in [("antes do fine-tuning", novo_classificador),
                          ("depois do fine-tuning", _carregar_ajustado)]:
        modelo = criar()
        m = metricas_de_classificacao(loader_teste, modelo, DISPOSITIVO)
        matrizes[rotulo] = m
        linhas.append({
            "modelo": rotulo, "exemplos": len(teste),
            "acuracia": m["acuracia"], "precisao": m["precisao"],
            "revocacao": m["revocacao"], "f1": m["f1"],
            "vp": m["vp"], "fp": m["fp"], "vn": m["vn"], "fn": m["fn"],
        })
        novas[rotulo] = [
            classificar_texto(t, modelo, TOKENIZADOR, DISPOSITIVO, max_length) for t in MENSAGENS_NOVAS
        ]
        if criar is _carregar_ajustado:
            previstos, alvos = prever_loader(loader_teste, modelo, DISPOSITIVO)
            erros = [i for i, (p, a) in enumerate(zip(previstos, alvos)) if p != a]
            probs_erros = {
                i: classificar_texto(teste.texto[i], modelo, TOKENIZADOR, DISPOSITIVO, max_length)[1]
                for i in erros
            }
        del modelo
        gc.collect()
    imprimir_tabela(linhas)
    salvar_tabela(linhas, "exp02-antes-e-depois")

    destino = RESULTADOS / "exp02-erros-e-exemplos.txt"
    with open(destino, "w", encoding="utf-8") as f:
        f.write("Classificador de spam antes e depois do fine-tuning\n\n")
        f.write(f"Mensagens novas (fora da base), max_length = {max_length}\n{'-' * 70}\n")
        for k, texto in enumerate(MENSAGENS_NOVAS):
            linha = [
                f"{rotulo.split()[0]}: {novas[rotulo][k][0]} (P(spam) = {novas[rotulo][k][1]['spam']:.3f})"
                for rotulo in novas
            ]
            f.write(f"{texto}\n  -> " + " | ".join(linha) + "\n\n")
            print(f"  {texto[:50]!r}... -> " + " | ".join(linha))
        f.write(f"\nErros no conjunto de teste depois do fine-tuning: {len(erros)} de {len(teste)}\n"
                f"{'-' * 70}\n")
        for i in erros:
            f.write(f"[verdadeiro: {NOMES_DAS_CLASSES[alvos[i]]} | previsto: "
                    f"{NOMES_DAS_CLASSES[previstos[i]]} | P(spam) = {probs_erros[i]['spam']:.3f}]\n"
                    f"{teste.texto[i]}\n\n")
    print(f"  erros e exemplos -> results-by-sprints/sprint06/{destino.name} ({len(erros)} erros)")
    _grafico(matrizes)
    return linhas


def _carregar_ajustado():
    modelo, dados = carregar_classificador(CHECKPOINT_CLASSIFICADOR, DISPOSITIVO)
    del dados  # o dicionário guarda uma segunda cópia dos pesos
    return modelo


def _grafico(matrizes: dict) -> None:
    fig, eixos = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, (rotulo, m) in zip(eixos, matrizes.items()):
        matriz = [[m["vn"], m["fp"]], [m["fn"], m["vp"]]]
        ax.imshow(matriz, cmap="Blues", vmin=0, vmax=m["vn"] + m["fp"] + m["fn"] + m["vp"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matriz[i][j]), ha="center", va="center", fontsize=16,
                        color="white" if matriz[i][j] > 100 else "black")
        ax.set_xticks([0, 1], NOMES_DAS_CLASSES)
        ax.set_yticks([0, 1], NOMES_DAS_CLASSES)
        ax.set_xlabel("classe prevista")
        ax.set_ylabel("classe verdadeira")
        ax.set_title(f"{rotulo}\nacurácia {m['acuracia']:.1%} | F1 {m['f1']:.3f}", fontsize=10)
    salvar_figura(fig, "exp02-antes-e-depois")


if __name__ == "__main__":
    executar()
