"""Experimento 06 — atenção com e sem a divisão por √d_k.

**O que varia:** ``d_k`` em {2, 8, 32, 128, 512, 2048} e a presença da escala.
**O que fica fixo:** 64 tokens, queries e keys sorteadas de uma normal padrão,
semente 123.

**Pergunta:** o que exatamente a escala evita?

A página 84 faz uma afirmação precisa, e este experimento a verifica em três
medidas:

1. **Desvio-padrão dos scores.** O produto escalar de dois vetores com entradas
   independentes de variância 1 tem variância ``d_k``, logo desvio-padrão
   ``√d_k``. Dividir por ``√d_k`` traz esse desvio de volta para perto de 1,
   qualquer que seja a dimensão.

2. **Concentração do softmax.** Scores com desvio grande fazem o softmax se
   aproximar de uma função degrau: quase toda a massa vai para um único token.
   Medimos o peso máximo por linha e a entropia normalizada.

3. **Magnitude do gradiente.** É a consequência que importa para o treinamento.
   Quando o softmax satura, sua derivada tende a zero e o gradiente que volta
   pelos scores praticamente desaparece — os pesos ``W_q`` e ``W_k`` deixam de
   ser atualizados. Medimos a norma do gradiente que chega aos scores, fazendo a
   retropropagação de uma perda simples sobre os pesos de atenção.

A terceira medida é o núcleo do argumento: sem escala, o problema não é estético,
é que o modelo para de aprender.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from comum import (
    CATEGORICA,
    MARCADORES,
    cabecalho,
    entropia_normalizada,
    estilizar_linhas,
    imprimir_tabela,
    salvar_figura,
    salvar_tabela,
    semear,
)

from attention import scaled_dot_product_attention

DIMENSOES_DE_CHAVE = [2, 8, 32, 128, 512, 2048]
NUM_TOKENS = 64
REPETICOES = 10  # sorteios independentes de Q e K, para reduzir o ruído


def _uma_amostra(d_k: int, escalar: bool, semente: int) -> dict:
    semear(semente)
    queries = torch.randn(NUM_TOKENS, d_k, requires_grad=True)
    keys = torch.randn(NUM_TOKENS, d_k, requires_grad=True)
    values = torch.randn(NUM_TOKENS, 16)

    scores = queries @ keys.T
    if escalar:
        scores = scores / d_k**0.5
    scores.retain_grad()

    pesos = torch.softmax(scores, dim=-1)
    # Perda arbitrária, mas fixa: a soma dos pesos da diagonal. Serve apenas
    # para gerar um gradiente e medir o que sobrevive à passagem pelo softmax.
    pesos.diagonal().sum().backward()

    p = pesos.detach()
    # Traço da jacobiana do softmax, soma_j p_j(1 - p_j). É a medida direta de
    # quanto gradiente o softmax deixa passar, sem depender da perda escolhida:
    # vale (1 - 1/n) quando a distribuição é uniforme e tende a 0 quando satura.
    sensibilidade = float((p * (1 - p)).sum(dim=-1).mean())

    _, pesos_ref = scaled_dot_product_attention(
        queries.detach(), keys.detach(), values, scale=escalar
    )

    return {
        "desvio_padrao_scores": float(scores.detach().std()),
        "peso_maximo_medio": float(p.max(dim=-1).values.mean()),
        "entropia_normalizada": entropia_normalizada(pesos_ref, causal=False),
        "sensibilidade_do_softmax": sensibilidade,
        "norma_gradiente_scores": float(scores.grad.norm()),
    }


def _medir(d_k: int, escalar: bool) -> dict:
    amostras = [
        _uma_amostra(d_k, escalar, semente) for semente in range(REPETICOES)
    ]
    medio = {
        chave: sum(a[chave] for a in amostras) / len(amostras)
        for chave in amostras[0]
    }
    return {"d_k": d_k, "escala": "com" if escalar else "sem", **medio}


def executar() -> list[dict]:
    cabecalho("Experimento 06 — atenção com e sem a divisão por raiz de d_k")
    print(
        f"  num_tokens={NUM_TOKENS}  Q e K ~ N(0, 1)  "
        f"média de {REPETICOES} sorteios por configuração"
    )

    linhas = [
        _medir(d_k, escalar)
        for d_k in DIMENSOES_DE_CHAVE
        for escalar in (True, False)
    ]

    imprimir_tabela(linhas)
    salvar_tabela(linhas, "exp06-com-e-sem-escala")

    sem_escala = [l for l in linhas if l["escala"] == "sem"][-1]
    com_escala = [l for l in linhas if l["escala"] == "com"][-1]
    print(
        f"\n  Em d_k = {sem_escala['d_k']}:"
        f"\n    sem escala — peso máximo médio {sem_escala['peso_maximo_medio']:.4f}, "
        f"sensibilidade do softmax {sem_escala['sensibilidade_do_softmax']:.3e}, "
        f"norma do gradiente {sem_escala['norma_gradiente_scores']:.3e}"
        f"\n    com escala — peso máximo médio {com_escala['peso_maximo_medio']:.4f}, "
        f"sensibilidade do softmax {com_escala['sensibilidade_do_softmax']:.3e}, "
        f"norma do gradiente {com_escala['norma_gradiente_scores']:.3e}"
    )

    _grafico(linhas)
    return linhas


def _grafico(linhas: list[dict]) -> None:
    com = [l for l in linhas if l["escala"] == "com"]
    sem = [l for l in linhas if l["escala"] == "sem"]
    d = [l["d_k"] for l in com]

    fig, eixos = plt.subplots(1, 3, figsize=(11.5, 3.7))

    paineis = [
        ("desvio_padrao_scores", "Desvio-padrão dos scores", "desvio-padrão", True),
        ("peso_maximo_medio", "Concentração do softmax", "peso máximo médio", False),
        (
            "sensibilidade_do_softmax",
            "Gradiente que o softmax deixa passar",
            "traço da jacobiana",
            True,
        ),
    ]

    for ax, (chave, titulo, rotulo_y, log_y) in zip(eixos, paineis):
        ax.plot(
            d,
            [l[chave] for l in com],
            color=CATEGORICA[0],
            marker=MARCADORES[0],
            label="com escala",
        )
        ax.plot(
            d,
            [l[chave] for l in sem],
            color=CATEGORICA[1],
            marker=MARCADORES[1],
            label="sem escala",
        )
        ax.set_xscale("log", base=2)
        if log_y:
            ax.set_yscale("log")
        estilizar_linhas(ax, titulo, "d_k (dimensão das keys)", rotulo_y)

    eixos[1].set_ylim(0, 1.05)
    eixos[0].legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "Experimento 06 — sem a divisão por √d_k o softmax satura e o gradiente "
        "desaparece",
        y=1.04,
    )
    fig.tight_layout()
    salvar_figura(fig, "exp06-com-e-sem-escala")


if __name__ == "__main__":
    executar()
