"""Experimento 05 — estratégias de decodificação.

**O que varia:** a regra de escolha do próximo token (``src/geracao.py``):

- *greedy* — sempre o token mais provável (temperatura 0);
- amostragem pura — sorteio na distribuição do softmax (temperatura 1);
- temperatura 0,3, 0,7, 1,5 e 3,0 — divide os logits antes do softmax:
  abaixo de 1 concentra a distribuição, acima de 1 a achata;
- top-k 3, 10 e 50 (temperatura 1) — sorteia só entre os ``k`` mais prováveis;
- temperatura 1,4 com top-k 25 — a combinação usada na seção 5.3.

**O que fica fixo:** o **modelo** (checkpoint do treino base, GPT-2 124M após
10 épocas), o **contexto inicial** "Every effort moves you" e 50 novos tokens.
Cada estratégia estocástica é amostrada com 5 sementes.

**Pergunta:** como cada estratégia troca coerência por diversidade?

Métricas sobre os tokens gerados (média das 5 amostras):

- ``distinct_1`` / ``distinct_2`` — fração de tokens e de bigramas distintos
  (diversidade dentro do texto);
- ``repeticao_3`` — fração de trigramas que repetem um trigrama anterior
  (laços como "the the the");
- ``copia_do_treino_8`` — fração dos 8-gramas gerados que existem literalmente
  no texto de treinamento (memorização);
- ``textos_distintos`` — quantas das 5 sementes produziram textos diferentes.

A coerência é avaliada qualitativamente nas amostras gravadas em
``exp05-amostras.txt``.
"""

from __future__ import annotations

import statistics

import torch

from comum import (
    CHECKPOINT_BASE,
    CONTEXTO,
    DISPOSITIVO,
    RESULTADOS,
    TOKENIZADOR,
    cabecalho,
    copia_do_treino,
    distinct_n,
    imprimir_tabela,
    plt,
    repeticao_n,
    salvar_figura,
    salvar_tabela,
)
from checkpoint import carregar_checkpoint
from geracao import gerar, texto_para_ids

NOVOS_TOKENS = 50
SEMENTES = [123, 1, 2, 3, 4]

# (rótulo, temperatura, top_k)
ESTRATEGIAS = [
    ("greedy", 0.0, None),
    ("temperatura 0,3", 0.3, None),
    ("temperatura 0,7", 0.7, None),
    ("amostragem pura (T=1)", 1.0, None),
    ("temperatura 1,5", 1.5, None),
    ("temperatura 3,0", 3.0, None),
    ("top-k 3", 1.0, 3),
    ("top-k 10", 1.0, 10),
    ("top-k 50", 1.0, 50),
    ("T=1,4 + top-k 25 (livro)", 1.4, 25),
]


def avaliar_estrategias(modelo, rotulo_modelo: str) -> tuple[list[dict], list[tuple[str, str]]]:
    """Gera com cada estratégia e mede as métricas. Devolve (linhas, amostras)."""
    ids_contexto = texto_para_ids(CONTEXTO, TOKENIZADOR).to(DISPOSITIVO)
    n_contexto = ids_contexto.shape[1]
    linhas, amostras = [], []
    for rotulo, temperatura, top_k in ESTRATEGIAS:
        gerados = []
        for semente in SEMENTES:
            torch.manual_seed(semente)
            with torch.no_grad():
                saida = gerar(modelo, ids_contexto, NOVOS_TOKENS, modelo.cfg["context_length"],
                              temperatura=temperatura, top_k=top_k)
            gerados.append(saida[0, n_contexto:].tolist())
        media = lambda f: statistics.mean(f(g) for g in gerados)  # noqa: E731
        linhas.append({
            "modelo": rotulo_modelo,
            "estrategia": rotulo,
            "temperatura": temperatura,
            "top_k": top_k if top_k is not None else "-",
            "distinct_1": media(lambda g: distinct_n(g, 1)),
            "distinct_2": media(lambda g: distinct_n(g, 2)),
            "repeticao_3": media(lambda g: repeticao_n(g, 3)),
            "copia_do_treino_8": media(lambda g: copia_do_treino(g, 8)),
            "textos_distintos": len({tuple(g) for g in gerados}),
        })
        amostras.append((rotulo, TOKENIZADOR.decode(gerados[0])))
    return linhas, amostras


def gravar_amostras(amostras: list[tuple[str, str]], nome: str, titulo: str) -> None:
    with open(RESULTADOS / f"{nome}.txt", "w", encoding="utf-8") as f:
        f.write(f"{titulo}\nContexto: {CONTEXTO!r}, {NOVOS_TOKENS} tokens, semente {SEMENTES[0]}\n\n")
        for rotulo, texto in amostras:
            f.write(f"[{rotulo}]\n{CONTEXTO}{texto}\n\n")
    print(f"  amostras -> results-by-sprints/sprint05/{nome}.txt")


def executar() -> list[dict]:
    cabecalho("Experimento 05 — estratégias de geração (modelo treinado)")
    modelo, _ = carregar_checkpoint(CHECKPOINT_BASE, DISPOSITIVO)
    linhas, amostras = avaliar_estrategias(modelo, "treinado")
    del modelo
    imprimir_tabela(linhas)
    salvar_tabela(linhas, "exp05-estrategias-de-geracao")
    gravar_amostras(amostras, "exp05-amostras", "Estratégias de geração — modelo treinado (treino base)")
    for rotulo, texto in amostras:
        print(f"  [{rotulo}] {CONTEXTO + texto!r}"[:150])
    _grafico(linhas)
    return linhas


def _grafico(linhas: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    rotulos = [l["estrategia"] for l in linhas]
    x = range(len(linhas))
    largura = 0.2
    for i, (chave, nome) in enumerate([
        ("distinct_2", "diversidade (distinct-2)"),
        ("repeticao_3", "repetição (trigramas)"),
        ("copia_do_treino_8", "cópia do treino (8-gramas)"),
    ]):
        ax.bar([k + (i - 1) * largura for k in x], [l[chave] for l in linhas], largura, label=nome)
    ax.set_xticks(list(x), rotulos, rotation=30, ha="right")
    ax.set_ylabel("fração (média de 5 amostras)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Estratégias de geração com o mesmo modelo e o mesmo contexto")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    salvar_figura(fig, "exp05-estrategias-de-geracao")


if __name__ == "__main__":
    executar()
