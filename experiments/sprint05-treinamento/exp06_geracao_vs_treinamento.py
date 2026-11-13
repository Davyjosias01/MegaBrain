"""Experimento 06 — efeito da decodificação × efeito do treinamento.

**O que varia:** os **pesos** — o mesmo GPT-2 124M (contexto 256) com pesos
aleatórios, recém-inicializado com a semente 123, e após o treino base de 10
épocas (checkpoint) — combinados com as **mesmas** estratégias de geração do
experimento 05.
**O que fica fixo:** arquitetura, contexto inicial, 50 tokens, 5 sementes.

**Pergunta:** o que, no texto gerado, vem da regra de decodificação e o que
vem do que o modelo aprendeu?

A estratégia de decodificação só *reordena a escolha* dentro da
distribuição que o modelo produz. Com pesos aleatórios a distribuição é quase
uniforme sobre 50.257 tokens: qualquer estratégia produz tokens sem relação
entre si, e a temperatura ou o top-k mudam pouco o resultado. Depois do
treino a distribuição fica concentrada nos tokens do conto, e a mesma
estratégia passa a produzir palavras, pontuação e frases. Comparar as duas
colunas isola o efeito dos pesos; comparar as linhas de uma coluna isola o
efeito da decodificação.
"""

from __future__ import annotations

import torch

from comum import (
    CHECKPOINT_BASE,
    DISPOSITIVO,
    SEMENTE,
    cabecalho,
    imprimir_tabela,
    plt,
    salvar_figura,
    salvar_tabela,
)
from checkpoint import carregar_checkpoint
from exp05_estrategias_de_geracao import avaliar_estrategias, gravar_amostras
from gpt import GPTModel


def executar() -> list[dict]:
    cabecalho("Experimento 06 — geração × treinamento")
    treinado, dados = carregar_checkpoint(CHECKPOINT_BASE, DISPOSITIVO)
    linhas_t, _ = avaliar_estrategias(treinado, "treinado")
    del treinado

    torch.manual_seed(SEMENTE)
    sem_treino = GPTModel(dados["cfg_modelo"]).to(DISPOSITIVO).eval()
    linhas_s, amostras_s = avaliar_estrategias(sem_treino, "sem treino")
    del sem_treino

    linhas = linhas_s + linhas_t
    imprimir_tabela(linhas)
    salvar_tabela(linhas, "exp06-geracao-vs-treinamento")
    gravar_amostras(amostras_s, "exp06-amostras-sem-treino",
                    "Estratégias de geração — mesmo modelo, pesos aleatórios (sem treino)")
    for rotulo, texto in amostras_s[:3]:
        print(f"  sem treino [{rotulo}]: {texto!r}"[:150])
    _grafico(linhas_s, linhas_t)
    return linhas


def _grafico(sem: list[dict], com: list[dict]) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    rotulos = [l["estrategia"] for l in sem]
    x = range(len(rotulos))
    for ax, chave, titulo in [
        (a1, "distinct_2", "Diversidade (distinct-2)"),
        (a2, "copia_do_treino_8", "Cópia literal do treino (8-gramas)"),
    ]:
        ax.bar([k - 0.2 for k in x], [l[chave] for l in sem], 0.4, label="sem treino")
        ax.bar([k + 0.2 for k in x], [l[chave] for l in com], 0.4, label="treinado")
        ax.set_xticks(list(x), rotulos, rotation=35, ha="right", fontsize=8)
        ax.set_title(titulo)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3, axis="y")
        ax.legend()
    a1.set_ylabel("fração (média de 5 amostras)")
    salvar_figura(fig, "exp06-geracao-vs-treinamento")


if __name__ == "__main__":
    executar()
