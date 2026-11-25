"""Experimento 06 — avaliação das respostas geradas (*response evaluation*).

**O que se mede:** as respostas do experimento 05 — do modelo pré-treinado e
do modelo ajustado — contra as respostas esperadas do conjunto de teste.

**Pergunta:** quanto as respostas do modelo ajustado se aproximam das
esperadas, e o que as métricas automáticas deixam de ver?

Métricas (``src/instrucoes.py``):

- **correspondência exata** — a resposta tem as mesmas palavras da esperada,
  na mesma ordem, ignorando caixa e pontuação;
- **F1 de palavras** — sobreposição de palavras entre as duas respostas;
- **comprimento médio** — palavras geradas contra palavras esperadas; mostra se
  o modelo aprendeu a parar.

As duas primeiras medem **semelhança de texto**, não correção. Uma resposta
certa escrita de outra forma perde pontos; uma resposta errada que reaproveite
as palavras da pergunta ganha. Por isso o experimento também grava, para
leitura, as respostas de maior e de menor F1 e uma amostra do meio — é na
leitura que se vê se o erro é de conteúdo ou só de redação.

O capítulo avalia com um segundo LLM como juiz (nota de 0 a 100). Essa
alternativa não foi usada: exigiria um modelo pronto de terceiros.

Separa também os exemplos **com** e **sem** o campo de entrada, para ver se o
modelo aproveita a entrada quando ela existe.
"""

from __future__ import annotations

import json

from comum import RESULTADOS, cabecalho, imprimir_tabela, plt, salvar_figura, salvar_tabela
from instrucoes import avaliar_respostas, correspondencia_exata, f1_de_palavras

ARQUIVO = RESULTADOS / "exp05-respostas-de-teste.json"


def executar() -> list[dict]:
    cabecalho("Experimento 06 — avaliação das respostas")
    if not ARQUIVO.exists():
        raise FileNotFoundError("respostas ausentes: execute exp05_respostas_de_teste.py antes.")
    registros = json.loads(ARQUIVO.read_text(encoding="utf-8"))

    grupos = [
        ("todos", registros),
        ("com entrada", [r for r in registros if r["input"]]),
        ("sem entrada", [r for r in registros if not r["input"]]),
    ]
    linhas = []
    for modelo, campo in [("pré-treinado", "resposta_antes"), ("ajustado", "resposta_depois")]:
        for grupo, itens in grupos:
            m = avaliar_respostas([(r[campo], r["output"]) for r in itens])
            linhas.append({"modelo": modelo, "exemplos_de_teste": grupo, **m})
    imprimir_tabela(linhas)
    salvar_tabela(linhas, "exp06-avaliacao-das-respostas")

    for r in registros:
        r["f1"] = f1_de_palavras(r["resposta_depois"], r["output"])
        r["exata"] = correspondencia_exata(r["resposta_depois"], r["output"])
    ordenados = sorted(registros, key=lambda r: r["f1"])
    meio = len(ordenados) // 2
    destino = RESULTADOS / "exp06-amostra-para-leitura.txt"
    with open(destino, "w", encoding="utf-8") as f:
        f.write("Respostas do modelo ajustado no conjunto de teste, para leitura\n")
        f.write(f"{len(registros)} exemplos; F1 de palavras entre a resposta gerada e a esperada\n")
        for titulo, itens in [
            ("Menor F1", ordenados[:8]),
            ("F1 intermediário", ordenados[meio - 4: meio + 4]),
            ("Maior F1 sem correspondência exata", [r for r in reversed(ordenados) if not r["exata"]][:8]),
        ]:
            f.write(f"\n{'=' * 70}\n{titulo}\n{'=' * 70}\n")
            for r in itens:
                f.write(f"\nInstrução: {r['instruction']}\n")
                if r["input"]:
                    f.write(f"Entrada  : {r['input']}\n")
                f.write(f"Esperada : {r['output']}\n")
                f.write(f"Ajustado : {r['resposta_depois']}  [F1 {r['f1']:.2f}]\n")
                f.write(f"Antes    : {r['resposta_antes'][:160]}\n")
    print(f"  amostra -> results-by-sprints/sprint06/{destino.name}")
    _grafico(registros, linhas)
    return linhas


def _grafico(registros: list[dict], linhas: list[dict]) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    antes = [f1_de_palavras(r["resposta_antes"], r["output"]) for r in registros]
    depois = [r["f1"] for r in registros]
    caixas = [i / 10 for i in range(11)]
    a1.hist([antes, depois], bins=caixas, label=["pré-treinado", "ajustado"])
    a1.set_xlabel("F1 de palavras por exemplo de teste")
    a1.set_ylabel("exemplos")
    a1.set_title("Semelhança entre a resposta gerada e a esperada")
    a1.grid(alpha=0.3, axis="y")
    a1.legend()

    todos = {l["modelo"]: l for l in linhas if l["exemplos_de_teste"] == "todos"}
    rotulos = ["palavras\nesperadas", "geradas\n(pré-treinado)", "geradas\n(ajustado)"]
    valores = [todos["ajustado"]["palavras_esperadas"], todos["pré-treinado"]["palavras_geradas"],
               todos["ajustado"]["palavras_geradas"]]
    barras = a2.bar(rotulos, valores, color=["#888888", "#1f77b4", "#ff7f0e"])
    for barra, v in zip(barras, valores):
        a2.annotate(f"{v:.1f}", (barra.get_x() + barra.get_width() / 2, v), ha="center", va="bottom")
    a2.set_ylabel("palavras por resposta (média)")
    a2.set_title("O modelo aprendeu a parar?")
    a2.grid(alpha=0.3, axis="y")
    salvar_figura(fig, "exp06-avaliacao-das-respostas")


if __name__ == "__main__":
    executar()
