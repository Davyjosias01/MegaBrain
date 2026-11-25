"""Experimento 03 — quais camadas ficam treináveis.

**O que varia:** o congelamento (``adaptar_para_classificacao``):

- **só a cabeça** — todo o GPT-2 congelado; treina apenas a ``Linear(768, 2)``;
- **último bloco** — a configuração base do capítulo: último bloco
  Transformer + norma final + cabeça;
- **metade do modelo** — os 6 últimos blocos + norma final + cabeça.

**O que fica fixo:** pesos pré-treinados de partida, dados, semente, épocas,
lote, otimizador e taxa de aprendizado.

**Pergunta:** quanto do modelo é preciso ajustar para a tarefa, e a que custo?

O que esperar: treinar só a cabeça é um classificador linear sobre uma
representação que foi otimizada para prever o próximo token, não para separar
spam — aprende, mas menos. Liberar o último bloco deixa o modelo reorganizar
essa representação para a tarefa. Liberar mais blocos aumenta a capacidade de
ajuste, mas também o custo: a retropropagação precisa voltar por mais
camadas, e cada parâmetro treinável ocupa memória três vezes a mais (o
gradiente e as duas médias móveis do AdamW). Com pouco mais de 900 exemplos,
o ganho sobre o último bloco tende a ser pequeno.

O número de parâmetros treináveis mede a capacidade de ajuste; o tempo mede o
custo. O passo direto custa o mesmo nas três configurações — o que muda é até
onde a retropropagação precisa voltar.

O ajuste do modelo inteiro (*full fine-tuning*, ``blocos_treinaveis=None``)
não entra na comparação: com 124 milhões de parâmetros treináveis, modelo,
gradientes e estado do otimizador somam cerca de 2 GB, mais do que a máquina
usada (7,5 GB de RAM, sem GPU) tinha livre.

A configuração "último bloco" reaproveita o resultado de
``treinar_classificador.py`` quando ele existe, em vez de treinar de novo.
"""

from __future__ import annotations

import json

from comum import (
    RESULTADOS,
    TREINO_CLASSIFICADOR,
    cabecalho,
    imprimir_tabela,
    plt,
    salvar_figura,
    salvar_tabela,
    treinar_configuracao,
)

# (rótulo, blocos_treinaveis, treinar_norm_final)
CONFIGURACOES = [
    ("só a cabeça", 0, False),
    ("último bloco", 1, True),
    ("metade do modelo", 6, True),
]


def resultado_base() -> dict | None:
    """Resultado gravado por ``treinar_classificador.py``, se for da configuração base."""
    caminho = RESULTADOS / "classificador-config.json"
    if not caminho.exists():
        return None
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    mesma = all(dados["treino"].get(k) == v for k, v in TREINO_CLASSIFICADOR.items() if k != "max_length")
    if not mesma:
        return None
    return {
        "treinaveis": dados["parametros_treinaveis"],
        "parametros": dados["parametros_totais"],
        "tempo_s": dados["tempo_de_treino_s"],
        "acuracia_validacao": dados["acuracia_validacao"],
        "acuracia_teste": dados["acuracia_teste"],
    }


def executar() -> list[dict]:
    cabecalho("Experimento 03 — camadas treináveis")
    linhas = []
    for rotulo, blocos, norma in CONFIGURACOES:
        r = resultado_base() if (blocos, norma) == (1, True) else None
        origem = "treinar_classificador.py" if r else "este experimento"
        if r is None:
            r = treinar_configuracao(blocos_treinaveis=blocos, treinar_norm_final=norma)
            r.pop("modelo"), r.pop("otimizador"), r.pop("loaders")
        linhas.append({
            "configuracao": rotulo,
            "treinaveis": r["treinaveis"],
            "fracao_treinavel": r["treinaveis"] / r["parametros"],
            "acuracia_validacao": r["acuracia_validacao"],
            "acuracia_teste": r["acuracia_teste"],
            "tempo_min": r["tempo_s"] / 60,
            "origem": origem,
        })
        print(f"  {rotulo}: validação {r['acuracia_validacao']:.2%}, teste {r['acuracia_teste']:.2%}, "
              f"{r['tempo_s'] / 60:.1f} min")
    imprimir_tabela(linhas)
    salvar_tabela(linhas, "exp03-camadas-treinaveis")
    _grafico(linhas)
    return linhas


def _grafico(linhas: list[dict]) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    rotulos = [f"{l['configuracao']}\n{l['treinaveis']:,} parâmetros".replace(",", ".") for l in linhas]
    x = range(len(linhas))
    a1.bar([k - 0.2 for k in x], [l["acuracia_validacao"] for l in linhas], 0.4, label="validação")
    a1.bar([k + 0.2 for k in x], [l["acuracia_teste"] for l in linhas], 0.4, label="teste")
    a1.set_xticks(list(x), rotulos, fontsize=8)
    a1.set_ylim(0.5, 1.0)
    a1.set_ylabel("acurácia")
    a1.set_title("Acurácia por configuração de congelamento")
    a1.grid(alpha=0.3, axis="y")
    a1.legend()
    a2.bar(list(x), [l["tempo_min"] for l in linhas], color="#888888")
    a2.set_xticks(list(x), [l["configuracao"] for l in linhas], fontsize=8)
    a2.set_ylabel(f"tempo de treino (min, {TREINO_CLASSIFICADOR['num_epocas']} épocas)")
    a2.set_title("Custo computacional")
    a2.grid(alpha=0.3, axis="y")
    salvar_figura(fig, "exp03-camadas-treinaveis")


if __name__ == "__main__":
    executar()
