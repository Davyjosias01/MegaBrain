"""Experimento 03 — configuração do treinamento: lote, weight decay e dropout.

**O que varia:** uma coisa de cada vez em relação à configuração base —

- tamanho do lote: 1, 2 (base) e 4;
- ``weight_decay`` do AdamW: 0, 0,1 (base) e 0,5;
- ``drop_rate`` do modelo: 0, 0,1 (base) e 0,3.

**O que fica fixo:** GPT reduzido, ``lr = 4e-4``, **20 épocas** (o ponto em que,
no experimento 01, o sobreajuste já aparece), semente 123.

**Pergunta:** que configurações reduzem a lacuna entre treino e validação, e a
que custo?

- O **lote** muda quantos passos de otimização cabem numa época (com 18
  amostras de treino: 18, 9 e 4 passos — o lote de 4 descarta 2 amostras por
  ``drop_last``) e o ruído de cada gradiente.
- O **weight decay** puxa os pesos para zero a cada passo, limitando o quanto
  o modelo pode se ajustar ao conjunto de treino.
- O **dropout** desliga neurônios aleatoriamente durante o treino, impedindo
  que o modelo dependa de combinações específicas — é a regularização
  embutida na arquitetura GPT.
"""

from __future__ import annotations

from dataclasses import replace

from comum import (
    CONFIG_REDUZIDO,
    TREINO_BASE,
    cabecalho,
    imprimir_tabela,
    plt,
    salvar_figura,
    salvar_tabela,
    treinar_rapido,
)
from treinamento import perplexidade

EPOCAS = 20
BASE = replace(TREINO_BASE, num_epocas=EPOCAS)

# (grupo, rótulo, alterações no treino, alterações no modelo)
CONFIGURACOES = [
    ("base", "base (lote 2, wd 0,1, dropout 0,1)", {}, {}),
    ("lote", "lote 1", {"batch_size": 1}, {}),
    ("lote", "lote 4", {"batch_size": 4}, {}),
    ("weight decay", "weight decay 0", {"weight_decay": 0.0}, {}),
    ("weight decay", "weight decay 0,5", {"weight_decay": 0.5}, {}),
    ("dropout", "dropout 0", {}, {"drop_rate": 0.0}),
    ("dropout", "dropout 0,3", {}, {"drop_rate": 0.3}),
]


def executar() -> list[dict]:
    cabecalho("Experimento 03 — configuração do treinamento")
    linhas = []
    for grupo, rotulo, treino, modelo in CONFIGURACOES:
        r = treinar_rapido(dict(CONFIG_REDUZIDO, **modelo), replace(BASE, **treino))
        linhas.append({
            "grupo": grupo,
            "configuracao": rotulo,
            "passos": r["passos"],
            "perda_treino": r["perda_treino"],
            "perda_validacao": r["perda_validacao"],
            "melhor_validacao": r["melhor_validacao"],
            "lacuna_val_treino": r["perda_validacao"] - r["perda_treino"],
            "perplexidade_validacao": perplexidade(r["perda_validacao"]),
            "tempo_s": r["tempo_s"],
        })
        del r["modelo"]

    imprimir_tabela(linhas)
    salvar_tabela(linhas, "exp03-configuracao-de-treino")
    _grafico(linhas)
    return linhas


def _grafico(linhas: list[dict]) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    rotulos = [l["configuracao"] for l in linhas]
    y = range(len(linhas))
    a1.barh([i + 0.2 for i in y], [l["perda_treino"] for l in linhas], 0.4, label="treino")
    a1.barh([i - 0.2 for i in y], [l["perda_validacao"] for l in linhas], 0.4, label="validação")
    a1.set_yticks(list(y), rotulos)
    a1.invert_yaxis()
    a1.set_xlabel(f"perda final ({EPOCAS} épocas)")
    a1.set_title("Perdas por configuração")
    a1.legend()
    a1.grid(alpha=0.3, axis="x")

    a2.barh(list(y), [l["lacuna_val_treino"] for l in linhas], color="gray")
    for i, l in enumerate(linhas):
        a2.text(l["lacuna_val_treino"], i, f" {l['tempo_s']:.1f} s", va="center", fontsize=8)
    a2.set_yticks(list(y), rotulos)
    a2.invert_yaxis()
    a2.set_xlabel("lacuna validação − treino (maior = mais sobreajuste)")
    a2.set_title("Sobreajuste e custo (tempo de treino)")
    a2.grid(alpha=0.3, axis="x")
    salvar_figura(fig, "exp03-configuracao-de-treino")


if __name__ == "__main__":
    executar()
