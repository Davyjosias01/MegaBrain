"""Experimento 02 — taxa de aprendizado.

**O que varia:** ``taxa_aprendizado`` em {1e-5, 1e-4, 4e-4, 1e-3, 3e-3, 1e-2}.
**O que fica fixo:** GPT reduzido, 10 épocas, lote 2, AdamW com
``weight_decay = 0,1``, semente 123.

**Pergunta:** qual o efeito do tamanho do passo do otimizador sobre a
convergência?

- Taxa pequena demais: cada passo muda pouco os pesos e, no mesmo número de
  épocas, a perda mal sai do valor inicial (~ln V).
- Taxa grande demais: os passos "pulam" o mínimo; a perda oscila ou estaciona
  num valor alto, e o modelo pode colapsar em prever sempre os mesmos tokens.
- No meio está a faixa útil, que o livro adota com ``4e-4``.

Além das perdas finais, mede a **instabilidade**: o desvio-padrão das
variações da perda de treino entre avaliações consecutivas.
"""

from __future__ import annotations

import statistics
from dataclasses import replace

from comum import (
    TREINO_BASE,
    cabecalho,
    imprimir_tabela,
    plt,
    salvar_figura,
    salvar_tabela,
    treinar_rapido,
)
from treinamento import perplexidade

TAXAS = [1e-5, 1e-4, 4e-4, 1e-3, 3e-3, 1e-2]


def executar() -> list[dict]:
    cabecalho("Experimento 02 — taxa de aprendizado")
    linhas, curvas = [], {}
    for lr in TAXAS:
        r = treinar_rapido(cfg_treino=replace(TREINO_BASE, taxa_aprendizado=lr))
        h = r["historico"]
        variacoes = [b - a for a, b in zip(h["perda_treino"], h["perda_treino"][1:])]
        linhas.append({
            "taxa_aprendizado": f"{lr:g}",
            "perda_treino": r["perda_treino"],
            "perda_validacao": r["perda_validacao"],
            "melhor_validacao": r["melhor_validacao"],
            "perplexidade_validacao": perplexidade(r["perda_validacao"]),
            "instabilidade": statistics.pstdev(variacoes),
            "aumentos_da_perda": sum(v > 0 for v in variacoes),
            "tempo_s": r["tempo_s"],
        })
        curvas[lr] = h
        del r["modelo"]

    imprimir_tabela(linhas)
    salvar_tabela(linhas, "exp02-taxa-de-aprendizado")
    _graficos(linhas, curvas)
    return linhas


def _graficos(linhas: list[dict], curvas: dict) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for lr, h in curvas.items():
        a1.plot(h["tokens_vistos"], h["perda_treino"], label=f"lr = {lr:g}")
    a1.set_xlabel("tokens vistos")
    a1.set_ylabel("perda de treino")
    a1.set_title("Convergência por taxa de aprendizado")
    a1.legend(fontsize=8)
    a1.grid(alpha=0.3)

    taxas = list(curvas)
    a2.plot(taxas, [l["perda_treino"] for l in linhas], "o-", label="treino")
    a2.plot(taxas, [l["perda_validacao"] for l in linhas], "s--", label="validação")
    a2.set_xscale("log")
    a2.set_xlabel("taxa de aprendizado (escala log)")
    a2.set_ylabel("perda final (10 épocas)")
    a2.set_title("Perda final × taxa de aprendizado")
    a2.legend()
    a2.grid(alpha=0.3, which="both")
    salvar_figura(fig, "exp02-taxa-de-aprendizado")


if __name__ == "__main__":
    executar()
