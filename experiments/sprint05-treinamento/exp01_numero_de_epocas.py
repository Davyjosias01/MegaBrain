"""Experimento 01 — número de épocas.

**O que varia:** ``num_epocas`` em {1, 3, 5, 10, 20, 40}.
**O que fica fixo:** GPT reduzido (emb_dim 256, 4 blocos, 4 cabeças), lote 2,
contexto 256, AdamW ``lr = 4e-4``, ``weight_decay = 0,1``, semente 123.

**Pergunta:** quanto treinar? Mais épocas sempre reduzem a perda de
*treinamento*, mas a perda de *validação* é a que mede generalização. O ponto
em que ela para de cair — ou passa a subir — marca o início do sobreajuste.

Cada configuração é treinada do zero. Além da tabela final, a figura mostra a
curva completa do treino mais longo, com a época de menor perda de validação
marcada, e as amostras geradas ao fim de cada treino.
"""

from __future__ import annotations

from dataclasses import replace

from comum import (
    CONTEXTO,
    RESULTADOS,
    TREINO_BASE,
    cabecalho,
    imprimir_tabela,
    plt,
    salvar_figura,
    salvar_tabela,
    treinar_rapido,
)
from treinamento import gerar_amostra, perplexidade

EPOCAS = [1, 3, 5, 10, 20, 40]


def executar() -> list[dict]:
    cabecalho("Experimento 01 — número de épocas")
    linhas, amostras, mais_longo = [], [], None
    for n in EPOCAS:
        r = treinar_rapido(cfg_treino=replace(TREINO_BASE, num_epocas=n))
        amostra = gerar_amostra(r["modelo"], CONTEXTO, TREINO_BASE.dispositivo, max_novos_tokens=30)
        amostras.append((n, amostra))
        linhas.append({
            "epocas": n,
            "tokens_vistos": r["tokens_vistos"],
            "perda_treino": r["perda_treino"],
            "perda_validacao": r["perda_validacao"],
            "lacuna_val_treino": r["perda_validacao"] - r["perda_treino"],
            "perplexidade_validacao": perplexidade(r["perda_validacao"]),
            "melhor_validacao": r["melhor_validacao"],
            "tempo_s": r["tempo_s"],
        })
        mais_longo = r
        del r["modelo"]

    imprimir_tabela(linhas)
    salvar_tabela(linhas, "exp01-numero-de-epocas")
    with open(RESULTADOS / "exp01-amostras.txt", "w", encoding="utf-8") as f:
        f.write(f"Contexto: {CONTEXTO!r} — greedy, 30 tokens, ao fim de cada treino\n\n")
        for n, a in amostras:
            f.write(f"{n:>2} épocas: {a!r}\n")
            print(f"  {n:>2} épocas: {a!r}")
    _graficos(linhas, mais_longo["historico"])
    return linhas


def _graficos(linhas: list[dict], historico: dict) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ep = [l["epocas"] for l in linhas]
    a1.plot(ep, [l["perda_treino"] for l in linhas], "o-", label="treino")
    a1.plot(ep, [l["perda_validacao"] for l in linhas], "s--", label="validação")
    a1.set_xscale("log")
    a1.set_xticks(ep, [str(e) for e in ep])
    a1.set_xlabel("número de épocas (treinos independentes)")
    a1.set_ylabel("perda final")
    a1.set_title("Perda ao fim de cada treino")
    a1.legend()
    a1.grid(alpha=0.3)

    # posição de cada avaliação em épocas fracionárias: passos por época = lotes de treino
    passos_por_epoca = historico["passo"][-1] // max(historico["epoca"]) + 1
    epoca = [(p + 1) / passos_por_epoca for p in historico["passo"]]
    a2.plot(epoca, historico["perda_treino"], label="treino")
    a2.plot(epoca, historico["perda_validacao"], "--", label="validação")
    i = min(range(len(epoca)), key=lambda k: historico["perda_validacao"][k])
    a2.axvline(epoca[i], color="gray", lw=1, ls=":")
    a2.annotate(f"menor validação\n(época {epoca[i]:.0f}: {historico['perda_validacao'][i]:.2f})",
                (epoca[i], historico["perda_validacao"][i]), textcoords="offset points",
                xytext=(10, 20), fontsize=8)
    a2.set_xlabel("época")
    a2.set_ylabel("perda")
    a2.set_title(f"Curva do treino de {max(historico['epoca'])} épocas")
    a2.legend()
    a2.grid(alpha=0.3)
    salvar_figura(fig, "exp01-numero-de-epocas")


if __name__ == "__main__":
    executar()
