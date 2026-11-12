"""Experimento 04 — tamanho do modelo e custo computacional.

**O que varia:** a escala da arquitetura — ``(emb_dim, n_layers, n_heads)`` em
(64, 2, 2), (128, 2, 4), (256, 4, 4), (384, 6, 6), (512, 8, 8) e (768, 12, 12),
este último o GPT-2 124M do treino base.
**O que fica fixo:** vocabulário (50.257), contexto 256, 10 épocas, lote 2,
AdamW ``lr = 4e-4``, semente 123.

**Pergunta:** com tão poucos dados (≈4.600 tokens de treino), aumentar o
modelo melhora a validação ou só acelera a memorização — e quanto custa?

Mede parâmetros, memória dos pesos, tempo de treino, perdas finais e a lacuna
validação–treino. Uma parte grande dos parâmetros dos modelos pequenos está
nas tabelas de embedding (50.257 × ``emb_dim``) e na cabeça de saída, que
crescem só com ``emb_dim``.
"""

from __future__ import annotations

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
from gpt import tamanho_em_mb
from treinamento import perplexidade

ESCALAS = [(64, 2, 2), (128, 2, 4), (256, 4, 4), (384, 6, 6), (512, 8, 8), (768, 12, 12)]


def executar() -> list[dict]:
    cabecalho("Experimento 04 — tamanho do modelo")
    linhas = []
    for emb, camadas, cabecas in ESCALAS:
        cfg = dict(CONFIG_REDUZIDO, emb_dim=emb, n_layers=camadas, n_heads=cabecas)
        r = treinar_rapido(cfg, TREINO_BASE)
        linhas.append({
            "emb_dim": emb,
            "n_layers": camadas,
            "n_heads": cabecas,
            "parametros": r["parametros"],
            "pesos_mb": tamanho_em_mb(r["parametros"]),
            "perda_treino": r["perda_treino"],
            "perda_validacao": r["perda_validacao"],
            "melhor_validacao": r["melhor_validacao"],
            "lacuna_val_treino": r["perda_validacao"] - r["perda_treino"],
            "perplexidade_validacao": perplexidade(r["perda_validacao"]),
            "tempo_s": r["tempo_s"],
        })
        del r["modelo"]

    imprimir_tabela(linhas)
    salvar_tabela(linhas, "exp04-tamanho-do-modelo")
    _graficos(linhas)
    return linhas


def _graficos(linhas: list[dict]) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    p = [l["parametros"] / 1e6 for l in linhas]
    a1.plot(p, [l["perda_treino"] for l in linhas], "o-", label="treino")
    a1.plot(p, [l["perda_validacao"] for l in linhas], "s--", label="validação")
    for x, l in zip(p, linhas):
        a1.annotate(f"d={l['emb_dim']}", (x, l["perda_treino"]), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=7)
    a1.set_xscale("log")
    a1.set_xlabel("parâmetros (milhões, escala log)")
    a1.set_ylabel("perda final (10 épocas)")
    a1.set_title("Perda × tamanho do modelo")
    a1.legend()
    a1.grid(alpha=0.3, which="both")

    a2.plot(p, [l["tempo_s"] for l in linhas], "o-", color="tab:red")
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xlabel("parâmetros (milhões, escala log)")
    a2.set_ylabel("tempo de treino (s, escala log)")
    a2.set_title("Custo computacional")
    a2.grid(alpha=0.3, which="both")
    salvar_figura(fig, "exp04-tamanho-do-modelo")


if __name__ == "__main__":
    executar()
