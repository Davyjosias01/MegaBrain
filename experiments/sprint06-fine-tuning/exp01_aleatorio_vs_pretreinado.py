"""Experimento 01 — pesos aleatórios × pesos pré-treinados.

**O que varia:** apenas os **pesos**. A arquitetura é a mesma nos dois casos —
o ``GPTModel`` da Sprint 4 na configuração do GPT-2 124M (contexto 1024,
``qkv_bias=True``): uma vez recém-inicializado com a semente 123, outra com os
pesos publicados pela OpenAI carregados por ``pesos_pretreinados.py``.
**O que fica fixo:** arquitetura, tokenizador, texto de avaliação, contexto
inicial, estratégias de geração e sementes.

**Pergunta:** o que os pesos pré-treinados acrescentam a uma arquitetura que,
sozinha, não sabe nada?

Três medidas:

1. **Perda e perplexidade** em ``the-verdict.txt``, texto que nenhum dos dois
   modelos recebeu de nós. Com pesos aleatórios a previsão é quase uniforme
   sobre os 50.257 tokens (perda ≈ ln 50257 ≈ 10,8). O modelo pré-treinado
   não passou por nenhum treino nosso neste conto e ainda assim o prevê muito melhor.
2. **Próximo token** para o mesmo contexto: os cinco tokens mais prováveis e
   suas probabilidades.
3. **Geração** com o mesmo contexto e as mesmas estratégias de decodificação da
   Sprint 5 (greedy, temperatura, top-k), medindo diversidade e repetição.

Se o checkpoint do treino base da Sprint 5 existir localmente
(``results-by-sprints/sprint05/base-modelo.pt``), ele entra como terceira
linha: o modelo treinado do zero no próprio conto. Ele tem a perda mais baixa
de todas no conto porque o decorou — 90% do texto de avaliação é o seu texto
de treinamento —, o que serve de contraste entre memorizar um texto e ter
aprendido a língua.
"""

from __future__ import annotations

import statistics

import torch

from comum import (
    CHECKPOINT_SPRINT5,
    CONTEXTO,
    DISPOSITIVO,
    RESULTADOS,
    SEMENTE,
    TOKENIZADOR,
    cabecalho,
    distinct_n,
    imprimir_tabela,
    plt,
    repeticao_n,
    salvar_figura,
    salvar_tabela,
    semear,
)
from dados import carregar_texto, criar_dataloader
from geracao import gerar, texto_para_ids
from gpt import GPTModel, contar_parametros
from pesos_pretreinados import carregar_gpt2_pretreinado, config_gpt2_pretreinado
from treinamento import perda_do_loader, perplexidade

NOVOS_TOKENS = 40
SEMENTES = [123, 124, 125]
JANELA = 256  # tamanho das sequências de avaliação (o mesmo da Sprint 5)

# (rótulo, temperatura, top_k)
ESTRATEGIAS = [
    ("greedy", 0.0, None),
    ("T=1,0 top-k=50", 1.0, 50),
    ("T=1,5 top-k=50", 1.5, 50),
    ("amostragem pura", 1.0, None),
]


def _modelos() -> list[tuple[str, GPTModel]]:
    semear(SEMENTE)
    aleatorio = GPTModel(config_gpt2_pretreinado("small")).to(DISPOSITIVO).eval()
    pretreinado = carregar_gpt2_pretreinado("small", dispositivo=DISPOSITIVO)
    modelos = [("aleatório", aleatorio), ("pré-treinado (GPT-2 124M)", pretreinado)]
    if CHECKPOINT_SPRINT5.exists():
        from checkpoint import carregar_checkpoint

        sprint5, _ = carregar_checkpoint(CHECKPOINT_SPRINT5, DISPOSITIVO)
        modelos.append(("treinado do zero no conto (Sprint 5)", sprint5.eval()))
    else:
        print("  (checkpoint da Sprint 5 ausente: comparação só com dois modelos)")
    return modelos


def _proximos_tokens(modelo: GPTModel, k: int = 5) -> list[tuple[str, float]]:
    ids = texto_para_ids(CONTEXTO, TOKENIZADOR).to(DISPOSITIVO)
    with torch.no_grad():
        probabilidades = torch.softmax(modelo(ids)[0, -1], dim=-1)
    valores, indices = torch.topk(probabilidades, k)
    return [(TOKENIZADOR.decode([i]), v) for i, v in zip(indices.tolist(), valores.tolist())]


def _geracao(modelo: GPTModel) -> tuple[list[dict], list[tuple[str, str]]]:
    ids_contexto = texto_para_ids(CONTEXTO, TOKENIZADOR).to(DISPOSITIVO)
    n = ids_contexto.shape[1]
    tamanho_contexto = modelo.cfg["context_length"]
    linhas, amostras = [], []
    for rotulo, temperatura, top_k in ESTRATEGIAS:
        gerados = []
        for semente in SEMENTES:
            torch.manual_seed(semente)
            saida = gerar(modelo, ids_contexto, NOVOS_TOKENS, tamanho_contexto,
                          temperatura=temperatura, top_k=top_k)
            gerados.append(saida[0, n:].tolist())
        linhas.append({
            "estrategia": rotulo,
            "distinct_2": statistics.mean(distinct_n(g, 2) for g in gerados),
            "repeticao_3": statistics.mean(repeticao_n(g, 3) for g in gerados),
        })
        amostras.append((rotulo, TOKENIZADOR.decode(gerados[0])))
    return linhas, amostras


def executar() -> list[dict]:
    cabecalho("Experimento 01 — pesos aleatórios × pesos pré-treinados")
    texto = carregar_texto()
    loader = criar_dataloader(texto, batch_size=2, max_length=JANELA, stride=JANELA,
                              shuffle=False, drop_last=False, tokenizador=TOKENIZADOR)

    resumo, geracao, topk, amostras = [], [], {}, {}
    for rotulo, modelo in _modelos():
        perda = perda_do_loader(loader, modelo, DISPOSITIVO)
        proximos = _proximos_tokens(modelo)
        linhas_g, amostras_g = _geracao(modelo)
        resumo.append({
            "modelo": rotulo,
            "parametros": contar_parametros(modelo),
            "contexto": modelo.cfg["context_length"],
            "perda": perda,
            "perplexidade": perplexidade(perda),
            "proximo_token": repr(proximos[0][0]),
            "prob_proximo": proximos[0][1],
        })
        geracao += [{"modelo": rotulo, **linha} for linha in linhas_g]
        topk[rotulo], amostras[rotulo] = proximos, amostras_g
        del modelo

    print(f"  texto de avaliação: the-verdict.txt, {len(loader)} lotes de 2 × {JANELA} tokens")
    imprimir_tabela(resumo)
    salvar_tabela(resumo, "exp01-aleatorio-vs-pretreinado")
    print()
    imprimir_tabela(geracao)
    salvar_tabela(geracao, "exp01-geracao")
    _gravar_amostras(topk, amostras)
    _grafico(resumo, topk)
    return resumo


def _gravar_amostras(topk: dict, amostras: dict) -> None:
    destino = RESULTADOS / "exp01-amostras.txt"
    with open(destino, "w", encoding="utf-8") as f:
        f.write("Pesos aleatórios × pesos pré-treinados — mesma arquitetura\n")
        f.write(f"Contexto: {CONTEXTO!r}, {NOVOS_TOKENS} tokens, semente {SEMENTES[0]}\n")
        for rotulo in amostras:
            f.write(f"\n{'=' * 70}\n{rotulo}\n{'=' * 70}\n")
            f.write("Cinco próximos tokens mais prováveis: ")
            f.write(", ".join(f"{t!r} ({p:.2%})" for t, p in topk[rotulo]) + "\n\n")
            for estrategia, texto in amostras[rotulo]:
                f.write(f"[{estrategia}]\n{CONTEXTO}{texto}\n\n")
    print(f"  amostras -> results-by-sprints/sprint06/{destino.name}")
    for rotulo in amostras:
        print(f"  {rotulo} [greedy]: {CONTEXTO + amostras[rotulo][0][1]!r}"[:160])


def _grafico(resumo: list[dict], topk: dict) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    rotulos = [l["modelo"] for l in resumo]
    barras = a1.bar(range(len(rotulos)), [l["perplexidade"] for l in resumo],
                    color=["#888888", "#1f77b4", "#ff7f0e"][: len(rotulos)])
    a1.set_yscale("log")
    a1.set_xticks(range(len(rotulos)), [r.replace(" (", "\n(") for r in rotulos], fontsize=8)
    a1.set_ylabel("perplexidade em the-verdict.txt (escala log)")
    a1.set_title("Mesma arquitetura, pesos diferentes")
    a1.grid(alpha=0.3, axis="y")
    for barra, linha in zip(barras, resumo):
        a1.annotate(f"{linha['perplexidade']:,.1f}\n(perda {linha['perda']:.2f})",
                    (barra.get_x() + barra.get_width() / 2, barra.get_height()),
                    ha="center", va="bottom", fontsize=8)
    a1.set_ylim(top=a1.get_ylim()[1] * 4)

    rotulo = rotulos[1]
    tokens = [repr(t) for t, _ in topk[rotulo]]
    a2.barh(range(len(tokens)), [p for _, p in topk[rotulo]], color="#1f77b4", label=rotulo)
    a2.barh(range(len(tokens)), [topk[rotulos[0]][0][1]] * len(tokens), color="#888888",
            label=f"aleatório (maior probabilidade: {topk[rotulos[0]][0][1]:.4%})")
    a2.set_yticks(range(len(tokens)), tokens)
    a2.invert_yaxis()
    a2.set_xlabel("probabilidade do próximo token")
    a2.set_title(f"Próximo token após {CONTEXTO!r}")
    a2.legend(fontsize=8)
    a2.grid(alpha=0.3, axis="x")
    salvar_figura(fig, "exp01-aleatorio-vs-pretreinado")


if __name__ == "__main__":
    executar()
