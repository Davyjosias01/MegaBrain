"""Experimento 08 — estratégias de seleção do próximo token.

**O que varia:** a estratégia — greedy, temperatura em {0,5; 1; 1,5} e top-k em
{50, 5} — e a semente do sorteio.
**O que fica fixo:** GPT-2 small (124M) não treinado, semente de inicialização
123, prompt "Every effort moves you", 25 tokens novos, 5 sementes de sorteio.

**Pergunta:** como temperatura e top-k transformam a distribuição do próximo
token, e o que isso muda no texto gerado?

Duas partes:

(a) **O mecanismo isolado**, com o exemplo de 9 tokens da seção 5.3: os mesmos
    logits convertidos em probabilidades com temperaturas diferentes e com
    top-k = 3. Aqui os logits têm estrutura (três tokens se destacam), e o
    efeito de cada parâmetro aparece com clareza.

(b) **O GPT de verdade.** Para cada estratégia, 5 gerações com sementes
    diferentes, medindo a entropia da distribuição de onde o token é sorteado,
    a probabilidade e a posição no ranking do token escolhido (1 = o mais
    provável), a diversidade dos tokens gerados e se sementes diferentes
    produzem textos diferentes.

Como o modelo não foi treinado, a distribuição do próximo token é quase
uniforme sobre os 50.257 tokens. Isso é parte do resultado: a temperatura só
redistribui a confiança que o modelo já tem, e um modelo sem treinamento não
tem nenhuma.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from comum import (
    CATEGORICA,
    PROMPT,
    cabecalho,
    imprimir_tabela,
    salvar_figura,
    salvar_tabela,
    semear,
)

from geracao import ajustar_logits, ids_para_texto, selecionar_proximo_token, texto_para_ids
from gpt import CONFIG_GPT2_124M, GPTModel

# (a) exemplo da seção 5.3
VOCAB_EXEMPLO = ["closer", "every", "effort", "forward", "inches", "moves", "pizza", "toward", "you"]
LOGITS_EXEMPLO = torch.tensor([[4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]])

# (b) estratégias no GPT-2 small
ESTRATEGIAS = [
    ("greedy", 0.0, None),
    ("T = 0,5", 0.5, None),
    ("T = 1", 1.0, None),
    ("T = 1,5", 1.5, None),
    ("T = 1, top-k 50", 1.0, 50),
    ("T = 1, top-k 5", 1.0, 5),
    ("T = 1,5, top-k 5", 1.5, 5),
]
NOVOS_TOKENS = 25
SEMENTES = [0, 1, 2, 3, 4]


def _entropia(probabilidades: torch.Tensor) -> float:
    p = probabilidades.clamp_min(1e-30)
    return float(-(probabilidades * p.log()).sum(-1).mean())


def _exemplo_do_livro() -> list[dict]:
    linhas = []
    for rotulo, t, k in [("T = 0,1", 0.1, None), ("T = 1", 1.0, None), ("T = 5", 5.0, None),
                         ("T = 1, top-k 3", 1.0, 3)]:
        p = torch.softmax(ajustar_logits(LOGITS_EXEMPLO, t, k), dim=-1)[0]
        linhas.append({"configuracao": rotulo, **{tok: float(v) for tok, v in zip(VOCAB_EXEMPLO, p)}})
    return linhas


@torch.no_grad()
def _gerar_medindo(modelo, ids, temperatura, top_k) -> dict:
    entropias, escolhidos_p, ranks = [], [], []
    for _ in range(NOVOS_TOKENS):
        logits = modelo(ids[:, -1024:])[:, -1, :]
        if temperatura > 0:
            p = torch.softmax(ajustar_logits(logits, temperatura, top_k), dim=-1)
        else:
            p = torch.zeros_like(logits).scatter_(1, logits.argmax(-1, keepdim=True), 1.0)
        proximo = selecionar_proximo_token(logits, temperatura, top_k)
        entropias.append(_entropia(p))
        escolhidos_p.append(float(torch.softmax(logits, -1)[0, proximo.item()]))
        # Posição do token escolhido no ranking do modelo: 1 = o mais provável.
        ranks.append(int((logits[0] > logits[0, proximo.item()]).sum()) + 1)
        ids = torch.cat((ids, proximo), dim=1)
    return {"ids": ids, "entropia": sum(entropias) / len(entropias),
            "p_escolhido": sum(escolhidos_p) / len(escolhidos_p),
            "rank": sum(ranks) / len(ranks)}


def executar() -> list[dict]:
    cabecalho("Experimento 08 — estratégias de seleção do próximo token")

    exemplo = _exemplo_do_livro()
    print("  (a) exemplo da seção 5.3 — probabilidades por configuração")
    imprimir_tabela(exemplo)
    salvar_tabela(exemplo, "exp08-exemplo-temperatura-top-k")

    semear()
    modelo = GPTModel(CONFIG_GPT2_124M).eval()
    prompt = texto_para_ids(PROMPT)
    n_prompt = prompt.shape[1]
    with torch.no_grad():
        logits0 = modelo(prompt)[:, -1, :]
    p0 = torch.softmax(logits0, -1)
    print(f"\n  (b) GPT-2 small não treinado: entropia do próximo token = "
          f"{_entropia(p0):.3f} nats (máximo ln 50257 = {math.log(50257):.3f}); "
          f"probabilidade do token mais provável = {float(p0.max()):.2e} "
          f"(uniforme = {1 / 50257:.2e})")

    linhas: list[dict] = []
    for nome, t, k in ESTRATEGIAS:
        resultados = []
        for s in SEMENTES:
            torch.manual_seed(s)
            resultados.append(_gerar_medindo(modelo, prompt, t, k))
        gerados = [r["ids"][0, n_prompt:].tolist() for r in resultados]
        linhas.append(
            {
                "estrategia": nome,
                "entropia_nats": sum(r["entropia"] for r in resultados) / len(resultados),
                "p_token_escolhido": sum(r["p_escolhido"] for r in resultados) / len(resultados),
                "rank_medio_do_escolhido": sum(r["rank"] for r in resultados) / len(resultados),
                "tokens_distintos_pct": 100 * float(np.mean([len(set(g)) / len(g) for g in gerados])),
                "textos_distintos_em_5": len({tuple(g) for g in gerados}),
                "exemplo": ids_para_texto(resultados[0]["ids"]).replace("\n", " "),
            }
        )

    curtas = [{k: v for k, v in l.items() if k != "exemplo"} for l in linhas]
    imprimir_tabela(curtas)
    for l in linhas:
        print(f"    {l['estrategia']:<18} {l['exemplo'][:110]!r}")
    salvar_tabela(linhas, "exp08-estrategias-de-geracao")
    _graficos(exemplo, linhas)
    return linhas


def _graficos(exemplo: list[dict], linhas: list[dict]) -> None:
    fig, eixos = plt.subplots(1, 2, figsize=(12, 4.0), gridspec_kw={"width_ratios": [1.5, 1]})

    pos = np.arange(len(VOCAB_EXEMPLO))
    largura = 0.2
    for i, l in enumerate(exemplo):
        eixos[0].bar(pos + (i - 1.5) * largura, [l[t] for t in VOCAB_EXEMPLO],
                     width=largura, color=CATEGORICA[i], label=l["configuracao"])
    eixos[0].set_xticks(pos, VOCAB_EXEMPLO, rotation=30)
    eixos[0].grid(axis="x", visible=False)
    eixos[0].legend(fontsize=7)
    eixos[0].set_title("(a) Exemplo da seção 5.3: os mesmos logits, quatro distribuições", pad=8)
    eixos[0].set_ylabel("probabilidade")

    nomes = [l["estrategia"] for l in linhas]
    eixos[1].barh(np.arange(len(nomes)), [l["entropia_nats"] for l in linhas], color=CATEGORICA[0])
    eixos[1].axvline(math.log(50257), color="#52514e", linestyle="--", linewidth=1)
    eixos[1].set_yticks(np.arange(len(nomes)), nomes)
    eixos[1].invert_yaxis()
    eixos[1].grid(axis="y", visible=False)
    eixos[1].set_title("(b) GPT-2 não treinado: entropia da distribuição\nde onde o token é sorteado (--- ln 50257)", pad=8)
    eixos[1].set_xlabel("entropia (nats)")

    fig.suptitle("Experimento 08 — temperatura e top-k", y=1.03)
    fig.tight_layout()
    salvar_figura(fig, "exp08-estrategias-de-geracao")


if __name__ == "__main__":
    executar()
