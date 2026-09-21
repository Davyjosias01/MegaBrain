"""Experimento 07 — comportamento da máscara causal.

**O que varia:** a máscara ligada e desligada, sobre o mesmo módulo e a mesma
entrada; e a taxa de dropout aplicada aos pesos já mascarados.

**Perguntas:**

1. As duas formas de mascarar descritas no livro — multiplicar por ``tril`` e
   renormalizar (figura 3.20) ou preencher com ``-inf`` antes do softmax
   (figura 3.21) — dão exatamente o mesmo resultado?
2. Depois de mascarada, cada linha ainda soma 1?
3. A máscara realmente impede o vazamento de informação do futuro? O teste é
   direto: alterar drasticamente o embedding de um token e verificar quais
   vetores de contexto mudam. Se a máscara funciona, nenhum contexto anterior a
   ele pode se mover um décimo de milésimo sequer.
4. Como a máscara redistribui a atenção? Sem máscara, um token divide sua
   atenção entre ``n`` posições; com máscara, o token da posição ``i`` divide
   entre ``i+1``. As primeiras posições ficam, por construção, com pouquíssimas
   opções — e o token 0 atende apenas a si mesmo, com peso 1.
5. O dropout (seção 3.5.2) quebra a soma 1 das linhas? Sim, e propositalmente:
   ele zera pesos e reescala os sobreviventes por ``1/(1-p)`` para preservar a
   *esperança* da saída, não a soma de cada linha.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from comum import (
    CATEGORICA,
    MARCADORES,
    TINTA_SECUNDARIA,
    cabecalho,
    desenhar_matriz_atencao,
    entropia_normalizada,
    estilizar_linhas,
    imprimir_tabela,
    salvar_figura,
    salvar_tabela,
    semear,
)

from attention import CausalAttention, mascara_causal
from entradas_atencao import lote_de_embeddings

D_MODELO = 64
NUM_TOKENS = 24


def executar() -> list[dict]:
    cabecalho("Experimento 07 — comportamento da máscara causal")

    dados = lote_de_embeddings(d_in=D_MODELO, max_length=NUM_TOKENS, batch_size=1)
    x = dados["embeddings"]
    rotulos = dados["tokens"]

    semear()
    ca = CausalAttention(D_MODELO, D_MODELO, NUM_TOKENS, 0.0).eval()

    contexto_com, pesos_com = ca(x, return_attn_weights=True)
    contexto_sem, pesos_sem = ca(x, return_attn_weights=True, usar_mascara=False)

    # --- 1. equivalência entre as duas estratégias de máscara ---------------
    scores = ca.W_query(x) @ ca.W_key(x).transpose(-2, -1) / D_MODELO**0.5
    via_320 = torch.softmax(scores, dim=-1) * torch.tril(
        torch.ones(NUM_TOKENS, NUM_TOKENS)
    )
    via_320 = via_320 / via_320.sum(dim=-1, keepdim=True)
    via_321 = torch.softmax(
        scores.masked_fill(mascara_causal(NUM_TOKENS), -torch.inf), dim=-1
    )
    equivalentes = torch.allclose(via_320, via_321, atol=1e-6)
    print(
        f"  1. figura 3.20 (zerar e renormalizar) == figura 3.21 (-inf antes do "
        f"softmax): {'sim' if equivalentes else 'NAO'}  "
        f"(maior diferença = {float((via_320 - via_321).abs().max()):.2e})"
    )

    # --- 2. linhas continuam somando 1 -------------------------------------
    somas = pesos_com[0].sum(dim=-1)
    print(
        f"  2. soma das linhas após a máscara: mínimo {float(somas.min()):.6f}, "
        f"máximo {float(somas.max()):.6f}"
    )
    print(
        f"     pesos acima da diagonal: máximo "
        f"{float(pesos_com[0].triu(diagonal=1).max()):.1e}"
    )

    # --- 3. ausência de vazamento de informação ----------------------------
    posicao_alterada = NUM_TOKENS // 2
    x_alterado = x.clone()
    x_alterado[0, posicao_alterada] += 50.0

    with torch.no_grad():
        contexto_alterado = ca(x_alterado)
        contexto_alterado_sem = ca(x_alterado, usar_mascara=False)

    delta_com = (contexto_alterado - contexto_com).abs().amax(dim=-1)[0]
    delta_sem = (contexto_alterado_sem - contexto_sem).abs().amax(dim=-1)[0]
    print(
        f"  3. token {posicao_alterada} perturbado — maior mudança nos contextos "
        f"anteriores:"
    )
    print(
        f"     com máscara: {float(delta_com[:posicao_alterada].max()):.2e}  |  "
        f"sem máscara: {float(delta_sem[:posicao_alterada].max()):.2e}"
    )
    print(
        f"     nas posições a partir de {posicao_alterada}, com máscara: "
        f"{float(delta_com[posicao_alterada:].max()):.2e}"
    )

    # --- 4. redistribuição da atenção por posição --------------------------
    linhas: list[dict] = []
    for i in range(NUM_TOKENS):
        linhas.append(
            {
                "posicao": i,
                "token": rotulos[i],
                "posicoes_visiveis_com_mascara": i + 1,
                "peso_maximo_com_mascara": float(pesos_com[0, i].max()),
                "peso_maximo_sem_mascara": float(pesos_sem[0, i].max()),
                "atencao_ao_proprio_token": float(pesos_com[0, i, i]),
            }
        )
    imprimir_tabela(linhas[:8])
    print("  ... (tabela completa no CSV)")
    salvar_tabela(linhas, "exp07-mascara-causal-por-posicao")

    print(
        f"     entropia normalizada — com máscara "
        f"{entropia_normalizada(pesos_com):.4f}, sem máscara "
        f"{entropia_normalizada(pesos_sem, causal=False):.4f}"
    )

    # --- 5. efeito do dropout ----------------------------------------------
    resumo_dropout: list[dict] = []
    for taxa in (0.0, 0.1, 0.2, 0.5):
        semear()
        ca_drop = CausalAttention(D_MODELO, D_MODELO, NUM_TOKENS, taxa)
        ca_drop.load_state_dict(ca.state_dict())
        ca_drop.train()  # dropout só age em modo de treino
        semear(7)
        _, pesos_drop = ca_drop(x, return_attn_weights=True)
        p = pesos_drop[0].detach()
        validos = torch.tril(torch.ones(NUM_TOKENS, NUM_TOKENS)).bool()
        resumo_dropout.append(
            {
                "dropout": taxa,
                "fracao_pesos_zerados": float(
                    (p[validos] == 0).float().mean()
                ),
                "soma_media_das_linhas": float(p.sum(dim=-1).mean()),
                "peso_maximo": float(p.max()),
            }
        )
    print("\n  5. efeito do dropout sobre os pesos de atenção (modo treino):")
    imprimir_tabela(resumo_dropout)
    salvar_tabela(resumo_dropout, "exp07-mascara-causal-dropout")

    _graficos(pesos_com[0], pesos_sem[0], rotulos, linhas, delta_com, delta_sem)
    return linhas


def _graficos(pesos_com, pesos_sem, rotulos, linhas, delta_com, delta_sem) -> None:
    fig, eixos = plt.subplots(1, 3, figsize=(14, 4.6))

    desenhar_matriz_atencao(
        eixos[0], pesos_sem, "Sem máscara\n(cada token vê a sequência inteira)", rotulos
    )
    desenhar_matriz_atencao(
        eixos[1],
        pesos_com,
        "Com máscara causal\n(cinza = futuro escondido)",
        rotulos,
        marcar_mascarados=True,
    )

    eixos[2].plot(
        [l["posicao"] for l in linhas],
        [l["peso_maximo_com_mascara"] for l in linhas],
        color=CATEGORICA[0],
        marker=MARCADORES[0],
        label="com máscara",
    )
    eixos[2].plot(
        [l["posicao"] for l in linhas],
        [l["peso_maximo_sem_mascara"] for l in linhas],
        color=CATEGORICA[1],
        marker=MARCADORES[1],
        label="sem máscara",
    )
    eixos[2].set_ylim(0, 1.05)
    eixos[2].legend(loc="upper right", fontsize=8)
    estilizar_linhas(
        eixos[2],
        "Peso máximo de cada linha",
        "posição do token na sequência",
        "maior peso da linha",
    )
    eixos[2].text(
        0.5,
        0.55,
        "as primeiras posições têm poucas\nopções: a atenção é forçadamente\nconcentrada",
        transform=eixos[2].transAxes,
        fontsize=7.5,
        color=TINTA_SECUNDARIA,
        ha="left",
        va="top",
    )

    fig.suptitle(
        "Experimento 07 — a máscara causal restringe cada token ao próprio passado",
        y=1.03,
    )
    fig.tight_layout()
    salvar_figura(fig, "exp07-mascara-causal")

    # Segunda figura: o teste de vazamento de informação.
    fig2, ax = plt.subplots(figsize=(7.4, 3.6))
    posicoes = range(len(delta_com))
    ax.plot(
        posicoes,
        delta_com.detach().numpy(),
        color=CATEGORICA[0],
        marker=MARCADORES[0],
        label="com máscara causal",
    )
    ax.plot(
        posicoes,
        delta_sem.detach().numpy(),
        color=CATEGORICA[1],
        marker=MARCADORES[1],
        label="sem máscara",
    )
    ax.axvline(
        len(delta_com) // 2, color=TINTA_SECUNDARIA, linewidth=1, linestyle="--"
    )
    ax.set_yscale("symlog", linthresh=1e-12)
    ax.legend(loc="upper left", fontsize=8)
    estilizar_linhas(
        ax,
        "Teste de vazamento: quanto cada vetor de contexto muda quando\n"
        f"o token {len(delta_com) // 2} (linha tracejada) é perturbado",
        "posição do token",
        "maior variação absoluta",
    )
    fig2.tight_layout()
    salvar_figura(fig2, "exp07-vazamento-de-informacao")


if __name__ == "__main__":
    executar()
