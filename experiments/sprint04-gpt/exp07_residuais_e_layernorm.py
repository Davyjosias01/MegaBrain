"""Experimento 07 — conexões residuais e LayerNorm na estabilidade do modelo.

**O que varia:** a arquitetura do bloco — completa, sem conexões residuais, sem
LayerNorm, sem as duas — e a profundidade, de 2 a 24 blocos.
**O que fica fixo:** ``emb_dim = 128``, 4 cabeças, contexto 128, ``drop_rate = 0``,
lote real ``(4, 64)`` com seus alvos, semente 123.

**Pergunta:** o que acontece com o sinal e com o gradiente quando esses dois
mecanismos são retirados?

Nada é treinado. É feita **uma** passagem direta e **uma** retropropagação da
perda de entropia cruzada sobre os alvos reais — exatamente o primeiro passo do
treinamento da Sprint 5 — e mede-se:

- **desvio das ativações** na saída de cada bloco: se o sinal se mantém, explode
  ou se apaga ao atravessar a pilha;
- **norma do gradiente** que chega aos pesos de cada bloco: se os primeiros
  blocos recebem sinal suficiente para aprender;
- **perda inicial**: com previsões uniformes ela valeria ``ln(50257) ≈ 10,82``.

As duas peças são desligadas pelas chaves ``usar_residual`` e
``usar_layernorm`` de ``TransformerBlock``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from comum import (
    CATEGORICA,
    MARCADORES,
    cabecalho,
    estilizar_linhas,
    imprimir_tabela,
    lote_do_corpus,
    novo_modelo,
    salvar_figura,
    salvar_tabela,
)

VARIANTES = {
    "completo": dict(usar_residual=True, usar_layernorm=True),
    "sem residual": dict(usar_residual=False, usar_layernorm=True),
    "sem LayerNorm": dict(usar_residual=True, usar_layernorm=False),
    "sem os dois": dict(usar_residual=False, usar_layernorm=False),
}
PROFUNDIDADES = [2, 4, 8, 16, 24]
BASE = dict(emb_dim=128, n_heads=4, context_length=128, drop_rate=0.0)


def _medir(n_layers: int, chaves: dict, entradas, alvos) -> dict:
    modelo = novo_modelo(n_layers=n_layers, **BASE, **chaves).train()
    desvios: list[float] = []
    ganchos = [
        bloco.register_forward_hook(lambda m, e, s: desvios.append(float(s.detach().std())))
        for bloco in modelo.trf_blocks
    ]
    logits = modelo(entradas)
    perda = F.cross_entropy(logits.flatten(0, 1), alvos.flatten())
    perda.backward()
    for g in ganchos:
        g.remove()

    gradientes = [
        float(torch.sqrt(sum(p.grad.pow(2).sum() for p in bloco.parameters())))
        for bloco in modelo.trf_blocks
    ]
    return {"perda": perda.item(), "desvios": desvios, "gradientes": gradientes}


def executar() -> list[dict]:
    cabecalho("Experimento 07 — conexões residuais e LayerNorm")
    entradas, alvos = lote_do_corpus(batch=4, num_tokens=64)

    linhas: list[dict] = []
    perfis: dict[str, dict] = {}
    for nome, chaves in VARIANTES.items():
        for n in PROFUNDIDADES:
            r = _medir(n, chaves, entradas, alvos)
            if n == PROFUNDIDADES[-1]:
                perfis[nome] = r
            linhas.append(
                {
                    "variante": nome,
                    "n_layers": n,
                    "perda_inicial": r["perda"],
                    "desvio_primeiro_bloco": r["desvios"][0],
                    "desvio_ultimo_bloco": r["desvios"][-1],
                    "grad_primeiro_bloco": r["gradientes"][0],
                    "grad_ultimo_bloco": r["gradientes"][-1],
                    "razao_grad_primeiro_ultimo": r["gradientes"][0] / max(r["gradientes"][-1], 1e-30),
                }
            )

    imprimir_tabela(linhas)
    salvar_tabela(linhas, "exp07-residuais-e-layernorm")
    _graficos(perfis, linhas)
    return linhas


def _graficos(perfis: dict[str, dict], linhas: list[dict]) -> None:
    fig, eixos = plt.subplots(1, 3, figsize=(13, 3.9))
    for i, (nome, r) in enumerate(perfis.items()):
        blocos = range(1, len(r["desvios"]) + 1)
        estilo = dict(color=CATEGORICA[i], marker=MARCADORES[i], markersize=3, label=nome)
        eixos[0].plot(blocos, r["desvios"], **estilo)
        eixos[1].plot(blocos, [max(g, 1e-30) for g in r["gradientes"]], **estilo)
        serie = [l for l in linhas if l["variante"] == nome]
        eixos[2].plot([l["n_layers"] for l in serie], [l["perda_inicial"] for l in serie],
                      color=CATEGORICA[i], marker=MARCADORES[i], label=nome)

    eixos[0].set_yscale("log")
    eixos[0].legend(fontsize=7)
    estilizar_linhas(eixos[0], f"Desvio das ativações por bloco\n({PROFUNDIDADES[-1]} blocos)",
                     "bloco", "desvio-padrão (log)")
    eixos[1].set_yscale("log")
    estilizar_linhas(eixos[1], f"Norma do gradiente por bloco\n({PROFUNDIDADES[-1]} blocos)",
                     "bloco", "‖gradiente‖ (log)")
    eixos[2].axhline(10.825, color="#52514e", linestyle="--", linewidth=1)
    eixos[2].set_yscale("log")
    estilizar_linhas(eixos[2], "Perda inicial (entropia cruzada)\n--- ln(50257): previsão uniforme",
                     "n_layers", "perda (log)")

    fig.suptitle("Experimento 07 — o papel das conexões residuais e da LayerNorm", y=1.04)
    fig.tight_layout()
    salvar_figura(fig, "exp07-residuais-e-layernorm")


if __name__ == "__main__":
    executar()
