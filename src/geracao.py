"""Inferência e geração autorregressiva — Sprint 4 (Capítulo 4 e seção 5.3).

O ``GPTModel`` só sabe fazer uma coisa: dado um trecho de texto, produzir para
cada posição uma pontuação (logit) para cada token do vocabulário ser o
próximo. A geração de texto é um laço em volta dessa operação:

1. passar o contexto atual pelo modelo -> logits ``(b, L, V)``;
2. ficar só com os logits da **última** posição -> ``(b, V)``: é ela que
   prevê o token seguinte ao texto inteiro;
3. transformar os logits em uma escolha de token (``selecionar_proximo_token``);
4. anexar o token escolhido ao contexto e voltar ao passo 1.

Relação entre logits, probabilidades e a escolha do token:

- **logits** são pontuações reais, sem escala definida;
- o **softmax** os converte em uma distribuição de probabilidade, positiva e
  que soma 1, preservando a ordem (o maior logit vira a maior probabilidade);
- a **seleção** escolhe um token a partir dessa distribuição: o mais provável
  (*greedy*, ``temperatura=0``) ou um sorteio proporcional às probabilidades
  (``temperatura > 0``), opcionalmente restrito aos ``top_k`` mais prováveis.

A temperatura divide os logits antes do softmax: abaixo de 1 a distribuição
fica mais concentrada (texto mais previsível), acima de 1 fica mais espalhada
(texto mais variado). O top-k descarta os tokens fora dos ``k`` mais
prováveis, impedindo que o sorteio escolha opções muito improváveis.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from tokenizacao import TokenizadorBPE

__all__ = [
    "texto_para_ids",
    "ids_para_texto",
    "ajustar_logits",
    "selecionar_proximo_token",
    "gerar",
    "gerar_texto_simples",
]


def texto_para_ids(texto: str, tokenizador=None) -> torch.Tensor:
    """Texto -> tensor ``(1, num_tokens)``: um lote com uma única sequência."""
    tokenizador = tokenizador or TokenizadorBPE()
    return torch.tensor(tokenizador.encode(texto)).unsqueeze(0)


def ids_para_texto(ids: torch.Tensor, tokenizador=None) -> str:
    """Tensor ``(1, num_tokens)`` ou ``(num_tokens,)`` -> texto."""
    tokenizador = tokenizador or TokenizadorBPE()
    return tokenizador.decode(ids.reshape(-1).tolist())


def ajustar_logits(
    logits: torch.Tensor,
    temperatura: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Aplica top-k e temperatura aos logits da última posição, ``(b, V)``.

    - **top-k**: os logits abaixo do ``k``-ésimo maior viram ``-inf``; depois
      do softmax, a probabilidade desses tokens é exatamente zero;
    - **temperatura**: divide os logits. Como o softmax depende das diferenças
      entre logits, dividir por ``T > 1`` aproxima as probabilidades e dividir
      por ``T < 1`` as afasta.

    ``temperatura`` deve ser positiva; a escolha *greedy* não passa por aqui.
    """
    if top_k is not None:
        top_k = min(top_k, logits.shape[-1])
        limiar = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < limiar, -torch.inf)
    return logits / temperatura


def selecionar_proximo_token(
    logits: torch.Tensor,
    temperatura: float = 0.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Escolhe o próximo token a partir dos logits da última posição.

    Args:
        logits: ``(b, V)``.
        temperatura: ``0`` -> escolha *greedy* (``argmax``), determinística;
            ``> 0`` -> sorteio de ``softmax(logits / temperatura)``.
        top_k: se definido, restringe a escolha aos ``top_k`` tokens mais
            prováveis.

    Returns:
        IDs escolhidos, ``(b, 1)``.

    No modo *greedy* não é preciso calcular o softmax: ele preserva a ordem dos
    logits, então o maior logit é também a maior probabilidade.
    """
    if temperatura <= 0.0:
        if top_k is not None:
            logits = ajustar_logits(logits, 1.0, top_k)
        return torch.argmax(logits, dim=-1, keepdim=True)

    probabilidades = torch.softmax(ajustar_logits(logits, temperatura, top_k), dim=-1)
    return torch.multinomial(probabilidades, num_samples=1)


@torch.no_grad()
def gerar(
    modelo: nn.Module,
    ids: torch.Tensor,
    max_novos_tokens: int,
    tamanho_contexto: int,
    temperatura: float = 0.0,
    top_k: int | None = None,
    id_fim: int | None = None,
) -> torch.Tensor:
    """Geração autorregressiva — listagem 4.8, com as estratégias da seção 5.3.

    Args:
        modelo: um ``GPTModel``, em modo ``eval()``.
        ids: contexto inicial, ``(b, num_tokens)``.
        max_novos_tokens: quantos tokens acrescentar.
        tamanho_contexto: o ``context_length`` do modelo. O contexto é
            truncado aos últimos ``tamanho_contexto`` tokens antes de cada
            passo, porque o modelo não aceita sequências maiores — os tokens
            mais antigos deixam de influenciar a previsão.
        temperatura, top_k: estratégia de seleção (``selecionar_proximo_token``).
        id_fim: se definido, a geração para quando todas as sequências do lote
            produzem esse token (por exemplo, ``<|endoftext|>`` = 50256).

    Returns:
        ``(b, num_tokens + n)`` com ``n <= max_novos_tokens``: o contexto inicial
        seguido dos tokens gerados.
    """
    if modelo.training:
        warnings.warn(
            "o modelo está em modo de treino: o dropout está ativo e a geração "
            "não é determinística. Chame modelo.eval() antes de gerar.",
            stacklevel=2,
        )

    for _ in range(max_novos_tokens):
        contexto = ids[:, -tamanho_contexto:]       # (b, ≤ tamanho_contexto)
        logits = modelo(contexto)                    # (b, L, V)
        logits = logits[:, -1, :]                    # (b, V): só a última posição
        proximo = selecionar_proximo_token(logits, temperatura, top_k)  # (b, 1)
        if id_fim is not None and bool((proximo == id_fim).all()):
            break
        ids = torch.cat((ids, proximo), dim=1)       # (b, num_tokens + 1)
    return ids


def gerar_texto_simples(
    modelo: nn.Module, ids: torch.Tensor, max_novos_tokens: int, tamanho_contexto: int
) -> torch.Tensor:
    """Geração *greedy* do livro (listagem 4.8): sempre o token mais provável."""
    return gerar(modelo, ids, max_novos_tokens, tamanho_contexto, temperatura=0.0)
