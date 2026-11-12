"""Treinamento e avaliação do GPT — Sprint 5 (Capítulo 5 do livro-texto).

Avaliação quantitativa do modelo pela perda de entropia cruzada (seção 5.1).

O modelo devolve, para cada posição da entrada, ``V`` logits — uma pontuação
para cada token do vocabulário ser o próximo. A tarefa é prever o token
seguinte, que é exatamente o alvo produzido pela janela deslizante da Sprint 2
(a entrada deslocada de uma posição).

A **entropia cruzada** mede o quanto essa previsão erra: para cada posição,
transforma os logits em probabilidades com softmax e toma ``-log`` da
probabilidade atribuída ao token correto. A perda do lote é a média desse valor
sobre todas as ``b · L`` posições::

    logits (b, L, V) -> flatten -> (b·L, V)
    alvos  (b, L)    -> flatten -> (b·L,)
    perda = média( -log softmax(logits)[alvo] )

Um modelo que acerta com certeza tem perda 0. Um modelo que distribui a
probabilidade uniformemente entre os ``V`` tokens tem perda ``ln(V)`` — para o
vocabulário do GPT-2, ``ln(50257) ≈ 10,82``. É perto disso que um modelo recém
inicializado começa.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["perda_do_lote"]


def perda_do_lote(
    entrada: torch.Tensor,
    alvo: torch.Tensor,
    modelo: nn.Module,
    dispositivo: torch.device | str,
) -> torch.Tensor:
    """Perda de entropia cruzada de um lote — ``calc_loss_batch`` da seção 5.1.

    ``entrada`` e ``alvo`` têm forma ``(b, L)`` com token IDs. Devolve um tensor
    escalar ainda ligado ao grafo de computação, para que o laço de treinamento
    possa chamar ``backward()`` sobre ele.
    """
    entrada = entrada.to(dispositivo)
    alvo = alvo.to(dispositivo)
    logits = modelo(entrada)  # (b, L, V)
    return F.cross_entropy(logits.flatten(0, 1), alvo.flatten())
