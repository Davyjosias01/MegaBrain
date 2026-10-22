"""Preparação das sequências de treinamento — Sprint 2 (Capítulo 2 do livro-texto).

Um LLM é treinado para prever o próximo token. Este módulo transforma a longa
sequência de token IDs de um corpus em muitos pares ``(entrada, alvo)`` de
tamanho fixo e os entrega em lotes por meio de um ``DataLoader`` (seção 2.6).

Fluxo::

    texto -> tokenizador -> token IDs -> janela deslizante -> pares (entrada, alvo)
          -> Dataset -> DataLoader -> lotes (b, max_length)

O alvo é a entrada deslocada de uma posição. Em cada par, a posição *i* da
entrada vê os tokens ``0..i`` e deve prever o token ``i+1`` — por isso um par de
``max_length`` tokens contém, na verdade, ``max_length`` tarefas de previsão.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from tokenizacao import TokenizadorBPE

RAIZ = Path(__file__).resolve().parent.parent
CORPUS_PADRAO = RAIZ / "sprints" / "the-verdict.txt"

__all__ = [
    "CORPUS_PADRAO",
    "carregar_texto",
    "pares_entrada_alvo",
    "numero_de_amostras",
    "GPTDatasetV1",
    "criar_dataloader",
]


def carregar_texto(caminho: Path | str = CORPUS_PADRAO) -> str:
    """Lê o corpus em UTF-8.

    O corpus padrão é ``sprints/the-verdict.txt``, o conto de Edith Wharton
    (domínio público) usado no Capítulo 2. Fica versionado no repositório para
    que nenhum experimento dependa de download.
    """
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"corpus não encontrado em {caminho}. "
            "O arquivo the-verdict.txt é o corpus usado desde a Sprint 2."
        )
    return caminho.read_text(encoding="utf-8")


def pares_entrada_alvo(
    token_ids: Sequence[int], tamanho_contexto: int
) -> list[tuple[list[int], int]]:
    """Lista as tarefas de previsão contidas em uma janela de ``tamanho_contexto``.

    Versão explícita, para inspeção, do que um único par ``(entrada, alvo)``
    representa (seção 2.6): com contexto 4, ``[a] -> b``, ``[a, b] -> c``,
    ``[a, b, c] -> d`` e ``[a, b, c, d] -> e``.
    """
    if len(token_ids) <= tamanho_contexto:
        raise ValueError(
            f"são necessários pelo menos {tamanho_contexto + 1} tokens, "
            f"recebidos {len(token_ids)}."
        )
    return [
        (list(token_ids[:i]), int(token_ids[i])) for i in range(1, tamanho_contexto + 1)
    ]


def numero_de_amostras(num_tokens: int, max_length: int, stride: int) -> int:
    """Quantos pares a janela deslizante produz a partir de ``num_tokens`` tokens.

    A janela começa nas posições ``0, stride, 2·stride, …`` enquanto ainda
    houver um token depois dela para servir de alvo, ou seja, enquanto
    ``inicio < num_tokens - max_length``. Isso dá
    ``ceil((num_tokens - max_length) / stride)`` amostras.

    A fórmula explicita os dois botões: aumentar o contexto reduz pouco o número
    de amostras; aumentar o ``stride`` o reduz na mesma proporção.
    """
    if stride < 1:
        raise ValueError("stride deve ser pelo menos 1.")
    return len(range(0, num_tokens - max_length, stride))


class GPTDatasetV1(Dataset):
    """Dataset de janela deslizante sobre um corpus tokenizado.

    Percorre os token IDs em passos de ``stride``, recortando pares
    ``(entrada, alvo)`` de ``max_length`` tokens, em que o alvo é a entrada
    deslocada de uma posição.

    - ``stride == max_length``: janelas encostadas, sem sobreposição — cada
      token é alvo uma única vez por época;
    - ``stride < max_length``: janelas sobrepostas — mais amostras, mas
      correlacionadas entre si (risco de sobreajuste);
    - ``stride > max_length``: tokens entre as janelas são descartados.

    Args:
        texto: corpus completo.
        tokenizador: qualquer objeto com ``encode(texto) -> list[int]`` — em
            geral ``TokenizadorBPE`` ou ``TokenizadorSimples``.
        max_length: tamanho do contexto, em tokens.
        stride: passo da janela deslizante.
    """

    def __init__(self, texto: str, tokenizador, max_length: int, stride: int) -> None:
        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []

        token_ids = tokenizador.encode(texto)
        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i : i + max_length]))
            self.target_ids.append(torch.tensor(token_ids[i + 1 : i + max_length + 1]))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


def criar_dataloader(
    texto: str,
    batch_size: int = 8,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = False,
    drop_last: bool = True,
    num_workers: int = 0,
    tokenizador=None,
) -> DataLoader:
    """DataLoader sobre o dataset de janela deslizante.

    O ``DataLoader`` agrupa as amostras em lotes ``(batch_size, max_length)``,
    embaralha a ordem a cada época se ``shuffle=True`` e, com ``drop_last``,
    descarta o último lote quando incompleto — um lote menor produziria uma
    perda calculada sobre menos exemplos, com gradiente mais ruidoso.

    Args:
        tokenizador: padrão ``TokenizadorBPE()`` (GPT-2), como no restante do projeto.
    """
    tokenizador = tokenizador or TokenizadorBPE()
    dataset = GPTDatasetV1(texto, tokenizador, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
