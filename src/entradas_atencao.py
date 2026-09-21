"""Ponte entre a Sprint 2 e a Sprint 3: entradas reais para os mecanismos de atenção.

Os mecanismos de ``src/attention.py`` recebem tensores de embeddings, e não texto.
Este módulo refaz o caminho construído na Sprint 2 — tokenização BPE, janela
deslizante de pares entrada/alvo, embeddings de token e embeddings posicionais —
para produzir exatamente esses tensores a partir de ``sprints/the-verdict.txt``,
o mesmo corpus usado no Capítulo 2.

O objetivo é apenas alimentar os experimentos da Sprint 3 com dados reais: nada
aqui é treinado. As matrizes de embedding são inicializadas aleatoriamente com
uma semente fixa, o que mantém os experimentos reprodutíveis.

Fluxo::

    texto -> tiktoken (BPE) -> token IDs -> janela deslizante -> lote de IDs
          -> nn.Embedding (token) + nn.Embedding (posição) -> (b, num_tokens, d_in)
"""

from __future__ import annotations

from pathlib import Path

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

RAIZ = Path(__file__).resolve().parent.parent
CORPUS_PADRAO = RAIZ / "sprints" / "the-verdict.txt"

__all__ = [
    "CORPUS_PADRAO",
    "GPTDatasetV1",
    "carregar_texto",
    "criar_dataloader",
    "lote_de_embeddings",
    "embutir_sequencia",
    "tokens_legiveis",
]


def carregar_texto(caminho: Path | str = CORPUS_PADRAO) -> str:
    """Lê o corpus em UTF-8."""
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"corpus não encontrado em {caminho}. "
            "O arquivo the-verdict.txt é o mesmo usado na Sprint 2."
        )
    return caminho.read_text(encoding="utf-8")


class GPTDatasetV1(Dataset):
    """Dataset de janela deslizante da Sprint 2.

    Percorre a sequência de token IDs em passos de ``stride``, recortando pares
    ``(entrada, alvo)`` de ``max_length`` tokens, em que o alvo é a entrada
    deslocada de uma posição. Na Sprint 3 só a entrada é usada — o alvo voltará
    a ser necessário no treinamento da Sprint 5.
    """

    def __init__(self, texto: str, tokenizador, max_length: int, stride: int) -> None:
        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []

        token_ids = tokenizador.encode(texto, allowed_special={"<|endoftext|>"})
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
) -> DataLoader:
    """DataLoader sobre o dataset de janela deslizante."""
    tokenizador = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(texto, tokenizador, max_length, stride)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )


def _camadas_de_embedding(
    d_in: int, context_length: int, vocab_size: int, seed: int
) -> tuple[nn.Embedding, nn.Embedding]:
    torch.manual_seed(seed)
    token_emb = nn.Embedding(vocab_size, d_in)
    pos_emb = nn.Embedding(context_length, d_in)
    return token_emb, pos_emb


def lote_de_embeddings(
    d_in: int,
    max_length: int = 16,
    batch_size: int = 4,
    stride: int | None = None,
    seed: int = 123,
    caminho: Path | str = CORPUS_PADRAO,
) -> dict:
    """Produz um lote de embeddings pronto para os mecanismos de atenção.

    Args:
        d_in: dimensão do embedding de cada token.
        max_length: número de tokens por sequência.
        batch_size: número de sequências no lote.
        stride: passo da janela deslizante (padrão: ``max_length``, sem sobreposição).
        seed: semente das matrizes de embedding.
        caminho: corpus de origem.

    Returns:
        Dicionário com ``embeddings`` ``(batch_size, max_length, d_in)``,
        ``token_ids`` ``(batch_size, max_length)`` e ``tokens``, a lista de
        strings da primeira sequência do lote (útil para rotular eixos).
    """
    texto = carregar_texto(caminho)
    carregador = criar_dataloader(
        texto,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride or max_length,
        shuffle=False,
        drop_last=True,
    )
    entradas, _ = next(iter(carregador))

    tokenizador = tiktoken.get_encoding("gpt2")
    token_emb, pos_emb = _camadas_de_embedding(
        d_in, max_length, tokenizador.n_vocab, seed
    )

    with torch.no_grad():
        posicoes = torch.arange(max_length)
        embeddings = token_emb(entradas) + pos_emb(posicoes)

    return {
        "embeddings": embeddings,
        "token_ids": entradas,
        "tokens": tokens_legiveis(entradas[0]),
    }


def embutir_sequencia(
    texto: str,
    d_in: int,
    seed: int = 123,
    context_length: int | None = None,
    usar_posicional: bool = True,
) -> dict:
    """Mesma operação de ``lote_de_embeddings``, mas para um texto avulso.

    Usado no experimento que compara diferentes sequências de entrada.
    ``usar_posicional=False`` omite o embedding posicional da Sprint 2, deixando
    apenas o embedding de token — útil para mostrar que o mecanismo de atenção,
    sozinho, não enxerga a ordem dos tokens.
    """
    tokenizador = tiktoken.get_encoding("gpt2")
    ids = torch.tensor(tokenizador.encode(texto)).unsqueeze(0)
    num_tokens = ids.shape[1]
    context_length = context_length or num_tokens

    token_emb, pos_emb = _camadas_de_embedding(
        d_in, context_length, tokenizador.n_vocab, seed
    )
    with torch.no_grad():
        embeddings = token_emb(ids)
        if usar_posicional:
            embeddings = embeddings + pos_emb(torch.arange(num_tokens))

    return {
        "embeddings": embeddings,
        "token_ids": ids,
        "tokens": tokens_legiveis(ids[0]),
    }


def tokens_legiveis(token_ids: torch.Tensor) -> list[str]:
    """Decodifica cada token ID isoladamente, para rotular eixos de gráficos."""
    tokenizador = tiktoken.get_encoding("gpt2")
    return [tokenizador.decode([int(i)]).strip() or "·" for i in token_ids]
