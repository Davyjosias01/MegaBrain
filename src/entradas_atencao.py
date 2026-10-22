"""Ponte entre a Sprint 2 e a Sprint 3: entradas reais para os mecanismos de atenção.

Os mecanismos de ``src/attention.py`` recebem tensores de embeddings, e não texto.
Este módulo encadeia os componentes da Sprint 2 — tokenização BPE
(``tokenizacao.py``), janela deslizante e DataLoader (``dados.py``), embeddings
de token e posicionais (``embeddings.py``) — para produzir exatamente esses
tensores a partir de ``sprints/the-verdict.txt``, o mesmo corpus do Capítulo 2.

O objetivo é apenas alimentar os experimentos da Sprint 3 com dados reais: nada
aqui é treinado. As matrizes de embedding são inicializadas aleatoriamente com
uma semente fixa, o que mantém os experimentos reprodutíveis.

Fluxo::

    texto -> TokenizadorBPE -> token IDs -> janela deslizante -> lote de IDs
          -> EmbeddingDeEntrada (token + posição) -> (b, num_tokens, d_in)
"""

from __future__ import annotations

from pathlib import Path

import torch

from dados import CORPUS_PADRAO, GPTDatasetV1, carregar_texto, criar_dataloader
from embeddings import EmbeddingDeEntrada
from tokenizacao import TokenizadorBPE

__all__ = [
    "CORPUS_PADRAO",
    "GPTDatasetV1",
    "carregar_texto",
    "criar_dataloader",
    "lote_de_embeddings",
    "embutir_sequencia",
    "tokens_legiveis",
]


def _camada_de_entrada(
    d_in: int,
    context_length: int,
    vocab_size: int,
    seed: int,
    usar_posicional: bool = True,
) -> EmbeddingDeEntrada:
    torch.manual_seed(seed)
    return EmbeddingDeEntrada(vocab_size, d_in, context_length, usar_posicional)


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
    tokenizador = TokenizadorBPE()
    carregador = criar_dataloader(
        texto,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride or max_length,
        shuffle=False,
        drop_last=True,
        tokenizador=tokenizador,
    )
    entradas, _ = next(iter(carregador))

    camada = _camada_de_entrada(d_in, max_length, tokenizador.tamanho_vocabulario, seed)
    with torch.no_grad():
        embeddings = camada(entradas)

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
    tokenizador = TokenizadorBPE()
    ids = torch.tensor(tokenizador.encode(texto)).unsqueeze(0)
    context_length = context_length or ids.shape[1]

    camada = _camada_de_entrada(
        d_in, context_length, tokenizador.tamanho_vocabulario, seed, usar_posicional
    )
    with torch.no_grad():
        embeddings = camada(ids)

    return {
        "embeddings": embeddings,
        "token_ids": ids,
        "tokens": tokens_legiveis(ids[0]),
    }


def tokens_legiveis(token_ids: torch.Tensor) -> list[str]:
    """Decodifica cada token ID isoladamente, para rotular eixos de gráficos."""
    return [t.strip() or "·" for t in TokenizadorBPE().ids_para_tokens(token_ids)]
