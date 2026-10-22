"""Embeddings de token e posicionais — Sprint 2 (Capítulo 2 do livro-texto).

Última etapa do processamento de entrada: os token IDs, que são apenas índices,
viram vetores densos que as camadas seguintes conseguem processar (seções 2.7
e 2.8).

Fluxo::

    token IDs (b, num_tokens)
      -> embedding de token      (b, num_tokens, d_emb)
      +  embedding posicional    (num_tokens, d_emb)     difundido sobre o lote
      =  embeddings de entrada   (b, num_tokens, d_emb)  -> atenção (Sprint 3)

A soma é a forma de *absolute positional embedding* usada no GPT-2: cada
posição ``0 .. context_length-1`` tem o seu próprio vetor treinável.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["EmbeddingDeEntrada"]


class EmbeddingDeEntrada(nn.Module):
    """Camada de entrada do modelo: embedding de token + embedding posicional.

    **Embedding de token.** Um ``nn.Embedding(vocab_size, d_emb)`` é uma matriz
    de pesos ``(vocab_size, d_emb)`` usada como tabela de consulta: o token de
    ID *k* recebe a linha *k*. É equivalente a multiplicar um vetor one-hot pela
    matriz, mas sem materializar o one-hot. Os valores começam aleatórios e são
    ajustados no treinamento (Sprint 5), quando tokens de uso parecido passam a
    ter vetores próximos.

    **Embedding posicional.** O embedding de token depende só da identidade do
    token: ``"the"`` na posição 0 e na posição 7 recebe exatamente o mesmo
    vetor. Como a atenção da Sprint 3 também não enxerga a ordem, sem esta
    segunda tabela *"o gato mordeu o cão"* e *"o cão mordeu o gato"* seriam
    indistinguíveis. Somar um vetor por posição torna a representação de um
    token dependente de onde ele aparece.

    A ordem de criação das duas tabelas (token, depois posição) é mantida fixa,
    pois define o consumo do gerador aleatório e, portanto, a
    reprodutibilidade dos experimentos a partir de uma semente.

    Args:
        vocab_size: número de linhas da tabela de tokens (50.257 no GPT-2).
        d_emb: dimensão de cada vetor.
        context_length: maior sequência suportada — número de linhas da tabela
            posicional.
        usar_posicional: ``False`` omite a soma posicional (usado nos
            experimentos que mostram por que ela é necessária).
    """

    def __init__(
        self,
        vocab_size: int,
        d_emb: int,
        context_length: int,
        usar_posicional: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_emb = d_emb
        self.context_length = context_length
        self.usar_posicional = usar_posicional
        self.token_emb = nn.Embedding(vocab_size, d_emb)
        self.pos_emb = nn.Embedding(context_length, d_emb)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """``(b, num_tokens)`` ou ``(num_tokens,)`` -> ``(..., num_tokens, d_emb)``."""
        num_tokens = token_ids.shape[-1]
        if num_tokens > self.context_length:
            raise ValueError(
                f"sequência com {num_tokens} tokens excede o context_length "
                f"de {self.context_length}."
            )

        embeddings = self.token_emb(token_ids)
        if self.usar_posicional:
            posicoes = torch.arange(num_tokens, device=token_ids.device)
            embeddings = embeddings + self.pos_emb(posicoes)
        return embeddings
